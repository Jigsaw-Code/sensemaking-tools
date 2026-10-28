# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for System One judgments. The transport is fake; Clef is not called."""

import asyncio
import http.client
import io
import json
import unittest
from unittest import mock

import pandas as pd

from src.get_gemini_scores_lib import ContentScorer
from src.models import decision
from src.models import custom_types
from src.models import systemone
from src.simulated_jury import simulated_jury
from src.tasks import categorization


def _answers_for(payload: dict) -> dict:
  """Builds deterministic fake answers for a request payload."""
  answers = {}
  state = payload["state"]
  state_text = state if isinstance(state, str) else json.dumps(state)
  for key, question in payload["questions"].items():
    kind = question["type"]
    if kind == "noul":
      instructions = question["instructions"]
      if "effectively equivalent" in instructions:
        sides = instructions.split("B (", 1)
        probability = (
            0.95 if "respect" in sides[0] and "respect" in sides[-1] else 0.05
        )
      elif "Buses" in instructions and "bus" in state_text.lower():
        probability = 0.92
      elif "More buses" in instructions and "bus" in state_text.lower():
        probability = 0.91
      else:
        probability = 0.08
      answers[key] = {"type": "noul", "noul": probability}
    elif kind == "choice":
      chosen = next(
          (
              option
              for option, description in question["criteria"].items()
              if "WIN" in str(description)
          ),
          next(iter(question["criteria"])),
      )
      probabilities = {
          option: 1.0 if option == chosen else 0.0
          for option in question["criteria"]
      }
      answers[key] = {
          "type": "choice",
          "choice": chosen,
          "probabilities": probabilities,
          "confidence": 0.8,
      }
    else:
      level = 0 if "minimality" in question["instructions"] else len(
          question["criteria"]
      ) - 1
      answers[key] = {
          "type": "score",
          "score": float(level),
          "probabilities": {
              str(index): 1.0 if index == level else 0.0
              for index in range(len(question["criteria"]))
          },
          "confidence": 1.0,
      }
  return answers


class RecordingTransport:
  """Fake transport that records request bodies and answers them."""

  def __init__(self):
    """Starts with no recorded bodies."""
    self.bodies = []

  def __call__(self, url: str, body: bytes) -> bytes:
    """Records the body and returns fake answers for it."""
    payload = json.loads(body)
    self.bodies.append(payload)
    encoded = {
        "model": payload["model"],
        "answers": _answers_for(payload),
        "usage": {"input_tokens": 4, "output_tokens": 0},
    }
    return json.dumps(encoded).encode("utf-8")


def _client() -> tuple[systemone.SystemOneClient, RecordingTransport]:
  """Returns a client wired to a recording fake transport."""
  transport = RecordingTransport()
  return (
      systemone.SystemOneClient(
          model="clef", transport=transport, max_concurrent=1
      ),
      transport,
  )


class DecisionTests(unittest.TestCase):
  """Tests for System One judgments and the client."""

  def test_assigns_matching_topic_and_never_asks_about_other(self):
    """Tests that only specific labels are asked and the match is kept."""
    client, transport = _client()

    chosen = asyncio.run(
        decision.assign_labels(
            client,
            "The city should add more bus lanes.",
            ["Buses", "Parking", "Other"],
            kind="topic",
            cutoff=0.5,
        )
    )

    self.assertEqual(chosen, ["Buses"])
    asked = [
        question["instructions"]
        for body in transport.bodies
        for question in body["questions"].values()
    ]
    self.assertEqual(len(asked), 2)
    self.assertFalse(any("'Other'" in text for text in asked))

  def test_assigns_other_when_no_label_clears_the_cutoff(self):
    """Tests that Other is assigned only when nothing clears the cutoff."""
    client, _ = _client()

    chosen = asyncio.run(
        decision.assign_labels(
            client,
            "The weather was fine.",
            ["Parking", "Other"],
            kind="topic",
            cutoff=0.5,
        )
    )

    self.assertEqual(chosen, ["Other"])

  def test_rubric_is_a_score_question_read_at_its_most_likely_level(self):
    """Tests that the rubric uses the most likely level, not the mean."""
    bodies = []

    def transport(url: str, body: bytes) -> bytes:
      """Returns a canned System One response."""
      bodies.append(json.loads(body))
      return json.dumps({
          "model": "clef",
          "answers": {
              "rubric": {
                  "type": "score",
                  "score": 2.4,
                  "probabilities": {"0": 0.4, "4": 0.6},
                  "confidence": 0.5,
              }
          },
      }).encode("utf-8")

    client = systemone.SystemOneClient(model="clef", transport=transport)
    level = asyncio.run(decision.rubric_level(client, "An answer to rate."))

    self.assertEqual(level, 4)
    question = bodies[0]["questions"]["rubric"]
    self.assertEqual(question["type"], "score")
    self.assertEqual(question["criteria"], list(decision.RUBRIC))

  def test_clusters_only_equivalent_pairs(self):
    """Tests that only equivalent propositions are clustered."""
    client, _ = _client()

    sets = asyncio.run(
        decision.equivalence_sets(
            client,
            {
                "0:1": "Everyone should be treated with respect.",
                "1:2": "All people deserve respect.",
                "2:1": "Add more bus lanes downtown.",
            },
            cutoff=0.5,
        )
    )

    self.assertEqual(sets, [["0:1", "1:2"]])

  def test_equivalence_asks_a_few_pairs_per_request(self):
    """Tests that equivalence pairs are split into small requests."""
    client, transport = _client()
    items = {
        "a": "Everyone should be treated with respect.",
        "b": "All people deserve respect.",
        "c": "Add more bus lanes downtown.",
        "d": "Lower property taxes.",
        "e": "Fix the potholes on 5th Street.",
    }

    sets = asyncio.run(decision.equivalence_sets(client, items, cutoff=0.5))

    self.assertEqual(sets, [["a", "b"]])
    self.assertEqual(
        sorted(len(body["questions"]) for body in transport.bodies),
        [2, 4, 4],
    )

  def test_transient_failures_are_retried(self):
    """Tests that a 5xx is retried and a 4xx is raised at once."""
    calls = []
    ok = json.dumps({
        "model": "clef",
        "answers": {"q": {"type": "noul", "noul": 0.5}},
    }).encode("utf-8")
    question = {
        "q": systemone.NoulQuestion(instructions="Yes?", true="Yes", false="No")
    }
    for status, expected_calls, succeeds in ((502, 2, True), (400, 1, False)):
      with self.subTest(status=status):
        calls.clear()

        def transport(url: str, body: bytes, status=status) -> bytes:
          """Returns a canned System One response."""
          calls.append(url)
          if len(calls) == 1:
            raise systemone.SystemOneError(
                systemone.FailureKind.HTTP, "upstream", status=status
            )
          return ok

        client = systemone.SystemOneClient(
            model="clef", transport=transport, retry_delay_seconds=0
        )
        if succeeds:
          answers = asyncio.run(client.ask("state", question))
          self.assertEqual(answers["q"].noul, 0.5)
        else:
          with self.assertRaises(systemone.SystemOneError):
            asyncio.run(client.ask("state", question))
        self.assertEqual(len(calls), expected_calls)

  def test_tournament_selects_marked_option_past_26(self):
    """Tests the choice tournament beyond 26 options."""
    client, transport = _client()
    options = [(f"id-{index}", f"option {index}") for index in range(27)]
    options[20] = ("id-20", "WIN option")

    winner = asyncio.run(
        decision.choose_one(
            client,
            "Pick one.",
            options,
            "Which option wins?",
        )
    )

    self.assertEqual(winner, "id-20")
    self.assertGreaterEqual(len(transport.bodies), 2)

  def test_packs_questions_past_the_64_limit(self):
    """Tests packing at the default 64-question limit."""
    client, transport = _client()
    questions = {
        f"q{index}": systemone.NoulQuestion(
            instructions=f"Is this item {index}?",
            true="Yes",
            false="No",
        )
        for index in range(65)
    }

    answers = asyncio.run(client.ask("a short state", questions))

    self.assertEqual(len(answers), 65)
    self.assertEqual(len(transport.bodies), 2)
    self.assertEqual(len(transport.bodies[0]["questions"]), 64)

  def test_rejects_an_error_payload(self):
    """Tests that an error payload raises an HTTP failure."""
    def transport(url: str, body: bytes) -> bytes:
      """Returns a canned System One response."""
      return b'{"error": "clef does not support chat"}'

    client = systemone.SystemOneClient(model="clef", transport=transport)
    with self.assertRaises(systemone.SystemOneError) as caught:
      asyncio.run(
          client.ask(
              "state",
              {
                  "q": systemone.NoulQuestion(
                      instructions="Yes?", true="Yes", false="No"
                  )
              },
          )
      )
    self.assertEqual(caught.exception.kind, systemone.FailureKind.HTTP)

  def test_scores_attribute_as_probability(self):
    """Tests that attributes are scored as probabilities."""
    client, _ = _client()

    scores = asyncio.run(
        decision.score_attributes(
            client,
            "You are an idiot.",
            [
                decision.AttributeSpec(
                    name="TOXICITY",
                    label="Toxicity",
                    definition="A rude comment.",
                )
            ],
        )
    )

    self.assertEqual(scores, {"TOXICITY": 0.08})

  def test_topic_assignment_does_not_call_the_generative_model(self):
    """Tests that topic assignment uses only the decision model."""
    client, transport = _client()
    statement = custom_types.Statement(id="s1", text="Please add bus lanes.")
    with decision.use_decision_client(client):
      records = asyncio.run(
          categorization._process_topic_categorization(
              [statement],
              model=object(),
              target_topics=[
                  custom_types.FlatTopic(name="Buses"),
                  custom_types.FlatTopic(name="Other"),
              ],
          )
      )

    self.assertEqual([topic.name for topic in records[0].topics], ["Buses"])
    self.assertTrue(transport.bodies)
    self.assertEqual(transport.bodies[0]["model"], "clef")

  def test_failing_opinion_autorater_becomes_other(self):
    """Tests that a failing autorater finalizes the quote as Other."""
    client, _ = _client()
    statement = custom_types.Statement(
        id="s1",
        text="Please add bus lanes.",
        quotes=[
            custom_types.Quote(
                id="q1",
                text="Please add bus lanes.",
                topic=custom_types.FlatTopic(name="Transit"),
            )
        ],
    )
    learned = {
        "Transit": custom_types.NestedTopic(
            name="Transit",
            subtopics=[
                custom_types.FlatTopic(name="More buses"),
                custom_types.FlatTopic(name="Other"),
            ],
        )
    }
    with decision.use_decision_client(client):
      updated = list(
          asyncio.run(
              categorization._categorize_opinions_with_decision(
                  [statement],
                  [custom_types.FlatTopic(name="Transit")],
                  learned,
                  None,
                  True,
                  client,
              )
          )
      )

    self.assertEqual(updated[0].quotes[0].topic.subtopics[0].name, "Other")

  def test_autorater_skips_a_quote_that_already_fell_back_to_other(self):
    """Tests that no rubric is asked for a quote already assigned Other."""
    client, transport = _client()
    statement = custom_types.Statement(
        id="s1",
        text="The weather was fine.",
        quotes=[
            custom_types.Quote(
                id="q1",
                text="The weather was fine.",
                topic=custom_types.FlatTopic(name="Transit"),
            )
        ],
    )
    learned = {
        "Transit": custom_types.NestedTopic(
            name="Transit",
            subtopics=[custom_types.FlatTopic(name="More buses")],
        )
    }
    updated = list(
        asyncio.run(
            categorization._categorize_opinions_with_decision(
                [statement],
                [custom_types.FlatTopic(name="Transit")],
                learned,
                None,
                True,
                client,
            )
        )
    )

    self.assertEqual(updated[0].quotes[0].topic.subtopics[0].name, "Other")
    asked = [
        question["type"]
        for body in transport.bodies
        for question in body["questions"].values()
    ]
    self.assertEqual(asked, ["noul"])

  def test_autorater_min_score_zero_keeps_the_assignment(self):
    """Tests that autorater_min_score sets the rejection threshold."""
    client, _ = _client()
    statement = custom_types.Statement(
        id="s1",
        text="Please add bus lanes.",
        quotes=[
            custom_types.Quote(
                id="q1",
                text="Please add bus lanes.",
                topic=custom_types.FlatTopic(name="Transit"),
            )
        ],
    )
    learned = {
        "Transit": custom_types.NestedTopic(
            name="Transit",
            subtopics=[custom_types.FlatTopic(name="More buses")],
        )
    }
    updated = list(
        asyncio.run(
            categorization._categorize_opinions_with_decision(
                [statement],
                [custom_types.FlatTopic(name="Transit")],
                learned,
                None,
                True,
                client,
                autorater_min_score=0,
            )
        )
    )

    self.assertEqual(
        updated[0].quotes[0].topic.subtopics[0].name, "More buses"
    )

  def test_content_scorer_uses_decision_client_without_gemini(self):
    """Tests that ContentScorer scores without a Gemini client."""
    client, transport = _client()
    with decision.use_decision_client(client):
      scorer = ContentScorer(gemini_api_key="", model_name="unused")
      rows = asyncio.run(
          scorer.score_async(
              [{"row_id": "1", "text": "hello"}],
              ["TOXICITY"],
          )
      )

    self.assertIsNone(scorer.client)
    self.assertEqual(rows[0]["scores"]["TOXICITY"], 0.08)
    self.assertEqual(transport.bodies[0]["model"], "clef")

  def test_approval_votes_map_to_positive_labels(self):
    """Tests that approval votes map to positive labels."""
    client, _ = _client()
    frame = pd.DataFrame(
        [{"participant_id": "p1", "survey_text": "I want more buses."}]
    )
    results, stats = asyncio.run(
        simulated_jury._run_approval_with_decision(
            client,
            frame,
            ["Add bus lanes.", "Remove parking."],
            simulated_jury.ApprovalScale.AGREE_DISAGREE,
            "Transit",
            "",
        )
    )

    self.assertEqual(stats["decision_model"], "clef")
    self.assertEqual(stats["n_complete_fails"], 0)
    self.assertTrue(results.iloc[0]["result"]["Add bus lanes."])
    self.assertIn("participant_id", results.iloc[0]["data_row"])

  def test_choice_question_round_trips(self):
    """Tests a choice question round trip."""
    client, _ = _client()
    answers = asyncio.run(
        client.ask(
            "state",
            {
                "winner": systemone.ChoiceQuestion(
                    instructions="Pick.",
                    criteria={"a": "WIN", "b": "lose"},
                )
            },
        )
    )
    self.assertEqual(answers["winner"].choice, "a")

  def test_out_of_range_choice_is_a_parse_error(self):
    """Tests that an unknown choice key is a parse error."""
    def transport(url: str, body: bytes) -> bytes:
      """Returns a canned System One response."""
      return json.dumps({
          "model": "clef",
          "answers": {
              "winner": {
                  "type": "choice",
                  "choice": "k9",
                  "probabilities": {"k9": 1.0},
                  "confidence": 1.0,
              }
          },
      }).encode("utf-8")

    client = systemone.SystemOneClient(model="clef", transport=transport)
    with self.assertRaises(systemone.SystemOneError) as caught:
      asyncio.run(
          decision.choose_one(
              client, "Pick one.", [("a", "A"), ("b", "B")], "Which wins?"
          )
      )
    self.assertEqual(caught.exception.kind, systemone.FailureKind.PARSE)

  def test_endpoint_url_reads_hosts_and_full_urls(self):
    """Tests bare hosts, base URLs, and full endpoint URLs."""
    cases = {
        "0.0.0.0": "http://0.0.0.0:11434/v1/systemone",
        "localhost:8080": "http://localhost:8080/v1/systemone",
        " 127.0.0.1 ": "http://127.0.0.1:11434/v1/systemone",
        "[::1]": "http://[::1]:11434/v1/systemone",
        "http://gpu-box:11434/": "http://gpu-box:11434/v1/systemone",
        "https://api.typesafe.ai": "https://api.typesafe.ai/v1/systemone",
        "https://api.inceptionlabs.ai/v1/decisions/": (
            "https://api.inceptionlabs.ai/v1/decisions"
        ),
    }
    for host, expected in cases.items():
      with self.subTest(host=host):
        self.assertEqual(systemone.endpoint_url(host), expected)

  def test_invalid_host_is_a_config_error(self):
    """Tests that a blank or malformed host is a config error."""
    for host in ("", "http://", "host:port"):
      with self.subTest(host=host):
        with self.assertRaises(systemone.SystemOneError) as caught:
          systemone.endpoint_url(host)
        self.assertEqual(caught.exception.kind, systemone.FailureKind.CONFIG)

  def test_keep_alive_is_sent_only_when_set(self):
    """Tests that keep_alive is omitted from the body unless set."""
    question = {
        "q": systemone.NoulQuestion(instructions="Yes?", true="Yes", false="No")
    }
    for keep_alive in (None, "30m"):
      with self.subTest(keep_alive=keep_alive):
        transport = RecordingTransport()
        client = systemone.SystemOneClient(
            model="clef", transport=transport, keep_alive=keep_alive
        )
        asyncio.run(client.ask("state", question))
        if keep_alive is None:
          self.assertNotIn("keep_alive", transport.bodies[0])
        else:
          self.assertEqual(transport.bodies[0]["keep_alive"], keep_alive)

  def test_packs_at_the_configured_question_limit(self):
    """Tests that requests are packed at max_questions."""
    transport = RecordingTransport()
    client = systemone.SystemOneClient(
        model="clef", transport=transport, max_questions=16
    )
    questions = {
        f"q{index}": systemone.NoulQuestion(
            instructions=f"Is this item {index}?", true="Yes", false="No"
        )
        for index in range(17)
    }

    answers = asyncio.run(client.ask("a short state", questions))

    self.assertEqual(len(answers), 17)
    self.assertEqual(
        [len(body["questions"]) for body in transport.bodies], [16, 1]
    )

  def test_reads_answers_inside_a_result_envelope(self):
    """Tests that a Workers AI result envelope is unwrapped."""
    def transport(url: str, body: bytes) -> bytes:
      """Returns a canned System One response."""
      return json.dumps({
          "result": {
              "model": "clef",
              "answers": {"q": {"type": "noul", "noul": 0.75}},
          },
          "success": True,
          "errors": [],
      }).encode("utf-8")

    client = systemone.SystemOneClient(model="clef", transport=transport)
    answers = asyncio.run(
        client.ask(
            "state",
            {
                "q": systemone.NoulQuestion(
                    instructions="Yes?", true="Yes", false="No"
                )
            },
        )
    )
    self.assertEqual(answers["q"].noul, 0.75)

  def test_api_key_is_sent_as_a_bearer_header(self):
    """Tests that the key is sent as a bearer header, and only when set."""
    sent = []

    class _Response:
      """A minimal urlopen response."""

      def __enter__(self):
        """Enters the response context."""
        return self

      def __exit__(self, *exc):
        """Leaves the response context without suppressing errors."""
        return False

      def read(self):
        """Returns the canned response body."""
        return json.dumps({
            "model": "jev",
            "answers": {"q": {"type": "noul", "noul": 0.5}},
        }).encode("utf-8")

    def urlopen(request, timeout):
      """Stands in for urllib.request.urlopen."""
      sent.append(request)
      return _Response()

    question = {
        "q": systemone.NoulQuestion(instructions="Yes?", true="Yes", false="No")
    }
    with mock.patch.object(systemone.urllib.request, "urlopen", urlopen):
      for api_key, expected in (("secret", "Bearer secret"), (None, None)):
        client = systemone.SystemOneClient(
            model="jev",
            endpoint="https://api.typesafe.ai/v1/systemone",
            api_key=api_key,
        )
        asyncio.run(client.ask("state", question))
        self.assertEqual(sent[-1].get_header("Authorization"), expected)
    self.assertEqual(sent[0].full_url, "https://api.typesafe.ai/v1/systemone")

  def test_dropped_connection_is_retried(self):
    """Tests that a connection dropped by the host is retried."""
    calls = []

    class _Response:
      """A minimal urlopen response."""

      def __enter__(self):
        """Enters the response context."""
        return self

      def __exit__(self, *exc):
        """Leaves the response context without suppressing errors."""
        return False

      def read(self):
        """Returns the canned response body."""
        return json.dumps({
            "model": "clef",
            "answers": {"q": {"type": "noul", "noul": 0.25}},
        }).encode("utf-8")

    def urlopen(request, timeout):
      """Stands in for urllib.request.urlopen."""
      calls.append(request.full_url)
      if len(calls) == 1:
        raise http.client.RemoteDisconnected(
            "Remote end closed connection without response"
        )
      return _Response()

    client = systemone.SystemOneClient(model="clef", retry_delay_seconds=0)
    question = {
        "q": systemone.NoulQuestion(instructions="Yes?", true="Yes", false="No")
    }
    with mock.patch.object(systemone.urllib.request, "urlopen", urlopen):
      answers = asyncio.run(client.ask("state", question))

    self.assertEqual(answers["q"].noul, 0.25)
    self.assertEqual(len(calls), 2)

  def test_a_404_says_the_host_does_not_serve_decisions(self):
    """Tests that a 404 names the Ollama version and DECISION_MODEL_HOST."""
    url = "http://127.0.0.1:11434/v1/systemone"

    def urlopen(request, timeout):
      """Stands in for urllib.request.urlopen."""
      raise systemone.urllib.error.HTTPError(
          url, 404, "Not Found", {}, io.BytesIO(b"404 page not found")
      )

    client = systemone.SystemOneClient(model="clef", endpoint=url)
    question = {
        "q": systemone.NoulQuestion(instructions="Yes?", true="Yes", false="No")
    }
    with mock.patch.object(systemone.urllib.request, "urlopen", urlopen):
      with self.assertRaises(systemone.SystemOneError) as caught:
        asyncio.run(client.ask("state", question))
    self.assertEqual(caught.exception.status, 404)
    self.assertIn("0.35.1", str(caught.exception))
    self.assertIn("DECISION_MODEL_HOST", str(caught.exception))

  def test_extract_answer_spans_come_back_in_state_order(self):
    """Tests the extract wire format, span order, and code-point offsets."""
    state = "🚲 Bikes! Build protected lanes. Fix potholes."
    bodies = []

    def transport(url: str, body: bytes) -> bytes:
      """Returns two spans out of order for the one extract question."""
      bodies.append(json.loads(body))
      spans = [
          {"text": "Fix potholes", "start": 32, "end": 44, "confidence": 0.7},
          {"text": "Bikes", "start": 2, "end": 7, "confidence": 0.9},
      ]
      return json.dumps({
          "model": "mercury-decide",
          "answers": {
              "q": {
                  "type": "extract",
                  "spans": {"q": spans},
                  "not_stated": {"q": 0.01},
              }
          },
      }).encode("utf-8")

    client = systemone.SystemOneClient(
        model="mercury-decide", transport=transport
    )
    answers = asyncio.run(
        client.ask(
            state, {"q": systemone.ExtractQuestion(instructions="Ideas.")}
        )
    )

    answer = answers["q"]
    self.assertIsInstance(answer, systemone.ExtractAnswer)
    assert isinstance(answer, systemone.ExtractAnswer)
    self.assertEqual([s.text for s in answer.spans], ["Bikes", "Fix potholes"])
    self.assertEqual(answer.not_stated, 0.01)
    self.assertEqual(
        bodies[0]["questions"]["q"],
        {"type": "extract", "instructions": "Ideas."},
    )

  def test_extract_span_off_its_offsets_is_a_parse_error(self):
    """Tests that a span whose text is not at its offsets is rejected."""

    def transport(url: str, body: bytes) -> bytes:
      """Returns a span whose offsets point at different text."""
      return json.dumps({
          "model": "mercury-decide",
          "answers": {
              "q": {
                  "type": "extract",
                  "spans": {
                      "q": [{
                          "text": "lanes",
                          "start": 0,
                          "end": 5,
                          "confidence": 0.9,
                      }]
                  },
              }
          },
      }).encode("utf-8")

    client = systemone.SystemOneClient(
        model="mercury-decide", transport=transport
    )
    with self.assertRaises(systemone.SystemOneError) as caught:
      asyncio.run(
          client.ask(
              "Build protected lanes.",
              {"q": systemone.ExtractQuestion(instructions="Ideas.")},
          )
      )
    self.assertEqual(caught.exception.kind, systemone.FailureKind.PARSE)

  def test_installed_client_enables_the_decision_path(self):
    """Tests that an installed client turns the decision path on and off."""
    client, _ = _client()
    with decision.use_decision_client(client):
      self.assertTrue(decision.decision_enabled())
      self.assertIs(decision.decision_client(), client)
    with decision.use_decision_client(None):
      self.assertFalse(decision.decision_enabled())
      self.assertIsNone(decision.decision_client())


if __name__ == "__main__":
  unittest.main()
