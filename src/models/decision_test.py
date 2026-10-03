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
import json
import unittest

import pandas as pd

from src.get_gemini_scores_lib import ContentScorer
from src.models import decision
from src.models.custom_types import FlatTopic, NestedTopic, Quote, Statement
from src.models.decision import AttributeSpec
from src.models.systemone import (
    ChoiceQuestion,
    FailureKind,
    NoulQuestion,
    SystemOneClient,
    SystemOneError,
)
from src.simulated_jury import simulated_jury
from src.tasks.categorization import (
    _categorize_opinions_with_decision,
    _process_topic_categorization,
)


def _answers_for(payload: dict) -> dict:
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
      if "minimality" in question["instructions"]:
        chosen = "0"
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
      answers[key] = {
          "type": "score",
          "score": 1.0,
          "probabilities": {"0": 0.0, "1": 1.0},
          "confidence": 1.0,
      }
  return answers


class RecordingTransport:

  def __init__(self):
    self.bodies = []

  def __call__(self, url: str, body: bytes) -> bytes:
    payload = json.loads(body)
    self.bodies.append(payload)
    encoded = {
        "model": payload["model"],
        "answers": _answers_for(payload),
        "usage": {"input_tokens": 4, "output_tokens": 0},
    }
    return json.dumps(encoded).encode("utf-8")


def _client() -> tuple[SystemOneClient, RecordingTransport]:
  transport = RecordingTransport()
  return (
      SystemOneClient(model="clef", transport=transport, max_concurrent=1),
      transport,
  )


class DecisionTests(unittest.TestCase):

  def test_assigns_matching_topic_and_drops_other_when_specific(self):
    client, _ = _client()

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

  def test_assigns_highest_label_when_none_clear_the_cutoff(self):
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

    self.assertEqual(chosen, ["Parking"])

  def test_clusters_only_equivalent_pairs(self):
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

  def test_tournament_selects_marked_option_past_26(self):
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
    client, transport = _client()
    questions = {
        f"q{index}": NoulQuestion(
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
    def transport(url: str, body: bytes) -> bytes:
      return b'{"error": "clef does not support chat"}'

    client = SystemOneClient(model="clef", transport=transport)
    with self.assertRaises(SystemOneError) as caught:
      asyncio.run(
          client.ask(
              "state",
              {"q": NoulQuestion(instructions="Yes?", true="Yes", false="No")},
          )
      )
    self.assertEqual(caught.exception.kind, FailureKind.HTTP)

  def test_scores_attribute_as_probability(self):
    client, _ = _client()

    scores = asyncio.run(
        decision.score_attributes(
            client,
            "You are an idiot.",
            [
                AttributeSpec(
                    name="TOXICITY",
                    label="Toxicity",
                    definition="A rude comment.",
                )
            ],
        )
    )

    self.assertEqual(scores, {"TOXICITY": 0.08})

  def test_topic_assignment_does_not_call_the_generative_model(self):
    client, transport = _client()
    statement = Statement(id="s1", text="Please add bus lanes.")
    with decision.use_decision_client(client):
      records = asyncio.run(
          _process_topic_categorization(
              [statement],
              model=object(),
              target_topics=[FlatTopic(name="Buses"), FlatTopic(name="Other")],
          )
      )

    self.assertEqual([topic.name for topic in records[0].topics], ["Buses"])
    self.assertTrue(transport.bodies)
    self.assertEqual(transport.bodies[0]["model"], "clef")

  def test_failing_opinion_autorater_becomes_other(self):
    client, _ = _client()
    statement = Statement(
        id="s1",
        text="Please add bus lanes.",
        quotes=[
            Quote(
                id="q1",
                text="Please add bus lanes.",
                topic=FlatTopic(name="Transit"),
            )
        ],
    )
    learned = {
        "Transit": NestedTopic(
            name="Transit",
            subtopics=[FlatTopic(name="More buses"), FlatTopic(name="Other")],
        )
    }
    with decision.use_decision_client(client):
      updated = list(
          asyncio.run(
              _categorize_opinions_with_decision(
                  [statement],
                  [FlatTopic(name="Transit")],
                  learned,
                  None,
                  True,
                  client,
              )
          )
      )

    self.assertEqual(updated[0].quotes[0].topic.subtopics[0].name, "Other")

  def test_content_scorer_uses_decision_client_without_gemini(self):
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
    client, _ = _client()
    answers = asyncio.run(
        client.ask(
            "state",
            {
                "winner": ChoiceQuestion(
                    instructions="Pick.",
                    criteria={"a": "WIN", "b": "lose"},
                )
            },
        )
    )
    self.assertEqual(answers["winner"].choice, "a")

  def test_missing_decision_model_is_a_config_error(self):
    with self.assertRaises(SystemOneError) as caught:
      decision.client_from_env()
    self.assertEqual(caught.exception.kind, FailureKind.CONFIG)


if __name__ == "__main__":
  unittest.main()
