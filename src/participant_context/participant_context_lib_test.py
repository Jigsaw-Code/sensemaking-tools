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

"""Tests for participant context. Models are fakes."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.models import extraction
from src.models import systemone
from src.participant_context import participant_context_lib as lib
from src.participant_context import participant_context_runner as runner


class _PhraseExtractor:
  """Marks the first phrase from a fixed list that appears in the text."""

  model = "fake-extractor"

  def __init__(self, phrases: list[str]):
    """Stores the phrases to look for and records requests."""
    self.phrases = phrases
    self.requests = []

  async def extract(self, text, fields):
    """Returns the first matching phrase for every field."""
    self.requests.append((text, dict(fields)))
    results = {}
    for name in fields:
      spans = ()
      for phrase in self.phrases:
        if phrase in text:
          start = text.index(phrase)
          spans = (
              systemone.Span(
                  text=phrase,
                  start=start,
                  end=start + len(phrase),
                  confidence=0.9,
              ),
          )
          break
      results[name] = extraction.Extraction(
          spans=spans, not_stated=0.05 if spans else 0.95
      )
    return results


def _decision_client(bodies: list[dict[str, object]]) -> systemone.SystemOneClient:
  """Returns a client that labels bakery evidence business_owner."""

  def transport(url: str, body: bytes) -> bytes:
    """Picks business_owner when the state mentions a bakery, else none."""
    payload = json.loads(body)
    bodies.append(payload)
    pick = "business_owner" if "bakery" in payload["state"] else "none"
    options = payload["questions"]["label"]["criteria"]
    return json.dumps({
        "model": "clef",
        "answers": {
            "label": {
                "type": "choice",
                "choice": pick,
                "probabilities": {k: 0.9 if k == pick else 0.02 for k in options},
                "confidence": 0.9,
            }
        },
    }).encode("utf-8")

  return systemone.SystemOneClient(model="clef", transport=transport)


def _checking_client(bodies: list[dict[str, object]]) -> systemone.SystemOneClient:
  """Returns a client that confirms only evidence about years lived here."""

  def transport(url: str, body: bytes) -> bytes:
    """Answers yes when the evidence mentions years, else no."""
    payload = json.loads(body)
    bodies.append(payload)
    evidence = payload["state"].rsplit("Evidence: ", 1)[1]
    return json.dumps({
        "model": "clef",
        "answers": {
            "evidence": {
                "type": "noul",
                "noul": 0.9 if "years" in evidence else 0.1,
            }
        },
    }).encode("utf-8")

  return systemone.SystemOneClient(model="clef", transport=transport)


class ParticipantContextTest(unittest.TestCase):
  """Tests for the participant context library and runner."""

  def test_labels_evidence_and_leaves_silence_as_none(self):
    """Tests evidence, labels, and that silence is never asked about."""
    extractor = _PhraseExtractor(["I run a bakery"])
    bodies = []
    results = asyncio.run(
        lib.describe_participants(
            extractor,
            _decision_client(bodies),
            [
                "<response>I run a bakery on Oak St</response>",
                "Please fix the potholes.",
            ],
        )
    )

    self.assertEqual(results[0]["role"].evidence, "I run a bakery")
    self.assertEqual(results[0]["role"].label, "business_owner")
    self.assertEqual(results[0]["role"].label_probability, 0.9)
    self.assertIsNone(results[1]["role"].evidence)
    self.assertEqual(results[1]["role"].label, lib.NONE_LABEL)
    self.assertIsNone(results[1]["role"].label_probability)
    self.assertEqual(len(bodies), 1)
    self.assertIn(
        lib.NONE_LABEL, bodies[0]["questions"]["label"]["criteria"]
    )
    self.assertEqual(extractor.requests[0][0], "I run a bakery on Oak St.")
    self.assertFalse(extractor.requests[0][1]["role"].many)

  def test_without_a_decision_model_only_evidence_is_written(self):
    """Tests the evidence-only mode."""
    results = asyncio.run(
        lib.describe_participants(
            _PhraseExtractor(["my son"]), None, ["Please, my son walks."]
        )
    )
    self.assertEqual(results[0]["role"].evidence, "my son")
    self.assertIsNone(results[0]["role"].label)

  def test_decision_model_checks_evidence_without_labels(self):
    """Tests that rejected evidence is dropped and silence is not asked."""
    tenure = lib.Attribute("tenure", "How long they have lived here.")
    bodies = []
    with mock.patch.dict(os.environ, {"DECISION_THRESHOLD": "0.5"}):
      results = asyncio.run(
          lib.describe_participants(
              _PhraseExtractor(["30 years", "older people"]),
              _checking_client(bodies),
              [
                  "I have lived here 30 years.",
                  "Removing spaces hurts older people.",
                  "Fix the roads.",
              ],
              [tenure],
          )
      )

    self.assertEqual(results[0]["tenure"].evidence, "30 years")
    self.assertIsNone(results[1]["tenure"].evidence)
    self.assertIsNone(results[2]["tenure"].evidence)
    self.assertTrue(all(r["tenure"].label is None for r in results))
    self.assertEqual(len(bodies), 2)
    self.assertEqual(
        bodies[0]["questions"]["evidence"]["type"], "noul"
    )

  def test_load_attributes_validates(self):
    """Tests reading attributes and rejecting bad files."""
    with tempfile.TemporaryDirectory() as tmp:
      good = os.path.join(tmp, "good.json")
      with open(good, "w", encoding="utf-8") as handle:
        json.dump(
            [{"name": "tenure", "instructions": "Years lived here"}], handle
        )
      attributes = lib.load_attributes(good)
      self.assertEqual(attributes[0].name, "tenure")
      self.assertEqual(dict(attributes[0].labels), {})
      for bad in (
          [],
          [{"name": "has space", "instructions": "x"}],
          [{"name": "a", "instructions": "x", "labels": {"two words": "y"}}],
          [{"name": "a", "instructions": "x", "labels": {"none": "y"}}],
          [{"name": "a", "instructions": "x"}, {"name": "a", "instructions": "y"}],
      ):
        path = os.path.join(tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
          json.dump(bad, handle)
        with self.subTest(bad=bad):
          with self.assertRaises(ValueError):
            lib.load_attributes(path)

  def test_runner_writes_context_columns(self):
    """Tests the runner end to end with CSV files and fake models."""
    bodies = []
    with tempfile.TemporaryDirectory() as tmp:
      source = os.path.join(tmp, "in.csv")
      target = os.path.join(tmp, "out.csv")
      pd.DataFrame({
          "participant_id": ["p1", "p2"],
          "survey_text": ["I run a bakery downtown.", "Fix the roads."],
      }).to_csv(source, index=False)
      with mock.patch.object(
          runner.model_factory,
          "get_extractor",
          return_value=_PhraseExtractor(["I run a bakery"]),
      ), mock.patch.object(
          runner.decision,
          "decision_client",
          return_value=_decision_client(bodies),
      ):
        asyncio.run(
            runner.main(
                runner.get_args(
                    ["--input_csv", source, "--output_csv", target]
                )
            )
        )
      output = pd.read_csv(target)

    self.assertEqual(
        list(output.columns),
        [
            "participant_id",
            "survey_text",
            "role_evidence",
            "role_not_stated",
            "role_label",
            "role_label_probability",
        ],
    )
    self.assertEqual(output.role_label.tolist(), ["business_owner", "none"])
    self.assertEqual(output.role_evidence.fillna("").tolist(), ["I run a bakery", ""])

  def test_runner_requires_an_extractor(self):
    """Tests the error when CONTEXT_* is not configured."""
    with tempfile.TemporaryDirectory() as tmp:
      source = os.path.join(tmp, "in.csv")
      pd.DataFrame({"participant_id": ["p1"], "survey_text": ["x"]}).to_csv(
          source, index=False
      )
      with mock.patch.object(
          runner.model_factory, "get_extractor", return_value=None
      ):
        with self.assertRaises(ValueError):
          asyncio.run(
              runner.main(
                  runner.get_args(
                      ["--input_csv", source, "--output_csv", source]
                  )
              )
          )


if __name__ == "__main__":
  unittest.main()
