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

"""Tests for the extraction backends. Transports are fake."""

import asyncio
import json
import unittest

from src.models import extraction
from src.models import systemone

TEXT = "I run a bakery on Oak St. My son was hit by a car. Please add lanes."


def _offsets(phrase: str) -> list[int]:
  """Returns [start, end] of phrase in TEXT."""
  start = TEXT.index(phrase)
  return [start, start + len(phrase)]


class ClairicExtractorTest(unittest.TestCase):
  """Tests for ClairicExtractor."""

  def test_parses_text_and_list_fields(self):
    """Tests field types, the result envelope, null spans and order."""
    bodies = []

    def transport(url: str, body: bytes) -> bytes:
      """Answers like Workers AI: one text field and one list field."""
      bodies.append((url, json.loads(body)))
      return json.dumps({
          "result": {
              "answers": {
                  "role": {
                      "type": "text",
                      "value": "I run a bakery",
                      "span": _offsets("I run a bakery"),
                      "confidence": 0.8,
                      "p_null": 0.01,
                  },
                  "asks": {
                      "type": "list",
                      "value": ["Please add lanes.", "garbled", "My son"],
                      "spans": [
                          _offsets("Please add lanes."),
                          None,
                          _offsets("My son"),
                      ],
                      "confidence": 0.2,
                  },
                  "years": {"type": "text", "value": None, "p_null": 0.99},
              }
          },
          "success": True,
      }).encode("utf-8")

    extractor = extraction.ClairicExtractor(
        endpoint="https://example/clairic", transport=transport
    )
    result = asyncio.run(
        extractor.extract(
            TEXT,
            {
                "role": extraction.Field("Their role", many=False),
                "asks": extraction.Field("What they ask for"),
                "years": extraction.Field("Years lived here", many=False),
            },
        )
    )

    url, body = bodies[0]
    self.assertEqual(url, "https://example/clairic")
    self.assertEqual(body["state"], TEXT)
    self.assertEqual(body["fields"]["role"]["type"], "text")
    self.assertEqual(
        body["fields"]["asks"], {
            "type": "list",
            "items": "text",
            "instructions": "What they ask for",
        }
    )
    self.assertEqual([s.text for s in result["role"].spans], ["I run a bakery"])
    self.assertEqual(result["role"].not_stated, 0.01)
    self.assertEqual(
        [s.text for s in result["asks"].spans], ["My son", "Please add lanes."]
    )
    self.assertEqual(result["years"].spans, ())
    self.assertEqual(result["years"].not_stated, 0.99)

  def test_span_off_its_offsets_is_a_parse_error(self):
    """Tests that a span not at its offsets is rejected."""

    def transport(url: str, body: bytes) -> bytes:
      """Answers with offsets that point at different text."""
      return json.dumps({
          "answers": {
              "role": {"value": "bakery", "span": [0, 6], "confidence": 0.9}
          }
      }).encode("utf-8")

    extractor = extraction.ClairicExtractor(
        endpoint="https://example/clairic", transport=transport
    )
    with self.assertRaises(systemone.SystemOneError) as caught:
      asyncio.run(
          extractor.extract(TEXT, {"role": extraction.Field("Role", many=False)})
      )
    self.assertEqual(caught.exception.kind, systemone.FailureKind.PARSE)

  def test_missing_field_or_malformed_offsets_is_a_parse_error(self):
    """Tests that an omitted field or a bad offset pair is not read as empty."""
    cases = {
        "omitted field": {"other": {"value": None}},
        "list offsets": {
            "role": {"type": "list", "value": ["My son"], "spans": [[26]]}
        },
        "text offsets": {
            "role": {"type": "text", "value": "My son", "span": [26, 32, 1]}
        },
    }
    for case, answers in cases.items():

      def transport(url: str, body: bytes, answers=answers) -> bytes:
        """Answers with the malformed case."""
        return json.dumps({"answers": answers}).encode("utf-8")

      extractor = extraction.ClairicExtractor(
          endpoint="https://example/clairic", transport=transport
      )
      with self.subTest(case=case):
        with self.assertRaises(systemone.SystemOneError) as caught:
          asyncio.run(
              extractor.extract(
                  TEXT, {"role": extraction.Field("Role", many=False)}
              )
          )
        self.assertEqual(caught.exception.kind, systemone.FailureKind.PARSE)

  def test_transient_failure_is_retried(self):
    """Tests that Clairic requests share the System One retry policy."""
    calls = []

    def transport(url: str, body: bytes) -> bytes:
      """Fails once with a 502, then answers."""
      calls.append(url)
      if len(calls) == 1:
        raise systemone.SystemOneError(
            systemone.FailureKind.HTTP, "upstream", status=502
        )
      return json.dumps(
          {"answers": {"role": {"type": "list", "value": []}}}
      ).encode("utf-8")

    extractor = extraction.ClairicExtractor(
        endpoint="https://example/clairic",
        transport=transport,
        retry_delay_seconds=0,
    )
    result = asyncio.run(extractor.extract(TEXT, {"role": extraction.Field("R")}))

    self.assertEqual(len(calls), 2)
    self.assertEqual(result["role"].spans, ())


class SystemOneExtractorTest(unittest.TestCase):
  """Tests for SystemOneExtractor."""

  def test_asks_one_question_per_field(self):
    """Tests the extract questions and single-span selection."""
    bodies = []

    def transport(url: str, body: bytes) -> bytes:
      """Marks two spans for every question."""
      payload = json.loads(body)
      bodies.append(payload)
      spans = [
          {"text": "I run a bakery", "start": 0, "end": 14, "confidence": 0.4},
          {"text": "My son", "start": 26, "end": 32, "confidence": 0.9},
      ]
      return json.dumps({
          "model": "mercury-decide",
          "answers": {
              key: {
                  "type": "extract",
                  "spans": {key: spans},
                  "not_stated": {key: 0.1},
              }
              for key in payload["questions"]
          },
      }).encode("utf-8")

    client = systemone.SystemOneClient(
        model="mercury-decide", transport=transport
    )
    extractor = extraction.SystemOneExtractor(client)
    result = asyncio.run(
        extractor.extract(
            TEXT,
            {
                "one": extraction.Field("Single", many=False),
                "all": extraction.Field("Several"),
            },
        )
    )

    self.assertEqual(extractor.model, "mercury-decide")
    self.assertEqual(
        {k: q["type"] for k, q in bodies[0]["questions"].items()},
        {"one": "extract", "all": "extract"},
    )
    self.assertEqual([s.text for s in result["one"].spans], ["My son"])
    self.assertEqual(len(result["all"].spans), 2)
    self.assertEqual(result["all"].not_stated, 0.1)


if __name__ == "__main__":
  unittest.main()
