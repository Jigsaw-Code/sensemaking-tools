# Copyright 2025 Google LLC
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

"""Tests for run_health."""

import logging

import pytest

from src import run_health


@pytest.fixture(autouse=True)
def _reset():
  run_health.reset()
  yield
  run_health.reset()


def test_record_and_summary():
  run_health.record("a")
  run_health.record("a", n=2, detail="x")
  run_health.record("b", detail="y")
  run_health.record("c", n=3, detail=["z1", "z2"])
  assert run_health.summary() == {"a": 3, "b": 1, "c": 3}
  assert run_health.details("a") == ["x"]
  assert run_health.details("b") == ["y"]
  assert run_health.details("c") == ["z1", "z2"]


def test_record_ignores_non_positive():
  run_health.record("a", n=0)
  run_health.record("a", n=-1)
  assert run_health.summary() == {}


def test_details_are_capped():
  for i in range(100):
    run_health.record("a", detail=str(i))
  assert len(run_health.details("a")) == run_health._MAX_DETAILS_PER_EVENT
  assert run_health.summary()["a"] == 100


def test_record_alone_logs_summary_at_info(caplog):
  """Verifies record() does not mark run as degraded and summary logs at INFO."""
  run_health.record("other_source.pass1.model_choice", n=5, detail="item 1")
  with caplog.at_level(logging.INFO):
    counts = run_health.log_summary()
  assert counts == {"other_source.pass1.model_choice": 5}
  assert "other_source.pass1.model_choice: 5" in caplog.text
  assert any(r.levelno == logging.INFO for r in caplog.records)
  assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_warning_and_error_logging_and_escalation(caplog):
  """Verifies warning() and error() log at correct levels with stacklevel=2 and escalate summary."""
  with caplog.at_level(logging.INFO):
    run_health.warning(
        "topic_chunk_failed",
        "Topic learning chunk 3 failed",
        n=1,
        detail="chunk 3/10",
    )
    run_health.error(
        "model_stalled",
        "Model inference stalled completely",
        n=1,
        detail=["statement 42", "statement 43"],
    )

  # Check that direct log messages were emitted with stacklevel=2 (attribution to this file)
  warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
  error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
  assert len(warn_records) == 1
  assert warn_records[0].getMessage() == "Topic learning chunk 3 failed"
  assert warn_records[0].filename == "run_health_test.py"
  assert len(error_records) == 1
  assert error_records[0].getMessage() == "Model inference stalled completely"
  assert error_records[0].filename == "run_health_test.py"

  # Check counts and details
  assert run_health.summary() == {
      "topic_chunk_failed": 1,
      "model_stalled": 1,
  }
  assert run_health.details("topic_chunk_failed") == ["chunk 3/10"]
  assert run_health.details("model_stalled") == [
      "statement 42",
      "statement 43",
  ]

  # Check that log_summary() escalates to WARNING because degraded events occurred
  caplog.clear()
  with caplog.at_level(logging.INFO):
    counts = run_health.log_summary()
  assert counts == {"topic_chunk_failed": 1, "model_stalled": 1}
  assert any(r.levelno == logging.WARNING for r in caplog.records)
  assert "topic_chunk_failed: 1" in caplog.text
  assert "model_stalled: 1" in caplog.text


def test_empty_log_summary_at_info(caplog):
  """Verifies clean run with no events logs summary at INFO."""
  with caplog.at_level(logging.INFO):
    assert run_health.log_summary() == {}
  assert "No degraded-path events recorded." in caplog.text
  assert not any(r.levelno >= logging.WARNING for r in caplog.records)
