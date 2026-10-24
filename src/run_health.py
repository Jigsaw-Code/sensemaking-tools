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

"""Process-wide registry of degraded-path events for end-of-run assertions.

Pipeline stages that silently degrade (a skipped topic-learning chunk, a
failed opinion-learning chunk, a quote defaulted to 'Other' after exhausted
retries, ...) record an event here in addition to logging. At the end of a
run, `log_summary()` prints one consolidated block so degradations cannot hide
in thousands of log lines.

Usage:
  from src import run_health

  # Degraded-path events: log at WARNING / ERROR and track occurrence:
  run_health.warning(
      "topic_chunk_failed",
      "Topic learning chunk 118 failed",
      detail="chunk 118/298",
  )
  run_health.error(
      "model_stalled",
      "Model inference stalled on statement 42",
      detail="statement 42",
  )

  # Informational provenance / telemetry: record without logging:
  run_health.record("other_source.pass1.model_choice", n=3)

  # Emit end-of-run summary (WARNING if degraded events occurred, else INFO):
  run_health.log_summary()
"""

import collections
from collections.abc import Sequence
import logging
import threading

# Maximum number of detail strings retained per event type.
_MAX_DETAILS_PER_EVENT = 20

_lock = threading.Lock()
_counts: collections.Counter[str] = collections.Counter()
_details: dict[str, list[str]] = collections.defaultdict(list)
_has_degraded: bool = False


def _record(
    event: str,
    n: int = 1,
    detail: str | Sequence[str] | None = None,
    *,
    degraded: bool = False,
) -> None:
  """Internal helper to record event occurrences and details."""
  global _has_degraded
  if n <= 0:
    return
  if isinstance(detail, str):
    details_to_add = [detail]
  elif detail is not None:
    details_to_add = list(detail)
  else:
    details_to_add = []

  with _lock:
    if degraded:
      _has_degraded = True
    _counts[event] += n
    remaining_capacity = _MAX_DETAILS_PER_EVENT - len(_details[event])
    if remaining_capacity > 0 and details_to_add:
      _details[event].extend(details_to_add[:remaining_capacity])


def record(
    event: str,
    n: int = 1,
    detail: str | Sequence[str] | None = None,
) -> None:
  """Records `n` occurrences of a non-degraded / informational event without logging."""
  _record(event, n=n, detail=detail, degraded=False)


def warning(
    event: str,
    message: str,
    *,
    n: int = 1,
    detail: str | Sequence[str] | None = None,
) -> None:
  """Logs `message` at WARNING level and records `n` occurrences of a degraded event."""
  logging.warning(message, stacklevel=2)
  _record(event, n=n, detail=detail, degraded=True)


def error(
    event: str,
    message: str,
    *,
    n: int = 1,
    detail: str | Sequence[str] | None = None,
) -> None:
  """Logs `message` at ERROR level and records `n` occurrences of a degraded event."""
  logging.error(message, stacklevel=2)
  _record(event, n=n, detail=detail, degraded=True)


def summary() -> dict[str, int]:
  """Returns a snapshot of event counts."""
  with _lock:
    return dict(_counts)


def details(event: str) -> list[str]:
  """Returns the retained detail strings for an event."""
  with _lock:
    return list(_details.get(event, []))


def reset() -> None:
  """Clears all recorded events (used by tests and at run start)."""
  global _has_degraded
  with _lock:
    _counts.clear()
    _details.clear()
    _has_degraded = False


def format_summary() -> str:
  """Formats the run-health summary as a multi-line string."""
  counts = summary()
  lines = ["=" * 30 + " RUN HEALTH SUMMARY " + "=" * 30]
  if not counts:
    lines.append("No degraded-path events recorded.")
  else:
    for event in sorted(counts):
      lines.append(f"  {event}: {counts[event]}")
      for d in details(event)[:5]:
        lines.append(f"      e.g. {d}")
  lines.append("=" * 80)
  return "\n".join(lines)


def log_summary() -> dict[str, int]:
  """Logs the summary (WARNING if degraded events occurred, else INFO) and returns counts."""
  counts = summary()
  with _lock:
    degraded = _has_degraded
  text = format_summary()
  if degraded:
    logging.warning(text)
  else:
    logging.info(text)
  return counts
