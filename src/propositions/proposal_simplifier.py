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

"""Plain-English proposal simplifier with deterministic readability assertions.

Translates actionable proposals into clear, accessible plain-English text while
preserving full substantive meaning, operational mechanisms, and trade-offs:
- Uses `pyphen` directly for offline English syllable counting and deterministic
  Flesch-Kincaid Grade Level (FKGL) scoring.
- Returns already-compliant proposals verbatim without calling the model.
- Runs a bounded retry loop that feeds exact deterministic readability
  violations back to the model when a rewrite misses target thresholds.
- Preserves both pre-simplified and post-simplified statements alongside
  compliance diagnostics.
"""

import asyncio
from collections import abc
import dataclasses
import functools
import logging
import math
import re

import pandas as pd
import pyphen

from src.models import base_model
from src.models import retry_utils

DEFAULT_MAX_FKGL: float = 9.5
TARGET_FKGL: float = 8.0
TARGET_SENTENCES: int = 4
DEFAULT_MAX_SENTENCES: int = 6
DEFAULT_MAX_ATTEMPTS: int = 5
DEFAULT_MAX_CONCURRENT_CALLS: int = 20

_FKGL_PENALTY_WEIGHT: float = 25.0
_SENTENCE_PENALTY_WEIGHT: float = 20.0
_WORD_PENALTY_WEIGHT: float = 2.0
_BASE_TEMPERATURE: float = 0.1
_TEMPERATURE_STEP: float = 0.1

_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")
_WORD_RE = re.compile(r"\b[a-zA-Z0-9']+\b")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_FENCE_START_RE = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_END_RE = re.compile(r"\s*```$")
_BOLD_TITLE_PREFIX_RE = re.compile(
    r"^\*\*[^*]+\*\*[:\-–—]?\s*(.*)$", flags=re.DOTALL
)

SIMPLIFY_SYSTEM_PROMPT = (
    "You are an expert plain-English editor translating actionable proposals"
    " into clear, accessible ~8th-grade reading level text.\n\n"
    "GOAL:\n"
    "- Aim specifically for an 8th-grade reading level (Flesch-Kincaid Grade"
    " Level around 8.0) while preserving every substantive mechanism and"
    " qualification.\n"
    "- Keep proposals concise and clear without stripping concrete details.\n\n"
    "CORE EDITING DIRECTIVES:\n"
    "1. DIRECT OPERATIVE / IMPERATIVE MOOD (NO BOILERPLATE PREFIXES):\n"
    "   - Use the imperative mood (action-oriented directive) or a direct"
    " operative construction (e.g., opening directly with an action verb like"
    " 'Set aside...', 'Require...', 'Establish...', 'Publish...', 'Build...')"
    " without boilerplate subject prefixes, passive framing, or filler"
    " language.\n"
    "   - Varied subjects are permitted where natural to clarify"
    " responsibility (e.g., 'An independent oversight board will audit...').\n"
    "2. PRESERVE FULL MEANING, MECHANISMS & QUALIFICATIONS:\n"
    "   - Reuse the literal phrasing of the input statement to the extent"
    " consistent with an ~8th-grade reading level. If a sentence is already"
    " clear and accessible, keep it intact rather than rewriting from"
    " scratch.\n"
    "   - Preserve all specific instruments, numbers, funding constraints,"
    " qualifications, timelines, and trade-off disclosures (such as delays or"
    " conditions attached to external approvals).\n"
    "3. STRAIGHTFORWARD VOCABULARY:\n"
    "   - Stick to straightforward, everyday vocabulary, and avoid"
    " unnecessary jargon and filler adjectives that do not change the"
    " substantive meaning of the proposal.\n"
    "   - Use technical or formal terminology only when essential to the"
    " specific implementation pathway.\n"
    "4. ZERO META-COMMENTARY:\n"
    "   - Do not include meta-commentary or rhetorical framing explaining why"
    " the proposal was written.\n\n"
    "OUTPUT FORMAT:\n"
    "Output ONLY the plain-English proposal statement (without title prefixes,"
    " markdown formatting, or commentary)."
)

_ADDITIONAL_CONTEXT_BLOCK = (
    "\n\n<additionalContext>\n{additional_context}\n</additionalContext>"
)

_INITIAL_PROMPT_TEMPLATE = (
    "Rewrite the following actionable proposal to an ~8th-grade reading level"
    " (Flesch-Kincaid Grade Level <= {max_fkgl:g}, aiming for no more than"
    " {target_sentences} sentences when possible and at most {max_sentences}"
    " sentences{word_limit_clause}) while preserving all substantive"
    " mechanisms, qualifications, and literal phrasing where"
    " consistent:{context_block}\n\n"
    "Actionable Proposal:\n"
    "{statement}"
)

_RETRY_PROMPT_TEMPLATE = (
    "Original Actionable Proposal:\n"
    "{original_statement}\n\n"
    "Previous Attempt:\n"
    "{previous_attempt}\n\n"
    "Feedback to Resolve: {violations_text}{context_block}\n\n"
    "Requirements:\n"
    "1. Flesch-Kincaid Grade Level MUST be <= {max_fkgl:g} (aiming for"
    " ~{target_fkgl:g}).\n"
    "2. Aim for no more than {target_sentences} sentences when possible, and"
    " strictly no more than {max_sentences} sentences.\n"
    "3. {word_limit_requirement}\n"
    "4. Use direct operative / imperative construction without boilerplate"
    " prefixes.\n\n"
    "Revised Proposal Statement:"
)

_NO_WORD_LIMIT_REQUIREMENT = (
    "Preserve all substantive mechanisms, funding constraints, and"
    " qualifications without arbitrary word-count truncation."
)


@dataclasses.dataclass(frozen=True, slots=True)
class ReadabilityScore:
  """Deterministic readability metrics for a proposal statement.

  Attributes:
    fkgl: Flesch-Kincaid Grade Level (FKGL) score, where lower values indicate
      simpler reading levels (for example, ~8.0 corresponds to an 8th-grade
      reading level).
    word_count: Total number of word tokens in the statement.
    sentence_count: Total number of non-empty sentences in the statement.
    syllable_count: Total syllable count across all words.
    avg_sentence_length: Mean words per sentence, rounded to 2 decimal places.
    avg_syllables_per_word: Mean syllables per word, rounded to 2 decimal
      places.
  """

  fkgl: float
  word_count: int
  sentence_count: int
  syllable_count: int
  avg_sentence_length: float
  avg_syllables_per_word: float


@dataclasses.dataclass(frozen=True, slots=True)
class SimplificationResult:
  """Result of simplifying a single proposal with readability assertions."""

  pre_simplified_statement: str
  post_simplified_statement: str
  pre_score: ReadabilityScore
  post_score: ReadabilityScore
  compliant: bool
  attempts: int
  violations: tuple[str, ...]


def _validate_positive_int(name: str, value: int) -> int:
  """Validates that `value` is an integer >= 1 (excluding bools)."""
  # Reject bool explicitly because bool subclasses int in Python.
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
  if value < 1:
    raise ValueError(f"{name} must be >= 1, got {value}.")
  return value


def _validate_thresholds(
    *,
    max_fkgl: float,
    max_sentences: int,
    max_words: int | None,
) -> None:
  """Validates readability threshold arguments early."""
  if isinstance(max_fkgl, bool) or not isinstance(max_fkgl, (int, float)):
    raise TypeError(
        f"max_fkgl must be a number, got {type(max_fkgl).__name__}."
    )
  if not math.isfinite(max_fkgl) or max_fkgl <= 0:
    raise ValueError(f"max_fkgl must be a finite number > 0, got {max_fkgl}.")
  _validate_positive_int("max_sentences", max_sentences)
  if max_words is not None:
    _validate_positive_int("max_words", max_words)


def _validate_retry_delay(retry_delay_seconds: float) -> None:
  """Validates that `retry_delay_seconds` is a finite non-negative number."""
  if isinstance(retry_delay_seconds, bool) or not isinstance(
      retry_delay_seconds, (int, float)
  ):
    raise TypeError(
        "retry_delay_seconds must be a number, got"
        f" {type(retry_delay_seconds).__name__}."
    )
  if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
    raise ValueError(
        "retry_delay_seconds must be a finite number >= 0, got"
        f" {retry_delay_seconds}."
    )


@functools.lru_cache(maxsize=8)
def _get_pyphen_dict(lang: str = "en") -> pyphen.Pyphen:
  """Returns a cached pyphen hyphenation dictionary for `lang`."""
  return pyphen.Pyphen(lang=lang)


def count_syllables(word: str, lang: str = "en") -> int:
  """Counts syllables in a word using pyphen hyphenation.

  Args:
    word: Input token to count syllables for.
    lang: Language code passed to `pyphen.Pyphen` (default `"en"`).

  Returns:
    Integer syllable count (0 if the token contains no alphabetic characters,
    otherwise at least 1).
  """
  clean = _NON_ALPHA_RE.sub("", word).lower()
  if not clean:
    return 0
  if len(clean) <= 3:
    return 1
  hyphenated = _get_pyphen_dict(lang).inserted(clean)
  return max(1, hyphenated.count("-") + 1)


def score_readability(text: str, lang: str = "en") -> ReadabilityScore:
  """Computes deterministic Flesch-Kincaid Grade Level (FKGL) metrics.

  Args:
    text: Plain-English text to score.
    lang: Language code passed to `count_syllables` / `pyphen.Pyphen` (default
      `"en"`).

  Returns:
    A ReadabilityScore with FKGL, word/sentence/syllable counts, and averages.
  """
  words = _WORD_RE.findall(text)
  sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
  if not words or not sentences:
    return ReadabilityScore(
        fkgl=0.0,
        word_count=0,
        sentence_count=0,
        syllable_count=0,
        avg_sentence_length=0.0,
        avg_syllables_per_word=0.0,
    )

  word_count = len(words)
  sentence_count = len(sentences)
  syllable_count = sum(count_syllables(w, lang=lang) for w in words)
  avg_sentence_length = word_count / sentence_count
  avg_syllables_per_word = syllable_count / word_count
  fkgl = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
  return ReadabilityScore(
      fkgl=round(fkgl, 2),
      word_count=word_count,
      sentence_count=sentence_count,
      syllable_count=syllable_count,
      avg_sentence_length=round(avg_sentence_length, 2),
      avg_syllables_per_word=round(avg_syllables_per_word, 2),
  )


def check_readability_compliance(
    score: ReadabilityScore,
    *,
    max_fkgl: float = DEFAULT_MAX_FKGL,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_words: int | None = None,
) -> tuple[bool, tuple[str, ...]]:
  """Checks whether a ReadabilityScore meets all configured ceilings.

  Args:
    score: Deterministic readability score to verify.
    max_fkgl: Maximum allowed Flesch-Kincaid Grade Level.
    max_sentences: Maximum allowed sentence count.
    max_words: Optional maximum allowed word count.

  Returns:
    Tuple of `(compliant, violations)` where `violations` is empty when
    compliant and contains actionable diagnostic messages otherwise.
  """
  _validate_thresholds(
      max_fkgl=max_fkgl, max_sentences=max_sentences, max_words=max_words
  )
  violations: list[str] = []
  if score.word_count == 0 or score.sentence_count == 0:
    violations.append("Statement is empty.")
    return False, tuple(violations)

  if score.fkgl > max_fkgl:
    violations.append(
        f"Flesch-Kincaid Grade Level was {score.fkgl:.2f} (must be <="
        f" {max_fkgl:g}, aiming for ~{TARGET_FKGL:g}). Replace unnecessarily"
        " complex multi-syllable jargon or break up overly long multi-clause"
        " sentences."
    )
  if score.sentence_count > max_sentences:
    violations.append(
        f"Sentence count was {score.sentence_count} (must be <="
        f" {max_sentences}, aiming for <= {TARGET_SENTENCES} when possible)."
    )
  if max_words is not None and score.word_count > max_words:
    violations.append(
        f"Word count was {score.word_count} (must be <= {max_words})."
    )
  return not violations, tuple(violations)


def clean_candidate_text(raw_text: str) -> str:
  """Trims markdown wrappers, quotes, and accidental bold title prefixes."""
  candidate = raw_text.strip()
  if candidate.startswith("```"):
    candidate = _FENCE_START_RE.sub("", candidate)
    candidate = _FENCE_END_RE.sub("", candidate).strip()
  if candidate.lower().startswith("rewrite:"):
    candidate = candidate[len("rewrite:") :].strip()
  candidate = candidate.strip('"').strip("'").strip()
  if candidate.startswith("**"):
    match = _BOLD_TITLE_PREFIX_RE.match(candidate)
    if match and match.group(1).strip():
      candidate = match.group(1).strip()
  candidate = candidate.strip('"').strip("'").strip()
  return " ".join(candidate.split())


def _compute_violation_penalty(
    score: ReadabilityScore,
    *,
    max_fkgl: float,
    max_sentences: int,
    max_words: int | None,
) -> float:
  """Computes a weighted violation distance for ranking non-compliant drafts."""
  penalty = (
      max(0.0, score.fkgl - max_fkgl) * _FKGL_PENALTY_WEIGHT
      + max(0, score.sentence_count - max_sentences) * _SENTENCE_PENALTY_WEIGHT
  )
  if max_words is not None:
    penalty += max(0, score.word_count - max_words) * _WORD_PENALTY_WEIGHT
  return penalty


def _format_context_block(additional_context: str | None) -> str:
  """Formats optional additional context into a prompt XML block."""
  if not additional_context or not additional_context.strip():
    return ""
  return _ADDITIONAL_CONTEXT_BLOCK.format(
      additional_context=additional_context.strip()
  )


def _build_initial_prompt(
    statement: str,
    *,
    max_fkgl: float,
    max_sentences: int,
    max_words: int | None,
    additional_context: str | None,
) -> str:
  """Builds the first-attempt simplification prompt for a proposal."""
  word_limit_clause = (
      f" and at most {max_words} words" if max_words is not None else ""
  )
  return _INITIAL_PROMPT_TEMPLATE.format(
      max_fkgl=max_fkgl,
      target_sentences=min(TARGET_SENTENCES, max_sentences),
      max_sentences=max_sentences,
      word_limit_clause=word_limit_clause,
      context_block=_format_context_block(additional_context),
      statement=statement,
  )


def _build_retry_prompt(
    original_statement: str,
    previous_attempt: str,
    violations: abc.Sequence[str],
    *,
    max_fkgl: float,
    max_sentences: int,
    max_words: int | None,
    additional_context: str | None,
) -> str:
  """Builds a retry prompt with deterministic readability violation feedback."""
  if max_words is not None:
    word_limit_requirement = (
        f"Use strictly no more than {max_words} words while preserving all"
        " substantive mechanisms, funding constraints, and qualifications."
    )
  else:
    word_limit_requirement = _NO_WORD_LIMIT_REQUIREMENT
  return _RETRY_PROMPT_TEMPLATE.format(
      original_statement=original_statement,
      previous_attempt=previous_attempt,
      violations_text=" ".join(violations),
      context_block=_format_context_block(additional_context),
      max_fkgl=max_fkgl,
      target_fkgl=min(TARGET_FKGL, max_fkgl),
      target_sentences=min(TARGET_SENTENCES, max_sentences),
      max_sentences=max_sentences,
      word_limit_requirement=word_limit_requirement,
  )


async def simplify_proposal_with_assertion(
    model: base_model.BaseModel,
    statement: str,
    *,
    max_fkgl: float = DEFAULT_MAX_FKGL,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_words: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    additional_context: str | None = None,
    system_prompt: str | None = None,
    retry_delay_seconds: float = 0.0,
) -> SimplificationResult:
  """Simplifies a proposal with deterministic readability verification.

  Statements that already satisfy all configured thresholds are returned
  verbatim with zero model calls. Otherwise, the model is invoked up to
  `max_attempts` readability iterations with exact violation feedback until a
  compliant rewrite is produced. Transient transport errors and blank model
  responses are retried transparently via `retry_utils.RetryingModel` without
  consuming readability iterations or bumping temperature. If all attempts fail
  compliance (or transport retries are exhausted), the lowest-penalty candidate
  seen so far is returned with `compliant=False` and diagnostic `violations`.

  Args:
    model: Polymorphic BaseModel instance used for rewriting.
    statement: Original actionable proposal text.
    max_fkgl: Maximum allowable Flesch-Kincaid Grade Level (default 9.5).
    max_sentences: Maximum allowable sentence count (default 6).
    max_words: Optional maximum allowable word count (default None).
    max_attempts: Maximum number of readability rewrite attempts (default 5),
      also used as `max_retries` when wrapping `model` in
      `retry_utils.RetryingModel`.
    additional_context: Optional domain or language guidance appended to
      prompts.
    system_prompt: Optional override for SIMPLIFY_SYSTEM_PROMPT.
    retry_delay_seconds: Delay in seconds forwarded to
      `retry_utils.RetryingModel` between transient model errors.

  Returns:
    A SimplificationResult containing pre/post statements, scores, compliance
    status, attempt count, and any remaining violations.

  Raises:
    TypeError: If numeric threshold arguments have invalid types.
    ValueError: If threshold, attempt, or delay arguments are out of range.
  """
  _validate_thresholds(
      max_fkgl=max_fkgl, max_sentences=max_sentences, max_words=max_words
  )
  _validate_positive_int("max_attempts", max_attempts)
  _validate_retry_delay(retry_delay_seconds)

  cleaned_input = clean_candidate_text(statement)
  pre_score = score_readability(cleaned_input)
  pre_compliant, pre_violations = check_readability_compliance(
      pre_score,
      max_fkgl=max_fkgl,
      max_sentences=max_sentences,
      max_words=max_words,
  )

  # Short-circuit blank inputs and already-compliant proposals without an LLM
  # call so compliant wording is never paraphrased gratuitously.
  if not cleaned_input or pre_compliant:
    return SimplificationResult(
        pre_simplified_statement=cleaned_input,
        post_simplified_statement=cleaned_input,
        pre_score=pre_score,
        post_score=pre_score,
        compliant=pre_compliant,
        attempts=0,
        violations=pre_violations,
    )

  retrying_model = (
      model
      if isinstance(model, retry_utils.RetryingModel)
      else retry_utils.RetryingModel(
          model,
          max_retries=max_attempts,
          retry_delay_seconds=retry_delay_seconds,
      )
  )
  effective_system_prompt = (
      system_prompt if system_prompt is not None else SIMPLIFY_SYSTEM_PROMPT
  )
  current_prompt = _build_initial_prompt(
      cleaned_input,
      max_fkgl=max_fkgl,
      max_sentences=max_sentences,
      max_words=max_words,
      additional_context=additional_context,
  )
  best_candidate = cleaned_input
  best_score = pre_score
  best_violations = pre_violations
  best_penalty = _compute_violation_penalty(
      pre_score,
      max_fkgl=max_fkgl,
      max_sentences=max_sentences,
      max_words=max_words,
  )

  attempts_used = 0
  for attempt_index in range(max_attempts):
    attempt_num = attempt_index + 1
    attempts_used = attempt_num
    response = await retrying_model.generate_content(
        prompt=current_prompt,
        run_name="proposal_simplification",
        temperature=_BASE_TEMPERATURE + attempt_index * _TEMPERATURE_STEP,
        system_prompt=effective_system_prompt,
    )
    if response.get("error"):
      logging.warning(
          "Proposal simplification attempt %d/%d failed after transport"
          " retries: %s",
          attempt_num,
          max_attempts,
          response["error"],
      )
      break

    candidate = clean_candidate_text(response.get("text") or "")
    if not candidate:
      logging.warning(
          "Proposal simplification attempt %d/%d yielded empty text after"
          " cleaning.",
          attempt_num,
          max_attempts,
      )
      continue

    candidate_score = score_readability(candidate)
    compliant, violations = check_readability_compliance(
        candidate_score,
        max_fkgl=max_fkgl,
        max_sentences=max_sentences,
        max_words=max_words,
    )
    if compliant:
      return SimplificationResult(
          pre_simplified_statement=cleaned_input,
          post_simplified_statement=candidate,
          pre_score=pre_score,
          post_score=candidate_score,
          compliant=True,
          attempts=attempt_num,
          violations=(),
      )

    penalty = _compute_violation_penalty(
        candidate_score,
        max_fkgl=max_fkgl,
        max_sentences=max_sentences,
        max_words=max_words,
    )
    if penalty < best_penalty:
      best_penalty = penalty
      best_candidate = candidate
      best_score = candidate_score
      best_violations = violations

    current_prompt = _build_retry_prompt(
        cleaned_input,
        candidate,
        violations,
        max_fkgl=max_fkgl,
        max_sentences=max_sentences,
        max_words=max_words,
        additional_context=additional_context,
    )

  logging.warning(
      "Proposal did not reach readability compliance after %d attempts"
      " (best FKGL=%.2f, sentences=%d, words=%d).",
      attempts_used,
      best_score.fkgl,
      best_score.sentence_count,
      best_score.word_count,
  )
  return SimplificationResult(
      pre_simplified_statement=cleaned_input,
      post_simplified_statement=best_candidate,
      pre_score=pre_score,
      post_score=best_score,
      compliant=False,
      attempts=attempts_used,
      violations=best_violations,
  )


async def simplify_proposals(
    model: base_model.BaseModel,
    statements: abc.Sequence[str],
    *,
    max_fkgl: float = DEFAULT_MAX_FKGL,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_words: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    additional_context: str | None = None,
    system_prompt: str | None = None,
    retry_delay_seconds: float = 0.0,
) -> list[SimplificationResult]:
  """Simplifies a sequence of proposals concurrently while preserving order.

  Args:
    model: Polymorphic BaseModel instance used for rewriting.
    statements: Ordered sequence of proposal statements to simplify.
    max_fkgl: Maximum allowable Flesch-Kincaid Grade Level (default 9.5).
    max_sentences: Maximum allowable sentence count (default 6).
    max_words: Optional maximum allowable word count (default None).
    max_attempts: Maximum number of LLM rewrite attempts per statement.
    max_concurrent_calls: Maximum concurrent statement simplifications (>= 1).
    additional_context: Optional domain or language guidance appended to
      prompts.
    system_prompt: Optional override for SIMPLIFY_SYSTEM_PROMPT.
    retry_delay_seconds: Delay in seconds forwarded to
      `retry_utils.RetryingModel` between transient model errors.

  Returns:
    List of SimplificationResult objects aligned 1-to-1 with `statements`.
  """
  _validate_thresholds(
      max_fkgl=max_fkgl, max_sentences=max_sentences, max_words=max_words
  )
  _validate_positive_int("max_attempts", max_attempts)
  _validate_retry_delay(retry_delay_seconds)
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  if not statements:
    return []

  semaphore = asyncio.Semaphore(validated_concurrency)

  async def _simplify_one(statement: str) -> SimplificationResult:
    async with semaphore:
      return await simplify_proposal_with_assertion(
          model,
          statement,
          max_fkgl=max_fkgl,
          max_sentences=max_sentences,
          max_words=max_words,
          max_attempts=max_attempts,
          additional_context=additional_context,
          system_prompt=system_prompt,
          retry_delay_seconds=retry_delay_seconds,
      )

  return await asyncio.gather(*(_simplify_one(s) for s in statements))


async def simplify_proposals_dataframe(
    model: base_model.BaseModel,
    df: pd.DataFrame,
    *,
    statement_column: str = "proposition",
    max_fkgl: float = DEFAULT_MAX_FKGL,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_words: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    additional_context: str | None = None,
    system_prompt: str | None = None,
    retry_delay_seconds: float = 0.0,
) -> pd.DataFrame:
  """Simplifies proposals in a DataFrame without mutating the input.

  Args:
    model: Polymorphic BaseModel instance used for rewriting.
    df: Input DataFrame containing proposal statements. Never mutated in place.
    statement_column: Column name containing input statements.
    max_fkgl: Maximum allowable Flesch-Kincaid Grade Level (default 9.5).
    max_sentences: Maximum allowable sentence count (default 6).
    max_words: Optional maximum allowable word count (default None).
    max_attempts: Maximum number of LLM rewrite attempts per statement.
    max_concurrent_calls: Maximum concurrent statement simplifications (>= 1).
    additional_context: Optional domain or language guidance appended to
      prompts.
    system_prompt: Optional override for SIMPLIFY_SYSTEM_PROMPT.
    retry_delay_seconds: Delay in seconds forwarded to
      `retry_utils.RetryingModel` between transient model errors.

  Returns:
    A copy of `df` with added columns: `simplification`,
    `pre_simplified_statement`, `post_simplified_statement`, `pre_fkgl`,
    `post_fkgl`, `readability_compliant`, and `simplification_attempts`.

  Raises:
    ValueError: If `statement_column` is missing from `df` or any threshold
      argument is invalid.
  """
  if statement_column not in df.columns:
    raise ValueError(
        f"Column {statement_column!r} not found in input DataFrame."
    )
  _validate_thresholds(
      max_fkgl=max_fkgl, max_sentences=max_sentences, max_words=max_words
  )
  _validate_positive_int("max_attempts", max_attempts)
  _validate_retry_delay(retry_delay_seconds)
  base_model.validate_max_concurrent_calls(max_concurrent_calls)

  output_df = df.copy()
  statements = [
      "" if pd.isna(value) else str(value)
      for value in output_df[statement_column]
  ]
  results = await simplify_proposals(
      model,
      statements,
      max_fkgl=max_fkgl,
      max_sentences=max_sentences,
      max_words=max_words,
      max_attempts=max_attempts,
      max_concurrent_calls=max_concurrent_calls,
      additional_context=additional_context,
      system_prompt=system_prompt,
      retry_delay_seconds=retry_delay_seconds,
  )

  output_df["simplification"] = [r.post_simplified_statement for r in results]
  output_df["pre_simplified_statement"] = [
      r.pre_simplified_statement for r in results
  ]
  output_df["post_simplified_statement"] = [
      r.post_simplified_statement for r in results
  ]
  output_df["pre_fkgl"] = [r.pre_score.fkgl for r in results]
  output_df["post_fkgl"] = [r.post_score.fkgl for r in results]
  output_df["readability_compliant"] = [r.compliant for r in results]
  output_df["simplification_attempts"] = [r.attempts for r in results]

  if results:
    compliant_count = sum(r.compliant for r in results)
    passthrough_count = sum(r.compliant and r.attempts == 0 for r in results)
    mean_pre_fkgl = sum(r.pre_score.fkgl for r in results) / len(results)
    mean_post_fkgl = sum(r.post_score.fkgl for r in results) / len(results)
    logging.info(
        "Simplified %d proposals: %d/%d compliant (%d passthrough), overall"
        " mean FKGL %.2f -> %.2f (all rows, including passthrough and"
        " non-compliant).",
        len(results),
        compliant_count,
        len(results),
        passthrough_count,
        mean_pre_fkgl,
        mean_post_fkgl,
    )
  return output_df
