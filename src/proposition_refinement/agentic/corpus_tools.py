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

"""Domain-agnostic local corpus filesystem tools for agentic refinement.

Provides read-only list, search, and bounded line-range read tools over a local
directory tree of reference documents and datasets. Supports two corpus layouts:

1. **Manifest mode:** When ``config.json`` exists at the corpus root with a
   ``file_sources`` mapping describing named entries (``file_type``, optional
   ``description``, ``bytes``, and ``text_path`` for extracted PDF text), tool
   listings and searches use the manifest metadata.
2. **Directory mode:** When no ``config.json`` manifest is present, recursively
   indexes ``.md``, ``.txt``, ``.json``, and ``.csv`` files under the root.

All file reads enforce strict path containment inside the resolved corpus root,
blocking ``..`` traversal, absolute paths, and symlinks pointing outside the
corpus.
"""

from collections.abc import Callable
import json
import logging
import pathlib
import re
from typing import Any

from src.proposition_refinement.agentic import candidate_drafts

CORPUS_MANIFEST_FILENAME: str = "config.json"
INDEXED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".json", ".csv"})
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})
DATASET_EXTENSIONS: frozenset[str] = frozenset({".csv", ".json"})
SEARCHABLE_MANIFEST_TYPES: frozenset[str] = frozenset(
    {"md", "txt", "json", "csv"}
)
DEFAULT_MAX_READ_LINES: int = 100
MAX_READ_LINES_CAP: int = 150
MAX_READ_LINE_CHARS: int = 2000
MAX_SEARCH_MATCHES: int = 20
MAX_SEARCH_MATCHES_PER_FILE: int = 5
MAX_SNIPPET_LINE_CHARS: int = 220

_EXCLUDED_MARKERS: tuple[str, ...] = (".truncated",)
_STANDARD_CATEGORIES: frozenset[str] = frozenset(
    {"all", "documents", "datasets"}
)


def load_corpus_manifest(corpus_root: pathlib.Path) -> dict[str, Any] | None:
  """Loads a corpus manifest (`config.json`) if present at `corpus_root`.

  Args:
    corpus_root: Resolved corpus root directory.

  Returns:
    Parsed manifest dictionary when ``config.json`` exists, otherwise None.

  Raises:
    ValueError: If ``config.json`` exists but lacks a dict ``file_sources``
      mapping or contains invalid JSON.
  """
  manifest_path = corpus_root / CORPUS_MANIFEST_FILENAME
  if not manifest_path.is_file():
    return None
  try:
    manifest = json.loads(manifest_path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(
        f"Invalid JSON in corpus manifest {manifest_path}: {exc}"
    ) from exc
  if not isinstance(manifest, dict) or not isinstance(
      manifest.get("file_sources"), dict
  ):
    raise ValueError(
        f"Corpus manifest missing dict 'file_sources': {manifest_path}"
    )
  return manifest


def _is_unsafe_rel_path(raw_path: str) -> bool:
  """Returns True if `raw_path` attempts absolute or parent-dir traversal."""
  stripped = raw_path.strip()
  if not stripped or "\x00" in stripped:
    return True
  posix = stripped.replace("\\", "/")
  if posix.startswith("/") or pathlib.PurePath(stripped).is_absolute():
    return True
  return ".." in pathlib.PurePosixPath(posix).parts


class LocalCorpusTools:
  """Safe, bounded read-only tools for exploring a local reference corpus."""

  def __init__(
      self,
      corpus_dir: pathlib.Path | str,
      *,
      access_log: candidate_drafts.CorpusAccessLog | None = None,
      max_read_lines_cap: int = MAX_READ_LINES_CAP,
      max_line_chars: int = MAX_READ_LINE_CHARS,
      max_search_matches: int = MAX_SEARCH_MATCHES,
      max_search_matches_per_file: int = MAX_SEARCH_MATCHES_PER_FILE,
      max_snippet_chars: int = MAX_SNIPPET_LINE_CHARS,
      low_priority_prefixes: tuple[str, ...] = (),
  ) -> None:
    """Initializes corpus tools rooted at `corpus_dir`.

    Args:
      corpus_dir: Root directory containing reference documents and datasets.
      access_log: Optional CorpusAccessLog recording files hit by searches or
        reads.
      max_read_lines_cap: Maximum number of lines returned by one read call.
      max_line_chars: Maximum characters returned per line on read.
      max_search_matches: Maximum total search excerpts returned per query.
      max_search_matches_per_file: Maximum excerpts returned per single file.
      max_snippet_chars: Maximum characters per line in a search snippet.
      low_priority_prefixes: Optional relative path prefixes excluded from
        default ``all`` listings/searches in manifest mode.

    Raises:
      ValueError: If any numeric cap is < 1 or a present ``config.json``
        manifest is invalid.
    """
    for name, value in (
        ("max_read_lines_cap", max_read_lines_cap),
        ("max_line_chars", max_line_chars),
        ("max_search_matches", max_search_matches),
        ("max_search_matches_per_file", max_search_matches_per_file),
        ("max_snippet_chars", max_snippet_chars),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")

    self.corpus_root = pathlib.Path(corpus_dir).resolve()
    self.access_log = access_log
    if self.access_log is not None and self.access_log.root is None:
      self.access_log.root = self.corpus_root
    self.max_read_lines_cap = max_read_lines_cap
    self.max_line_chars = max_line_chars
    self.max_search_matches = max_search_matches
    self.max_search_matches_per_file = max_search_matches_per_file
    self.max_snippet_chars = max_snippet_chars
    self.low_priority_prefixes = low_priority_prefixes
    self.manifest = (
        load_corpus_manifest(self.corpus_root)
        if self.corpus_root.is_dir()
        else None
    )
    self._lines_cache: dict[pathlib.Path, list[str]] = {}

  def _record_access(self, rel_path: str) -> None:
    """Records a corpus file access if an access log was supplied."""
    if self.access_log is not None:
      self.access_log.record(rel_path)

  def _is_excluded(self, rel_path: str) -> bool:
    """Returns True if `rel_path` carries an excluded marker."""
    return any(marker in rel_path for marker in _EXCLUDED_MARKERS)

  def _is_low_priority(self, rel_path: str) -> bool:
    """Returns True if `rel_path` sits under a low-priority prefix."""
    return bool(self.low_priority_prefixes) and rel_path.startswith(
        self.low_priority_prefixes
    )

  def _resolve_contained_path(self, rel_path: str) -> pathlib.Path | None:
    """Resolves `rel_path` strictly inside `self.corpus_root`, or None."""
    if _is_unsafe_rel_path(rel_path):
      return None
    candidate = (self.corpus_root / rel_path.strip()).resolve()
    if not candidate.is_relative_to(self.corpus_root):
      return None
    return candidate

  def _cached_lines(self, path: pathlib.Path) -> list[str]:
    """Reads and caches a UTF-8 text file's lines for this tool instance."""
    lines = self._lines_cache.get(path)
    if lines is None:
      lines = path.read_text(errors="replace").splitlines()
      self._lines_cache[path] = lines
    return lines

  def _manifest_category_ok(
      self, rel_path: str, entry: dict[str, Any], category: str
  ) -> bool:
    """Checks whether a manifest entry matches `category`."""
    file_type = str(entry.get("file_type", "")).lower()
    if category == "all":
      return not self._is_low_priority(rel_path)
    if category == "documents":
      return file_type in ("md", "txt") and not self._is_low_priority(rel_path)
    if category == "datasets":
      return file_type in ("csv", "json") and not self._is_low_priority(
          rel_path
      )
    if category == "other":
      return self._is_low_priority(rel_path)
    return False

  def _discover_directory_files(self) -> list[tuple[str, pathlib.Path]]:
    """Recursively indexes supported files strictly inside `corpus_root`."""
    if not self.corpus_root.is_dir():
      return []
    discovered: list[tuple[str, pathlib.Path]] = []
    for path in sorted(self.corpus_root.rglob("*")):
      if not path.is_file():
        continue
      rel = path.relative_to(self.corpus_root)
      # Skip hidden directories and files (e.g. .git or .DS_Store).
      if any(part.startswith(".") for part in rel.parts):
        continue
      rel_posix = rel.as_posix()
      if self._is_excluded(rel_posix):
        continue
      if path.suffix.lower() not in INDEXED_EXTENSIONS:
        continue
      resolved = path.resolve()
      if not resolved.is_relative_to(self.corpus_root):
        continue
      discovered.append((rel_posix, resolved))
    return discovered

  def _directory_category_ok(
      self, rel_posix: str, path: pathlib.Path, category: str
  ) -> bool:
    """Checks whether a discovered file matches `category` or subdir prefix."""
    suffix = path.suffix.lower()
    if category == "all":
      return True
    if category == "documents":
      return suffix in DOCUMENT_EXTENSIONS
    if category == "datasets":
      return suffix in DATASET_EXTENSIONS
    prefix = category.strip().strip("/")
    if not prefix or _is_unsafe_rel_path(prefix):
      return False
    return rel_posix == prefix or rel_posix.startswith(f"{prefix}/")

  def list_corpus_files(self, category: str = "all") -> str:
    """Lists available files in the local reference corpus.

    Args:
      category: Filter such as ``all`` (default), ``documents`` (``.md`` and
        ``.txt``), ``datasets`` (``.csv`` and ``.json``), or in directory mode
        a relative subdirectory path prefix.

    Returns:
      Formatted listing of corpus files with types, byte sizes, and optional
      manifest descriptions.
    """
    if not self.corpus_root.is_dir():
      return f"Corpus directory not found: {self.corpus_root}"

    if self.manifest is not None:
      valid = _STANDARD_CATEGORIES | (
          {"other"} if self.low_priority_prefixes else set()
      )
      if category not in valid:
        return f"Unknown category '{category}'. Use one of: {sorted(valid)}."
      entries: dict[str, dict[str, Any]] = self.manifest["file_sources"]
      blocks: list[str] = []
      for rel, entry in entries.items():
        file_type = str(entry.get("file_type", "")).lower()
        if file_type == "pdf" or self._is_excluded(rel):
          continue
        if not self._manifest_category_ok(rel, entry, category):
          continue
        size = entry.get("bytes")
        if size is None:
          resolved = self._resolve_contained_path(rel)
          if resolved is not None and resolved.is_file():
            size = resolved.stat().st_size
        description = str(entry.get("description", "")).strip()
        size_part = f", {int(size):,} bytes" if size is not None else ""
        header = f"- {rel} [{file_type or 'file'}{size_part}]"
        blocks.append(f"{header}\n  {description}" if description else header)
      if not blocks:
        return "No matching files found."
      return f"{len(blocks)} files ({category}):\n" + "\n".join(blocks)

    lines: list[str] = []
    for rel_posix, resolved in self._discover_directory_files():
      if not self._directory_category_ok(rel_posix, resolved, category):
        continue
      file_type = resolved.suffix.lstrip(".").lower()
      lines.append(
          f"- {rel_posix} [{file_type}, {resolved.stat().st_size:,} bytes]"
      )
    if not lines:
      return "No matching files found."
    return f"{len(lines)} files ({category}):\n" + "\n".join(lines)

  def search_corpus(
      self, query: str, scope: str = "all", file_filter: str = ""
  ) -> str:
    """Searches indexed corpus files by keyword or regular expression.

    Document files (``.md``, ``.txt``) are searched before structured datasets
    (``.csv``, ``.json``). Results are bounded per file and in total to keep
    context usage predictable.

    Args:
      query: Keyword or regex pattern (case-insensitive). Invalid regular
        expressions fall back to literal matching.
      scope: Category filter (``all``, ``documents``, ``datasets``, or a
        subdirectory prefix in directory mode).
      file_filter: Optional case-insensitive substring that the relative file
        path must contain.

    Returns:
      Matching excerpts with relative file paths and 1-indexed line numbers.
    """
    if not self.corpus_root.is_dir():
      return f"Corpus directory not found: {self.corpus_root}"
    if not query or not query.strip():
      return "Please provide a non-empty search query."

    try:
      pattern = re.compile(query, re.IGNORECASE)
    except re.error:
      pattern = re.compile(re.escape(query), re.IGNORECASE)
    filter_needle = file_filter.strip().lower()

    candidates: list[tuple[bool, str, pathlib.Path]] = []
    if self.manifest is not None:
      valid = _STANDARD_CATEGORIES | (
          {"other"} if self.low_priority_prefixes else set()
      )
      if scope not in valid:
        return f"Unknown scope '{scope}'. Use one of: {sorted(valid)}."
      for rel, entry in self.manifest["file_sources"].items():
        file_type = str(entry.get("file_type", "")).lower()
        if file_type not in SEARCHABLE_MANIFEST_TYPES or self._is_excluded(rel):
          continue
        if not self._manifest_category_ok(rel, entry, scope):
          continue
        if filter_needle and filter_needle not in rel.lower():
          continue
        resolved = self._resolve_contained_path(rel)
        if resolved is None or not resolved.is_file():
          continue
        is_dataset = file_type in ("csv", "json")
        candidates.append((is_dataset, rel, resolved))
    else:
      for rel_posix, resolved in self._discover_directory_files():
        if not self._directory_category_ok(rel_posix, resolved, scope):
          continue
        if filter_needle and filter_needle not in rel_posix.lower():
          continue
        is_dataset = resolved.suffix.lower() in DATASET_EXTENSIONS
        candidates.append((is_dataset, rel_posix, resolved))

    # Search narrative documents before structured datasets.
    candidates.sort(key=lambda item: (item[0], item[1]))

    matches: list[str] = []
    for _, rel, path in candidates:
      try:
        lines = self._cached_lines(path)
      except OSError as exc:
        logging.warning("Skipping unreadable corpus file %s: %s", rel, exc)
        continue
      per_file = 0
      for idx, line in enumerate(lines):
        if not pattern.search(line):
          continue
        start = max(0, idx - 1)
        end = min(len(lines), idx + 3)
        snippet = "\n  ".join(
            f"L{ln + 1}: {lines[ln][:self.max_snippet_chars]}"
            for ln in range(start, end)
        )
        matches.append(f"[{rel}]\n  {snippet}")
        self._record_access(rel)
        per_file += 1
        if len(matches) >= self.max_search_matches:
          return "\n\n".join(matches)
        if per_file >= self.max_search_matches_per_file:
          matches.append(
              f"[{rel}] ... more matches truncated; use file_filter to narrow."
          )
          break

    if not matches:
      return (
          f"No matches found for query '{query}' (scope='{scope}',"
          f" file_filter='{file_filter}')."
      )
    return "\n\n".join(matches)

  def read_corpus_file(
      self,
      rel_path: str,
      start_line: int = 1,
      max_lines: int = DEFAULT_MAX_READ_LINES,
  ) -> str:
    """Reads a bounded slice of numbered lines from a file inside the corpus.

    Args:
      rel_path: Relative path within the corpus as returned by
        ``list_corpus_files``. In manifest mode, PDF paths with a ``text_path``
        are redirected to their extracted text file.
      start_line: 1-indexed starting line number.
      max_lines: Maximum number of lines to return (clamped to
        ``max_read_lines_cap``).

    Returns:
      Header showing line bounds followed by numbered lines, or an error string
      if the path escapes the corpus root or does not exist.
    """
    if not self.corpus_root.is_dir():
      return f"Corpus directory not found: {self.corpus_root}"

    raw_rel = (rel_path or "").strip()
    if _is_unsafe_rel_path(raw_rel):
      return (
          f"Invalid path '{rel_path}': path must stay within the corpus"
          " directory."
      )

    effective_rel = raw_rel
    if self.manifest is not None:
      entry = self.manifest["file_sources"].get(effective_rel)
      if (
          isinstance(entry, dict)
          and str(entry.get("file_type", "")).lower() == "pdf"
      ):
        redirected = str(entry.get("text_path", "")).strip()
        if redirected:
          effective_rel = redirected
          if _is_unsafe_rel_path(effective_rel):
            return (
                f"Invalid path '{rel_path}': redirected text path escapes the"
                " corpus directory."
            )

    target = self._resolve_contained_path(effective_rel)
    if target is None:
      return (
          f"Invalid path '{rel_path}': path must stay within the corpus"
          " directory."
      )
    if not target.is_file() or target.suffix.lower() == ".pdf":
      return (
          f"File not found in corpus: '{rel_path}'. Use list_corpus_files for"
          " valid paths."
      )

    try:
      lines = self._cached_lines(target)
    except OSError as exc:
      return f"Error reading '{rel_path}': {exc}"

    self._record_access(effective_rel)
    if effective_rel != raw_rel:
      self._record_access(raw_rel)

    start_idx = max(0, int(start_line) - 1)
    line_cap = max(1, min(int(max_lines), self.max_read_lines_cap))
    end_idx = min(len(lines), start_idx + line_cap)
    header = (
        f"=== {effective_rel} (Lines {start_idx + 1}..{end_idx} of"
        f" {len(lines)}) ==="
    )
    body = "\n".join(
        f"{i + 1}: {lines[i][:self.max_line_chars]}"
        for i in range(start_idx, end_idx)
    )
    return f"{header}\n{body}" if body else header

  def as_tools(self) -> list[Callable[..., str]]:
    """Returns `[list_corpus_files, search_corpus, read_corpus_file]`."""

    def list_corpus_files(category: str = "all") -> str:
      """Lists available files in the local reference corpus.

      Args:
        category: Optional filter ('all', 'documents', 'datasets', or a
          subdirectory prefix in directory mode).

      Returns:
        Formatted list of corpus files with types, sizes, and descriptions.
      """
      return self.list_corpus_files(category=category)

    def search_corpus(
        query: str, scope: str = "all", file_filter: str = ""
    ) -> str:
      """Searches the reference corpus for facts, constraints, or evidence.

      Args:
        query: Keyword or regex pattern to match (case-insensitive).
        scope: Optional filter ('all', 'documents', 'datasets', or a
          subdirectory prefix in directory mode).
        file_filter: Optional case-insensitive substring a file path must
          contain.

      Returns:
        Matching excerpts with relative file paths and line numbers.
      """
      return self.search_corpus(
          query=query, scope=scope, file_filter=file_filter
      )

    def read_corpus_file(
        rel_path: str,
        start_line: int = 1,
        max_lines: int = DEFAULT_MAX_READ_LINES,
    ) -> str:
      """Reads a bounded range of numbered lines from a corpus file.

      Args:
        rel_path: Relative path returned by list_corpus_files.
        start_line: 1-indexed starting line number.
        max_lines: Maximum number of lines to return.

      Returns:
        Numbered lines from the requested file.
      """
      return self.read_corpus_file(
          rel_path=rel_path, start_line=start_line, max_lines=max_lines
      )

    return [list_corpus_files, search_corpus, read_corpus_file]


def make_corpus_tools(
    corpus_dir: pathlib.Path | str,
    *,
    access_log: candidate_drafts.CorpusAccessLog | None = None,
    max_read_lines_cap: int = MAX_READ_LINES_CAP,
    max_line_chars: int = MAX_READ_LINE_CHARS,
    max_search_matches: int = MAX_SEARCH_MATCHES,
    max_search_matches_per_file: int = MAX_SEARCH_MATCHES_PER_FILE,
    max_snippet_chars: int = MAX_SNIPPET_LINE_CHARS,
    low_priority_prefixes: tuple[str, ...] = (),
) -> list[Callable[..., str]]:
  """Creates bounded local corpus tools bound to `corpus_dir`."""
  return LocalCorpusTools(
      corpus_dir,
      access_log=access_log,
      max_read_lines_cap=max_read_lines_cap,
      max_line_chars=max_line_chars,
      max_search_matches=max_search_matches,
      max_search_matches_per_file=max_search_matches_per_file,
      max_snippet_chars=max_snippet_chars,
      low_priority_prefixes=low_priority_prefixes,
  ).as_tools()
