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

"""Unit tests for corpus_tools."""

import json
import pathlib
import tempfile
import unittest

from src.proposition_refinement.agentic import candidate_drafts
from src.proposition_refinement.agentic import corpus_tools


class DirectoryModeCorpusToolsTest(unittest.TestCase):
  """Tests recursive directory mode, search, bounded reads, and access logs."""

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmpdir.cleanup)
    self.root = pathlib.Path(self._tmpdir.name) / "corpus"
    self.root.mkdir()

    (self.root / "README.md").write_text(
        "# Reference Corpus\nOverview of local transit budgets.\n"
    )
    (self.root / "briefs").mkdir()
    (self.root / "briefs" / "transit.md").write_text(
        "Line 1: Bus operations cost $12M annually.\n"
        "Line 2: Fare revenue covers 25 percent.\n"
        "Line 3: Dedicated shuttle grants expire in 2028.\n"
    )
    (self.root / "briefs" / "notes.txt").write_text(
        "Maintenance bay capacity is 40 vehicles.\n"
    )
    (self.root / "data").mkdir()
    (self.root / "data" / "metrics.csv").write_text(
        "route,ridership\nRouteA,1200\nRouteB,900\n"
    )
    (self.root / "data" / "summary.json").write_text(
        json.dumps({"routes": 2, "fare_revenue": "25%"})
    )
    # Non-indexed and hidden files should be ignored.
    (self.root / "briefs" / "draft.md.truncated").write_text("fare revenue")
    (self.root / "briefs" / "raw.bin").write_bytes(b"\x00\x01\x02")
    (self.root / ".hidden").mkdir()
    (self.root / ".hidden" / "secret.md").write_text("fare revenue hidden")

  def test_list_corpus_files_filters_categories_and_subdirs(self):
    tools = corpus_tools.LocalCorpusTools(self.root)

    all_listing = tools.list_corpus_files()
    self.assertIn("5 files (all):", all_listing)
    self.assertIn("README.md", all_listing)
    self.assertIn("briefs/transit.md", all_listing)
    self.assertIn("briefs/notes.txt", all_listing)
    self.assertIn("data/metrics.csv", all_listing)
    self.assertIn("data/summary.json", all_listing)
    self.assertNotIn(".truncated", all_listing)
    self.assertNotIn("raw.bin", all_listing)
    self.assertNotIn(".hidden", all_listing)

    docs_listing = tools.list_corpus_files(category="documents")
    self.assertIn("3 files (documents):", docs_listing)
    self.assertIn("briefs/transit.md", docs_listing)
    self.assertNotIn("data/metrics.csv", docs_listing)

    data_listing = tools.list_corpus_files(category="datasets")
    self.assertIn("2 files (datasets):", data_listing)
    self.assertIn("data/metrics.csv", data_listing)
    self.assertIn("data/summary.json", data_listing)
    self.assertNotIn("README.md", data_listing)

    subdir_listing = tools.list_corpus_files(category="briefs")
    self.assertIn("2 files (briefs):", subdir_listing)
    self.assertNotIn("README.md", subdir_listing)

  def test_search_corpus_supports_regex_literal_fallback_and_ordering(self):
    access_log = candidate_drafts.CorpusAccessLog()
    tools = corpus_tools.LocalCorpusTools(self.root, access_log=access_log)

    # Documents should appear before datasets in search output.
    res = tools.search_corpus(r"fare.?revenue")
    doc_pos = res.find("[briefs/transit.md]")
    data_pos = res.find("[data/summary.json]")
    self.assertGreaterEqual(doc_pos, 0)
    self.assertGreater(data_pos, doc_pos)
    self.assertCountEqual(
        access_log.files, ["briefs/transit.md", "data/summary.json"]
    )

    # File filter narrows results to matching relative paths.
    filtered = tools.search_corpus("fare", file_filter="transit")
    self.assertIn("[briefs/transit.md]", filtered)
    self.assertNotIn("[data/summary.json]", filtered)

    # Invalid regex falls back cleanly to literal search.
    literal_res = tools.search_corpus("[unclosed")
    self.assertIn("No matches found", literal_res)
    self.assertIn("Please provide a non-empty", tools.search_corpus("   "))

  def test_read_corpus_file_respects_line_range_and_bounds(self):
    long_file = self.root / "briefs" / "long.txt"
    long_file.write_text(
        "\n".join(f"line-{i:02d}-" + ("x" * 100) for i in range(1, 21))
    )
    access_log = candidate_drafts.CorpusAccessLog()
    tools = corpus_tools.LocalCorpusTools(
        self.root,
        access_log=access_log,
        max_read_lines_cap=3,
        max_line_chars=25,
    )

    output = tools.read_corpus_file(
        "briefs/long.txt", start_line=5, max_lines=50
    )
    lines = output.splitlines()
    self.assertEqual(lines[0], "=== briefs/long.txt (Lines 5..7 of 20) ===")
    self.assertEqual(len(lines), 4)
    self.assertTrue(lines[1].startswith("5: line-05-"))
    self.assertLessEqual(len(lines[1]), len("5: ") + 25)
    self.assertIn("briefs/long.txt", access_log.files)

  def test_search_respects_per_file_and_total_caps(self):
    dense = self.root / "briefs" / "dense.txt"
    dense.write_text("\n".join(f"budget item {i}" for i in range(10)))
    tools = corpus_tools.LocalCorpusTools(
        self.root,
        max_search_matches=4,
        max_search_matches_per_file=2,
        max_snippet_chars=20,
    )
    res = tools.search_corpus("budget")
    self.assertIn("more matches truncated", res)


class ManifestModeCorpusToolsTest(unittest.TestCase):
  """Tests config.json manifest listing, PDF redirection, and validation."""

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmpdir.cleanup)
    self.root = pathlib.Path(self._tmpdir.name) / "manifest_corpus"
    self.root.mkdir()

    (self.root / "budget_report.txt").write_text(
        "=== Page 1 ===\nCapital reserve balance is $5M.\n"
    )
    (self.root / "budget_report.pdf").write_bytes(b"%PDF-1.4 mock")
    (self.root / "rates.csv").write_text("tier,rate\nbase,10\n")
    (self.root / "logs").mkdir()
    (self.root / "logs" / "trace.md").write_text("Raw debug trace note.\n")

    manifest = {
        "file_sources": {
            "budget_report.pdf": {
                "file_type": "pdf",
                "text_path": "budget_report.txt",
                "description": "Original PDF report.",
            },
            "budget_report.txt": {
                "file_type": "txt",
                "description": "Extracted text of the annual budget report.",
            },
            "rates.csv": {
                "file_type": "csv",
                "bytes": 18,
                "description": "Current fee schedule by tier.",
            },
            "logs/trace.md": {
                "file_type": "md",
                "description": "Low-priority trace log.",
            },
        }
    }
    (self.root / corpus_tools.CORPUS_MANIFEST_FILENAME).write_text(
        json.dumps(manifest)
    )

  def test_manifest_listing_search_and_pdf_redirect(self):
    access_log = candidate_drafts.CorpusAccessLog()
    list_fn, search_fn, read_fn = corpus_tools.make_corpus_tools(
        self.root,
        access_log=access_log,
        low_priority_prefixes=("logs/",),
    )

    listing = list_fn("all")
    self.assertIn("2 files (all):", listing)
    self.assertIn("budget_report.txt", listing)
    self.assertIn("Extracted text of the annual budget report.", listing)
    self.assertIn("rates.csv", listing)
    self.assertNotIn("budget_report.pdf", listing)
    self.assertNotIn("logs/trace.md", listing)

    other_listing = list_fn("other")
    self.assertIn("logs/trace.md", other_listing)
    self.assertIn("Unknown category", list_fn("bogus"))

    search_out = search_fn("Capital reserve", scope="documents")
    self.assertIn("[budget_report.txt]", search_out)

    # Reading the PDF manifest entry redirects to the extracted text file.
    pdf_read = read_fn("budget_report.pdf", start_line=1, max_lines=5)
    self.assertIn("=== budget_report.txt (Lines 1..2 of 2) ===", pdf_read)
    self.assertIn("Capital reserve balance", pdf_read)
    self.assertIn("budget_report.txt", access_log.files)
    self.assertIn("budget_report.pdf", access_log.files)

  def test_malformed_manifest_and_invalid_caps_raise(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      bad_root = pathlib.Path(tmpdir)
      (bad_root / corpus_tools.CORPUS_MANIFEST_FILENAME).write_text(
          json.dumps({"not_file_sources": []})
      )
      with self.assertRaises(ValueError):
        corpus_tools.load_corpus_manifest(bad_root)

    with self.assertRaises(ValueError):
      corpus_tools.LocalCorpusTools(self.root, max_read_lines_cap=0)


class PathTraversalSecurityTest(unittest.TestCase):
  """Tests strict containment blocking traversal, absolute paths, and links."""

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmpdir.cleanup)
    self.base = pathlib.Path(self._tmpdir.name)
    self.root = self.base / "corpus"
    self.root.mkdir()
    (self.root / "allowed.md").write_text("Safe reference content.\n")

    # Outside file and sibling directory sharing the same path string prefix.
    self.secret = self.base / "secret.txt"
    self.secret.write_text("TOP SECRET\n")
    self.sibling = self.base / "corpus_sibling"
    self.sibling.mkdir()
    (self.sibling / "leak.txt").write_text("SIBLING SECRET\n")

    # Symlink inside corpus pointing outside corpus root.
    (self.root / "escape_link.md").symlink_to(self.secret)

  def test_rejects_parent_absolute_symlink_and_sibling_traversal(self):
    tools = corpus_tools.LocalCorpusTools(self.root)

    # Symlink escape must not appear in listings or searches.
    self.assertNotIn("escape_link.md", tools.list_corpus_files())
    self.assertIn("No matches found", tools.search_corpus("TOP SECRET"))

    unsafe_paths = [
        "../secret.txt",
        "sub/../../secret.txt",
        "../corpus_sibling/leak.txt",
        str(self.secret),
        "/etc/passwd",
        "..\\secret.txt",
        "allowed.md\x00.txt",
        "escape_link.md",
    ]
    for path in unsafe_paths:
      response = tools.read_corpus_file(path)
      self.assertIn("Invalid path", response, msg=f"Failed to block {path!r}")
      self.assertNotIn("SECRET", response)

    # Legitimate file inside corpus remains readable.
    ok = tools.read_corpus_file("allowed.md")
    self.assertIn("Safe reference content.", ok)

  def test_rejects_manifest_pdf_redirect_escaping_corpus(self):
    manifest = {
        "file_sources": {
            "escape.pdf": {
                "file_type": "pdf",
                "text_path": "../secret.txt",
                "description": "Malicious redirect.",
            }
        }
    }
    (self.root / corpus_tools.CORPUS_MANIFEST_FILENAME).write_text(
        json.dumps(manifest)
    )
    tools = corpus_tools.LocalCorpusTools(self.root)
    response = tools.read_corpus_file("escape.pdf")
    self.assertIn("Invalid path", response)
    self.assertNotIn("TOP SECRET", response)


if __name__ == "__main__":
  unittest.main()
