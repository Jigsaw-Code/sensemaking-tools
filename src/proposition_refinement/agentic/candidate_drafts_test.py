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

"""Unit tests for candidate_drafts."""

import json
import pathlib
import tempfile
import unittest

from src.proposition_refinement.agentic import candidate_drafts

_SEED_STATEMENT = (
    "Fund annual street tree planting along major bus routes. Prioritize"
    " neighborhoods with the highest summer heat."
)


class SentenceAndComponentHelpersTest(unittest.TestCase):
  """Tests sentence counting, net sentence addition, and component parsing."""

  def test_count_sentences_handles_punctuation_and_blanks(self):
    self.assertEqual(candidate_drafts.count_sentences(""), 0)
    self.assertEqual(candidate_drafts.count_sentences("   "), 0)
    self.assertEqual(
        candidate_drafts.count_sentences("Expand evening bus routes."), 1
    )
    self.assertEqual(
        candidate_drafts.count_sentences(
            'Expand evening bus routes. "Riders need reliable schedules!"'
            " (Pilot first?) Yes."
        ),
        4,
    )

  def test_added_sentence_count_floors_at_zero(self):
    shorter = "Fund annual street tree planting along major bus routes."
    self.assertEqual(
        candidate_drafts.added_sentence_count(shorter, _SEED_STATEMENT), 0
    )
    two_more = (
        f"{_SEED_STATEMENT} Publish an annual map of tree survival. Allow"
        " residents to nominate blocks."
    )
    self.assertEqual(
        candidate_drafts.added_sentence_count(two_more, _SEED_STATEMENT), 2
    )

  def test_parse_component_lines_strips_markers_and_placeholders(self):
    raw = "\n".join([
        "- Public survival dashboard",
        "2. Resident block nomination form",
        "• None",
        "  N/A  ",
        "",
        "* Maintenance reserve",
    ])
    self.assertEqual(
        candidate_drafts.parse_component_lines(raw),
        [
            "Public survival dashboard",
            "Resident block nomination form",
            "Maintenance reserve",
        ],
    )


class CandidateDraftSlateTest(unittest.TestCase):
  """Tests verification gating, persistence, stacking tracking, and tools."""

  def test_draft_write_revision_and_json_persistence(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      slate_path = pathlib.Path(tmpdir) / "nested" / "slate.json"
      log = candidate_drafts.CorpusAccessLog(root=pathlib.Path(tmpdir))
      log.record("/briefs/budget.md")
      slate = candidate_drafts.CandidateDraftSlate(
          access_log=log,
          slate_path=slate_path,
          seed_statement=_SEED_STATEMENT,
      )

      first_msg = slate.write(
          candidate_id="c01",
          title="Public tree map",
          statement=(
              f"{_SEED_STATEMENT} Publish a public map of planting locations"
              " each fall."
          ),
          target_friction_resolved="Residents want transparency on placement.",
      )
      self.assertIn("Saved c01 as 'draft'", first_msg)
      self.assertEqual(slate.drafts["c01"].revisions, 1)

      verify_msg = slate.write(
          candidate_id="c01",
          title="Public tree map",
          statement=(
              f"{_SEED_STATEMENT} Publish a public map of planting locations"
              " each fall."
          ),
          target_friction_resolved="Residents want transparency on placement.",
          feasibility_assessment=(
              "Checked briefs/budget.md; public reporting uses existing GIS"
              " staff."
          ),
          status="verified",
          new_components="Annual public planting location map",
          seed_purpose_preserved=(
              "Yes: retains annual street tree planting along heat-affected bus"
              " routes."
          ),
      )
      self.assertIn("Revised c01 as 'verified'", verify_msg)
      self.assertEqual(slate.drafts["c01"].revisions, 2)
      self.assertEqual(slate.drafts["c01"].status, "verified")

      persisted = json.loads(slate_path.read_text())
      self.assertEqual(persisted["seed_statement"], _SEED_STATEMENT)
      self.assertEqual(persisted["accessed_corpus_files"], ["briefs/budget.md"])
      self.assertEqual(len(persisted["drafts"]), 1)
      self.assertEqual(len(persisted["history"]), 2)

  def test_invalid_inputs_and_rejected_gate(self):
    slate = candidate_drafts.CandidateDraftSlate(seed_statement=_SEED_STATEMENT)
    self.assertIn("ERROR", slate.write("", "t", "stmt", "concern"))
    self.assertIn(
        "ERROR", slate.write("c01", "t", "stmt", "concern", status="approved")
    )
    self.assertIn("ERROR", slate.write("c01", "t", "   ", "concern"))

    rejected_no_reason = slate.write(
        candidate_id="c02",
        title="Unfunded idea",
        statement=_SEED_STATEMENT,
        target_friction_resolved="Cost concern.",
        status="rejected",
    )
    self.assertIn("NOT VERIFIED", rejected_no_reason)
    self.assertEqual(slate.drafts["c02"].status, "draft")

    rejected_ok = slate.write(
        candidate_id="c02",
        title="Unfunded idea",
        statement=_SEED_STATEMENT,
        target_friction_resolved="Cost concern.",
        feasibility_assessment="Corpus notes funding cap prohibits this use.",
        status="rejected",
    )
    self.assertIn("Revised c02 as 'rejected'", rejected_ok)
    self.assertEqual(slate.drafts["c02"].status, "rejected")

  def test_verification_refuses_missing_fields_and_stacking(self):
    slate = candidate_drafts.CandidateDraftSlate(seed_statement=_SEED_STATEMENT)
    over_sentence_stmt = (
        f"{_SEED_STATEMENT} Add a citizen board. Add an audit. Add a tax"
        " rebate."
    )
    response = slate.write(
        candidate_id="c03",
        title="Overstacked proposal",
        statement=over_sentence_stmt,
        target_friction_resolved="",
        feasibility_assessment="",
        status="verified",
        new_components="Citizen board\nAnnual audit\nTax rebate",
        seed_purpose_preserved="Maybe keeps trees.",
    )
    self.assertIn("NOT VERIFIED", response)
    self.assertIn(candidate_drafts.STACKING_PROBLEM_TAG, response)
    self.assertIn("Split it:", response)
    self.assertEqual(slate.drafts["c03"].status, "draft")

    last_problems = slate.history[-1]["gate_problems"]
    self.assertEqual(len(last_problems), 5)

  def test_verification_allows_exact_two_component_and_two_sentence_boundary(
      self,
  ):
    slate = candidate_drafts.CandidateDraftSlate(seed_statement=_SEED_STATEMENT)
    boundary_stmt = (
        f"{_SEED_STATEMENT} Publish annual tree survival data. Reserve 15% of"
        " funds for watering."
    )
    msg = slate.write(
        candidate_id="c04",
        title="Watering and tracking",
        statement=boundary_stmt,
        target_friction_resolved="Worried newly planted trees will die.",
        feasibility_assessment=(
            "Checked maintenance notes; 15% watering reserve fits budget."
        ),
        status="verified",
        new_components="Annual survival report\n15% watering reserve",
        seed_purpose_preserved="Yes: keeps street tree planting on bus routes.",
    )
    self.assertIn("Saved c04 as 'verified'", msg)
    self.assertEqual(slate.drafts["c04"].status, "verified")

  def test_stacking_failures_tracks_unresolved_stacking_only(self):
    slate = candidate_drafts.CandidateDraftSlate(seed_statement=_SEED_STATEMENT)
    # Candidate c01 fails for stacking and stays unverified.
    slate.write(
        candidate_id="c01",
        title="Too many changes",
        statement=f"{_SEED_STATEMENT} Extra one. Extra two. Extra three.",
        target_friction_resolved="Accountability.",
        feasibility_assessment="Checked records.",
        status="verified",
        new_components="One\nTwo\nThree",
        seed_purpose_preserved="Yes: keeps core goal.",
    )
    # Candidate c02 fails for stacking first, then gets trimmed and verified.
    slate.write(
        candidate_id="c02",
        title="Trimmed later",
        statement=f"{_SEED_STATEMENT} Extra one. Extra two. Extra three.",
        target_friction_resolved="Maintenance.",
        feasibility_assessment="Checked records.",
        status="verified",
        new_components="One\nTwo\nThree",
        seed_purpose_preserved="Yes: keeps core goal.",
    )
    slate.write(
        candidate_id="c02",
        title="Trimmed later",
        statement=f"{_SEED_STATEMENT} Reserve 10% for watering.",
        target_friction_resolved="Maintenance.",
        feasibility_assessment="Checked records.",
        status="verified",
        new_components="Watering reserve",
        seed_purpose_preserved="Yes: keeps core goal.",
    )
    # Candidate c03 fails only for missing feasibility assessment.
    slate.write(
        candidate_id="c03",
        title="Missing check",
        statement=f"{_SEED_STATEMENT} Add quarterly updates.",
        target_friction_resolved="Transparency.",
        feasibility_assessment="",
        status="verified",
        new_components="Quarterly updates",
        seed_purpose_preserved="Yes: keeps core goal.",
    )

    failures = slate.stacking_failures()
    self.assertEqual([d.candidate_id for d in failures], ["c01"])

  def test_make_draft_slate_tools_and_view(self):
    slate = candidate_drafts.CandidateDraftSlate(seed_statement=_SEED_STATEMENT)
    write_tool, view_tool = candidate_drafts.make_draft_slate_tools(slate)
    self.assertIn("draft slate is empty", view_tool())

    write_tool(
        candidate_id="c01",
        title="Resident nominations",
        statement=f"{_SEED_STATEMENT} Let residents nominate priority blocks.",
        target_friction_resolved="Neighborhood input.",
    )
    rendered = view_tool()
    self.assertIn("## c01 [draft] Resident nominations (revision 1)", rendered)
    self.assertIn("Let residents nominate priority blocks.", rendered)


class AuditAndHardFilterTest(unittest.TestCase):
  """Tests slate auditing and fallback hard addition filters."""

  def setUp(self):
    super().setUp()
    self.log = candidate_drafts.CorpusAccessLog()
    self.log.record("notes/trees.md")
    self.slate = candidate_drafts.CandidateDraftSlate(
        access_log=self.log, seed_statement=_SEED_STATEMENT
    )
    self.verified_stmt = (
        f"{_SEED_STATEMENT} Reserve 10% of funds for tree watering."
    )
    self.slate.write(
        candidate_id="c01",
        title="Watering reserve",
        statement=self.verified_stmt,
        target_friction_resolved="Tree survival.",
        feasibility_assessment="Confirmed eligible under tree fund rules.",
        status="verified",
        new_components="10% watering reserve",
        seed_purpose_preserved="Yes: preserves tree planting along bus routes.",
    )
    self.slate.write(
        candidate_id="c02",
        title="Unverified draft",
        statement=f"{_SEED_STATEMENT} Add an annual volunteer planting day.",
        target_friction_resolved="Community engagement.",
    )
    # Over-component draft that remains unverified in the slate.
    self.slate.write(
        candidate_id="c03",
        title="Three components",
        statement=f"{_SEED_STATEMENT} Add board, audit, and rebate.",
        target_friction_resolved="Oversight.",
        feasibility_assessment="Checked.",
        status="verified",
        new_components="Board\nAudit\nRebate",
        seed_purpose_preserved="Yes: keeps planting.",
    )

  def test_audit_final_against_slate_matches_verified_and_flags_unverified(
      self,
  ):
    variations = [
        # Exact verified match by candidate_id.
        {
            "variation_id": "v01",
            "candidate_id": "c01",
            "description": self.verified_stmt.upper(),
        },
        # Fallback match when candidate_id is omitted.
        {
            "variation_id": "v02",
            "description": self.verified_stmt,
        },
        # Matches an unverified draft.
        {
            "variation_id": "v03",
            "candidate_id": "c02",
            "description": (
                f"{_SEED_STATEMENT} Add an annual volunteer planting day."
            ),
        },
        # Unmatched fabricated statement.
        {
            "variation_id": "v04",
            "candidate_id": "c01",
            "description": "Replace all buses with rail transit lines.",
        },
    ]

    audit = candidate_drafts.audit_final_against_slate(variations, self.slate)
    self.assertEqual(audit["n_variations"], 4)
    self.assertEqual(audit["n_verified_matches"], 2)
    self.assertEqual(audit["n_unverified"], 2)
    self.assertEqual(audit["n_drafts"], 3)
    self.assertEqual(audit["n_drafts_verified"], 1)
    self.assertEqual(audit["n_accessed_corpus_files"], 1)
    self.assertEqual(
        [
            {
                "variation_id": row["variation_id"],
                "matched_draft": row["matched_draft"],
                "draft_status": row["draft_status"],
                "is_verified": row["is_verified"],
            }
            for row in audit["variations"]
        ],
        [
            {
                "variation_id": "v01",
                "matched_draft": "c01",
                "draft_status": "verified",
                "is_verified": True,
            },
            {
                "variation_id": "v02",
                "matched_draft": "c01",
                "draft_status": "verified",
                "is_verified": True,
            },
            {
                "variation_id": "v03",
                "matched_draft": "c02",
                "draft_status": "draft",
                "is_verified": False,
            },
            {
                "variation_id": "v04",
                "matched_draft": None,
                "draft_status": None,
                "is_verified": False,
            },
        ],
    )

  def test_hard_filter_checks_sentences_and_declared_components(self):
    good = {
        "variation_id": "v01",
        "candidate_id": "c01",
        "description": self.verified_stmt,
    }
    too_many_sentences = {
        "variation_id": "v02",
        "description": f"{_SEED_STATEMENT} Extra one. Extra two. Extra three.",
    }
    too_many_components = {
        "variation_id": "v03",
        "candidate_id": "c03",
        "description": f"{_SEED_STATEMENT} Add board, audit, and rebate.",
    }
    inputs = [good, too_many_sentences, too_many_components]

    kept, dropped = candidate_drafts.apply_hard_filter(
        inputs, _SEED_STATEMENT, self.slate
    )
    self.assertEqual([v["variation_id"] for v in kept], ["v01"])
    self.assertCountEqual(
        [v["variation_id"] for v, _ in dropped], ["v02", "v03"]
    )
    dropped_by_id = {v["variation_id"]: problems for v, problems in dropped}
    self.assertTrue(any("more sentences" in p for p in dropped_by_id["v02"]))
    self.assertTrue(
        any("declared new components" in p for p in dropped_by_id["v03"])
    )
    # Input dicts are not mutated.
    self.assertEqual(good["variation_id"], "v01")


if __name__ == "__main__":
  unittest.main()
