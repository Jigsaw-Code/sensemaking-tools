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

"""Tests for segment_synthesis."""

from collections import abc
import io
import json
import re
from typing import Any
import unittest

import pandas as pd

from src.models import base_model
from src.transcripts import segment_synthesis
from src.transcripts import stance_classifier


class _FakeModel(base_model.BaseModel):
  """Deterministic fake BaseModel capturing prompts and run names."""

  def __init__(self, responder: abc.Callable[[str, str], dict[str, Any]]):
    super().__init__()
    self._responder = responder
    self.calls: list[tuple[str, str]] = []

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs,
  ) -> dict[str, Any]:
    del temperature, system_prompt, response_mime_type, response_schema, kwargs
    self.calls.append((run_name, prompt))
    return self._responder(prompt, run_name)

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: abc.Callable[[Any, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    del prompts, response_parser, max_concurrent_calls, retry_attempts, kwargs
    raise NotImplementedError("segment_synthesis uses generate_content.")


def _make_proposal() -> stance_classifier.Proposal:
  return stance_classifier.Proposal(
      proposal_id="corridor_upgrades",
      title="Protected Bike & Pedestrian Corridors",
      text="Build protected bike lanes and raised crosswalks along arterials.",
      topic="Street Safety",
  )


def _make_statements() -> list[stance_classifier.DeliberationStatement]:
  return [
      stance_classifier.DeliberationStatement(
          "s1", "P1", "Protected lanes keep cyclists safe.", "Room 01"
      ),
      stance_classifier.DeliberationStatement(
          "s2", "P2", "Raised crosswalks slow speeding cars.", "Room 01"
      ),
      stance_classifier.DeliberationStatement(
          "s3", "P3", "Prioritize routes near schools first.", "Room 02"
      ),
      stance_classifier.DeliberationStatement(
          "s4", "P4", "I appreciate safer streets overall.", "Room 02"
      ),
      stance_classifier.DeliberationStatement(
          "s5", "P5", "Removing turn lanes could delay deliveries.", "Room 03"
      ),
      stance_classifier.DeliberationStatement(
          "s6", "P6", "Extend park closing times in summer.", "Room 03"
      ),
  ]


def _make_classifications() -> dict[str, stance_classifier.ClassifiedStatement]:
  return {
      "s1": stance_classifier.ClassifiedStatement(
          "s1", stance_classifier.StanceLabel.SUPPORT
      ),
      "s2": stance_classifier.ClassifiedStatement(
          "s2", stance_classifier.StanceLabel.SUPPORT
      ),
      "s3": stance_classifier.ClassifiedStatement(
          "s3", stance_classifier.StanceLabel.SUPPORT
      ),
      "s4": stance_classifier.ClassifiedStatement(
          "s4", stance_classifier.StanceLabel.SUPPORT
      ),
      "s5": stance_classifier.ClassifiedStatement(
          "s5", stance_classifier.StanceLabel.CRITIQUE
      ),
      "s6": stance_classifier.ClassifiedStatement(
          "s6", stance_classifier.StanceLabel.UNRELATED
      ),
  }


class SegmentAbstractionTest(unittest.TestCase):
  """Tests for SegmentSpec loaders and participant-to-segment mappers."""

  def test_load_segment_specs_from_json_with_unsupervised_cluster_context(self):
    payload = {
        "segmentation_context": (
            "Unsupervised opinion clusters derived via PCA + k-means over"
            " pre-deliberation vote vectors."
        ),
        "segments": [
            {
                "id": "cluster_a",
                "label": "Cluster A",
                "description": "High support for active-mobility projects.",
            },
            {
                "id": "cluster_b",
                "label": "Cluster B",
                "description": "Cautious about road capacity reductions.",
            },
        ],
    }
    specs, context = segment_synthesis.load_segment_specs(
        io.StringIO(json.dumps(payload))
    )

    self.assertIn("PCA + k-means", context)
    self.assertEqual(
        specs,
        [
            segment_synthesis.SegmentSpec(
                id="cluster_a",
                label="Cluster A",
                description="High support for active-mobility projects.",
            ),
            segment_synthesis.SegmentSpec(
                id="cluster_b",
                label="Cluster B",
                description="Cautious about road capacity reductions.",
            ),
        ],
    )

  def test_load_segment_specs_from_flat_mapping_and_columns(self):
    specs, context = segment_synthesis.load_segment_specs(
        io.StringIO(
            json.dumps({
                "urban": "Urban Residents",
                "suburban": {
                    "label": "Suburban Residents",
                    "description": "Commuters living outside the urban core.",
                },
            })
        )
    )
    self.assertEqual(context, "")
    self.assertCountEqual([s.id for s in specs], ["urban", "suburban"])
    self.assertEqual(specs[1].label, "Suburban Residents")

    derived = segment_synthesis.segment_specs_from_columns(
        ["age_18_34", "age_65_plus"]
    )
    self.assertEqual(
        derived,
        [
            segment_synthesis.SegmentSpec(id="age_18_34", label="age_18_34"),
            segment_synthesis.SegmentSpec(
                id="age_65_plus", label="age_65_plus"
            ),
        ],
    )

  def test_load_segment_specs_rejects_duplicate_ids(self):
    with self.assertRaisesRegex(ValueError, "Duplicate SegmentSpec id"):
      segment_synthesis.load_segment_specs(
          io.StringIO(
              json.dumps([
                  {"id": "dup", "label": "First"},
                  {"id": "dup", "label": "Second"},
              ])
          )
      )

  def test_map_participants_categorical_auto_derives_specs(self):
    df = pd.DataFrame({
        "participant_id": ["P1", "P2", "P3", "P4"],
        "cluster": ["cluster_b", "cluster_a", "cluster_a", None],
    })
    mapping, specs = segment_synthesis.map_participants_to_segments(
        df, "participant_id", segment_column="cluster"
    )
    self.assertEqual([s.id for s in specs], ["cluster_a", "cluster_b"])
    self.assertEqual(mapping["P1"], frozenset({"cluster_b"}))
    self.assertEqual(mapping["P2"], frozenset({"cluster_a"}))
    self.assertEqual(mapping["P4"], frozenset())
    counts = segment_synthesis.compute_segment_counts(mapping, specs)
    self.assertEqual(counts, {"cluster_a": 2, "cluster_b": 1})

  def test_map_participants_overlapping_binary_columns(self):
    df = pd.DataFrame({
        "participant_id": ["P1", "P2", "P3"],
        "commuter": [1, 0, "yes"],
        "renter": [True, "false", 1],
    })
    specs_in = [
        segment_synthesis.SegmentSpec(
            id="commuter",
            label="Daily Commuters",
            description="Travels across corridors 4+ days/week.",
        ),
        segment_synthesis.SegmentSpec(
            id="renter",
            label="Renters",
            description="Rents primary residence.",
        ),
    ]
    mapping, specs_out = segment_synthesis.map_participants_to_segments(
        df,
        "participant_id",
        segment_columns=["commuter", "renter"],
        segment_specs=specs_in,
    )
    self.assertEqual(specs_out, specs_in)
    self.assertEqual(mapping["P1"], frozenset({"commuter", "renter"}))
    self.assertEqual(mapping["P2"], frozenset())
    self.assertEqual(mapping["P3"], frozenset({"commuter", "renter"}))
    counts = segment_synthesis.compute_segment_counts(mapping, specs_out)
    self.assertEqual(counts, {"commuter": 2, "renter": 2})

  def test_map_participants_validates_mutually_exclusive_column_args(self):
    df = pd.DataFrame({"participant_id": ["P1"], "g": ["a"]})
    with self.assertRaisesRegex(ValueError, "exactly one"):
      segment_synthesis.map_participants_to_segments(df, "participant_id")
    with self.assertRaisesRegex(ValueError, "exactly one"):
      segment_synthesis.map_participants_to_segments(
          df, "participant_id", segment_column="g", segment_columns=["g"]
      )

  def test_partition_statements_excludes_unrelated_and_handles_overlap(self):
    specs = segment_synthesis.segment_specs_from_columns(["seg_a", "seg_b"])
    participant_map = {
        "P1": frozenset({"seg_a", "seg_b"}),
        "P5": frozenset({"seg_b"}),
        "P6": frozenset({"seg_a"}),
    }
    partitioned = segment_synthesis.partition_statements_by_segment_and_stance(
        _make_statements(),
        _make_classifications(),
        participant_map,
        specs,
    )
    self.assertCountEqual(
        [
            s.statement_id
            for s in partitioned["seg_a"][stance_classifier.StanceLabel.SUPPORT]
        ],
        ["s1"],
    )
    self.assertCountEqual(
        [
            s.statement_id
            for s in partitioned["seg_b"][stance_classifier.StanceLabel.SUPPORT]
        ],
        ["s1"],
    )
    self.assertCountEqual(
        [
            s.statement_id
            for s in partitioned["seg_b"][
                stance_classifier.StanceLabel.CRITIQUE
            ]
        ],
        ["s5"],
    )
    # Unrelated statement s6 must be omitted from every relevant stance bucket.
    all_partitioned_ids = [
        stmt.statement_id
        for stance_dict in partitioned.values()
        for stmts in stance_dict.values()
        for stmt in stmts
    ]
    self.assertNotIn("s6", all_partitioned_ids)


class OpinionTreeAndViewFormattingTest(unittest.TestCase):
  """Tests for deterministic markdown rendering and segment view builders."""

  def test_sort_clusters_tie_breaking_and_other_last(self):
    clusters = (
        segment_synthesis.SubOpinionCluster(
            opinion_id="other",
            opinion_statement="Other",
            summary="Residual remarks.",
            statement_ids=("s1", "s2", "s3", "s4", "s5"),
            is_other=True,
        ),
        segment_synthesis.SubOpinionCluster(
            opinion_id="op_c",
            opinion_statement="Tie cluster C",
            summary="Tied at 2 quotes, supplied before op_a.",
            statement_ids=("s6", "s7"),
            is_other=False,
        ),
        segment_synthesis.SubOpinionCluster(
            opinion_id="op_d",
            opinion_statement="Small cluster D",
            summary="Only 1 quote.",
            statement_ids=("s8",),
            is_other=False,
        ),
        segment_synthesis.SubOpinionCluster(
            opinion_id="op_b",
            opinion_statement="Top regular cluster B",
            summary="3 quotes.",
            statement_ids=("s9", "s10", "s11"),
            is_other=False,
        ),
        segment_synthesis.SubOpinionCluster(
            opinion_id="op_a",
            opinion_statement="Tie cluster A",
            summary="Tied at 2 quotes, supplied after op_c.",
            statement_ids=("s12", "s13"),
            is_other=False,
        ),
    )

    ordered = segment_synthesis._sort_clusters(clusters)
    self.assertEqual(
        [c.opinion_id for c in ordered],
        ["op_b", "op_a", "op_c", "op_d", "other"],
    )

  def test_render_opinion_tree_markdown_bolds_only_other(self):
    tree = segment_synthesis.ProposalOpinionTree(
        proposal_id="corridor_upgrades",
        categories=(
            segment_synthesis.StanceOpinionTree(
                stance=stance_classifier.StanceLabel.SUPPORT,
                stance_title="Support",
                total_statements=6,
                clusters=(
                    segment_synthesis.SubOpinionCluster(
                        opinion_id="other",
                        opinion_statement="Other",
                        summary="General remarks.",
                        statement_ids=("s4", "s5", "s6"),
                        is_other=True,
                    ),
                    segment_synthesis.SubOpinionCluster(
                        opinion_id="op_2",
                        opinion_statement="Slow vehicle speeds near schools",
                        summary="Protects students.",
                        statement_ids=("s2",),
                        is_other=False,
                    ),
                    segment_synthesis.SubOpinionCluster(
                        opinion_id="op_1",
                        opinion_statement="Separate cyclists from fast traffic",
                        summary="Reduces collisions.",
                        statement_ids=("s1", "s3"),
                        is_other=False,
                    ),
                ),
            ),
        ),
    )

    rendered = segment_synthesis.render_opinion_tree_markdown(tree)
    expected = (
        "## Support (6 quotes)\n\n"
        "1. Separate cyclists from fast traffic (2 quotes)\n"
        "2. Slow vehicle speeds near schools (1 quotes)\n"
        "3. **Other** (3 quotes)"
    )
    self.assertEqual(rendered, expected)
    # Verify that ONLY **Other** uses bold formatting across the entire tree.
    bold_spans = re.findall(r"\*\*[^*]+\*\*", rendered)
    self.assertEqual(bold_spans, ["**Other**"])

  def test_segment_conditioned_view_builders_all_four_structures(self):
    specs = [
        segment_synthesis.SegmentSpec(
            id="cluster_1",
            label="Safety-First Cluster",
            description="Participants prioritizing street-safety investments.",
        ),
        segment_synthesis.SegmentSpec(
            id="cluster_2",
            label="Throughput-First Cluster",
            description="Participants prioritizing freight and travel times.",
        ),
    ]
    context = "Unsupervised opinion clusters from pre-survey votes."
    statements = _make_statements()
    classifications = _make_classifications()
    participant_map = {
        "P1": frozenset({"cluster_1"}),
        "P2": frozenset({"cluster_1"}),
        "P3": frozenset({"cluster_1"}),
        "P4": frozenset({"cluster_1"}),
        "P5": frozenset({"cluster_2"}),
    }
    counts = segment_synthesis.compute_segment_counts(participant_map, specs)
    partitioned = segment_synthesis.partition_statements_by_segment_and_stance(
        statements, classifications, participant_map, specs
    )

    # 1. Flat stance quotes view.
    flat_view = segment_synthesis.format_segment_conditioned_view(
        segment_synthesis.SegmentViewStructure.FLAT_QUOTES,
        specs,
        partitioned,
        counts,
        segmentation_context=context,
    )
    self.assertIn("### Segmentation Context", flat_view)
    self.assertIn(context, flat_view)
    self.assertIn(
        "## Safety-First Cluster (4 participants | 4 Support, 0 Critique,"
        " 0 Suggestion, 0 Observation statements)",
        flat_view,
    )
    self.assertIn(
        "## Throughput-First Cluster (1 participants | 0 Support, 1 Critique,"
        " 0 Suggestion, 0 Observation statements)",
        flat_view,
    )
    self.assertNotIn("---", flat_view)

    # 2 & 3. Sub-tree all-quotes and capped top-quotes views.
    subtrees = {
        "cluster_1": {
            stance_classifier.StanceLabel.SUPPORT: [
                segment_synthesis.SubOpinionCluster(
                    opinion_id="op_1",
                    opinion_statement="Physical separation improves safety",
                    summary="Participants cite crash risk.",
                    statement_ids=("s1", "s2", "s3"),
                    is_other=False,
                ),
                segment_synthesis.SubOpinionCluster(
                    opinion_id="other",
                    opinion_statement="Other",
                    summary="Broad endorsement.",
                    statement_ids=("s4",),
                    is_other=True,
                ),
            ]
        },
        "cluster_2": {
            stance_classifier.StanceLabel.CRITIQUE: [
                segment_synthesis.SubOpinionCluster(
                    opinion_id="op_1",
                    opinion_statement="Delivery vehicle bottlenecks",
                    summary="Concerns over turn-lane removals.",
                    statement_ids=("s5",),
                    is_other=False,
                )
            ]
        },
    }
    all_quotes_view = segment_synthesis.format_segment_conditioned_view(
        segment_synthesis.SegmentViewStructure.SUBTREE_ALL_QUOTES,
        specs,
        partitioned,
        counts,
        segmentation_context=context,
        segment_subtrees=subtrees,
    )
    self.assertIn("Protected lanes keep cyclists safe.", all_quotes_view)
    self.assertIn("Prioritize routes near schools first.", all_quotes_view)
    self.assertIn("2. **Other** (1 quotes)", all_quotes_view)
    self.assertEqual(
        re.findall(r"\*\*[^*]+\*\*", all_quotes_view), ["**Other**"]
    )

    top_quotes_view = segment_synthesis.format_segment_conditioned_view(
        segment_synthesis.SegmentViewStructure.SUBTREE_TOP_QUOTES,
        specs,
        partitioned,
        counts,
        segmentation_context=context,
        segment_subtrees=subtrees,
        max_quotes_per_cluster=1,
    )
    self.assertIn("Protected lanes keep cyclists safe.", top_quotes_view)
    self.assertNotIn("Prioritize routes near schools first.", top_quotes_view)

    # 4. Worldview & friction synthesis view.
    worldviews = {
        "cluster_1": segment_synthesis.SegmentWorldviewProfile(
            segment_id="cluster_1",
            worldview_and_support_drivers="Strong focus on vulnerable users.",
            primary_friction_points=("Phasing order across neighborhoods.",),
            key_conditions_for_support=("Start with school corridors.",),
        ),
        "cluster_2": segment_synthesis.SegmentWorldviewProfile(
            segment_id="cluster_2",
            worldview_and_support_drivers="Values smooth corridor operations.",
            primary_friction_points=("Turn-lane loss impacts deliveries.",),
            key_conditions_for_support=("Keep dedicated loading zones.",),
        ),
    }
    worldview_view = segment_synthesis.format_segment_conditioned_view(
        segment_synthesis.SegmentViewStructure.WORLDVIEW_SYNTHESIS,
        specs,
        partitioned,
        counts,
        segmentation_context=context,
        worldviews=worldviews,
    )
    self.assertIn("**Worldview & Support Drivers:**", worldview_view)
    self.assertIn("- Turn-lane loss impacts deliveries.", worldview_view)
    self.assertIn("- Keep dedicated loading zones.", worldview_view)


class SegmentSynthesisAsyncTest(unittest.IsolatedAsyncioTestCase):
  """Async tests for opinion trees, segment sub-trees, and worldview calls."""

  async def test_build_proposal_opinion_tree_routes_unassigned_to_other(self):
    focal = _make_proposal()
    statements = _make_statements()
    classifications = _make_classifications()

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt
      if run_name.startswith("discover_"):
        return {
            "text": json.dumps({
                "opinions": [{
                    "opinion_id": "op_1",
                    "opinion_statement": "Physical protection reduces risk",
                }]
            })
        }
      if run_name.startswith("assign_"):
        # Omit s4 intentionally to verify automatic fallback to Other.
        return {
            "text": json.dumps({
                "assignments": [
                    {"statement_id": "s1", "assigned_opinions": ["op_1"]},
                    {"statement_id": "s2", "assigned_opinions": ["op_1"]},
                    {"statement_id": "s3", "assigned_opinions": ["op_1"]},
                    {"statement_id": "s5", "assigned_opinions": ["op_1"]},
                ]
            })
        }
      return {
          "text": json.dumps(
              {"summary": "Grounded summary of participant views."}
          )
      }

    model = _FakeModel(responder)
    tree = await segment_synthesis.build_proposal_opinion_tree(
        model=model,
        focal_proposal=focal,
        statements=statements,
        classifications=classifications,
        max_concurrent_calls=2,
    )

    self.assertEqual(tree.proposal_id, focal.proposal_id)
    self.assertCountEqual(
        [cat.stance for cat in tree.categories],
        [
            stance_classifier.StanceLabel.SUPPORT,
            stance_classifier.StanceLabel.CRITIQUE,
        ],
    )
    support_cat = tree.categories[0]
    self.assertEqual(support_cat.total_statements, 4)
    self.assertEqual(len(support_cat.clusters), 2)
    self.assertFalse(support_cat.clusters[0].is_other)
    self.assertCountEqual(
        support_cat.clusters[0].statement_ids, ["s1", "s2", "s3"]
    )
    self.assertTrue(support_cat.clusters[1].is_other)
    self.assertEqual(support_cat.clusters[1].statement_ids, ("s4",))

    markdown = segment_synthesis.render_opinion_tree_markdown(tree)
    self.assertIn("## Support (4 quotes)", markdown)
    self.assertIn("1. Physical protection reduces risk (3 quotes)", markdown)
    self.assertIn("2. **Other** (1 quotes)", markdown)

  async def test_build_segment_subtrees_and_worldviews_propagate_context(self):
    focal = _make_proposal()
    demographic_specs = [
        segment_synthesis.SegmentSpec(
            id="transit_riders",
            label="Regular Transit Riders",
            description="Survey respondents riding transit at least weekly.",
        ),
        segment_synthesis.SegmentSpec(
            id="rural_commuters",
            label="Rural Commuters",
            description="Respondents commuting from outside the city core.",
        ),
    ]
    segmentation_context = (
        "Self-reported commuter cohorts from the pre-deliberation survey."
    )
    statements = _make_statements()
    classifications = _make_classifications()
    participant_map = {
        "P1": frozenset({"transit_riders"}),
        "P2": frozenset({"transit_riders"}),
        "P3": frozenset({"transit_riders"}),
        "P4": frozenset({"transit_riders"}),
    }
    counts = segment_synthesis.compute_segment_counts(
        participant_map, demographic_specs
    )
    partitioned = segment_synthesis.partition_statements_by_segment_and_stance(
        statements, classifications, participant_map, demographic_specs
    )

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt
      if run_name.startswith("segcluster_"):
        return {
            "text": json.dumps({
                "clusters": [{
                    "opinion_id": "op_1",
                    "opinion_statement": "Protected infrastructure first",
                    "summary": "Emphasizes physical separation.",
                    "statement_ids": ["s1", "s2", "s3"],
                }],
                "other_statement_ids": ["s4"],
                "other_summary": "General endorsement.",
            })
        }
      return {
          "text": json.dumps({
              "worldview_and_support_drivers": (
                  "Strongly values safe multimodal corridors."
              ),
              "primary_friction_points": ["Construction timeline uncertainty."],
              "key_conditions_for_support": [
                  "Sequence school corridors first."
              ],
          })
      }

    model = _FakeModel(responder)
    subtrees = await segment_synthesis.build_segment_subtrees(
        model=model,
        focal_proposal=focal,
        segments=demographic_specs,
        segment_stance_statements=partitioned,
        segmentation_context=segmentation_context,
        max_concurrent_calls=2,
    )
    worldviews = await segment_synthesis.synthesize_all_segment_worldviews(
        model=model,
        focal_proposal=focal,
        segments=demographic_specs,
        segment_counts=counts,
        segment_stance_statements=partitioned,
        segmentation_context=segmentation_context,
        max_concurrent_calls=2,
    )

    # Verify both SegmentSpec description and segmentation_context reach LLM.
    all_prompts = "\n".join(prompt for _, prompt in model.calls)
    self.assertIn("Regular Transit Riders", all_prompts)
    self.assertIn("riding transit at least weekly", all_prompts)
    self.assertIn(segmentation_context, all_prompts)
    self.assertNotIn("policy", all_prompts.lower())

    # Verify 4-statement bucket produced a main cluster + residual Other.
    rider_support = subtrees["transit_riders"][
        stance_classifier.StanceLabel.SUPPORT
    ]
    self.assertEqual(len(rider_support), 2)
    self.assertTrue(rider_support[1].is_other)
    self.assertEqual(rider_support[1].statement_ids, ("s4",))

    # Verify 0-statement segment returns a factual empty-evidence profile.
    rural_profile = worldviews["rural_commuters"]
    self.assertIn(
        "No proposal-relevant statements",
        rural_profile.worldview_and_support_drivers,
    )
    self.assertEqual(rural_profile.primary_friction_points, ())
    self.assertEqual(rural_profile.key_conditions_for_support, ())

    # Verify populated segment worldview matches model response.
    rider_profile = worldviews["transit_riders"]
    self.assertEqual(
        rider_profile.key_conditions_for_support,
        ("Sequence school corridors first.",),
    )

  async def test_small_segment_bucket_uses_single_cluster_and_retries_json(
      self,
  ):
    focal = _make_proposal()
    spec = segment_synthesis.SegmentSpec(
        id="cluster_x",
        label="Cluster X",
        description="Unsupervised cluster emphasizing freight access.",
    )
    statements = _make_statements()[:2]
    attempts = {"count": 0}

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt, run_name
      attempts["count"] += 1
      if attempts["count"] == 1:
        return {"text": "not valid json"}
      return {
          "text": json.dumps({
              "opinion_statement": (
                  "Protected crossings improve corridor safety"
              ),
              "summary": "Participants cite cyclist and pedestrian protection.",
          })
      }

    model = _FakeModel(responder)
    clusters = await segment_synthesis.cluster_segment_stance(
        model=model,
        focal_proposal=focal,
        segment=spec,
        stance=stance_classifier.StanceLabel.SUPPORT,
        statements=statements,
        segmentation_context="Unsupervised vote-matrix clustering.",
        max_retries=2,
    )

    self.assertEqual(attempts["count"], 2)
    self.assertEqual(len(clusters), 1)
    self.assertFalse(clusters[0].is_other)
    self.assertEqual(clusters[0].statement_ids, ("s1", "s2"))
    self.assertIn("Unsupervised vote-matrix clustering.", model.calls[0][1])

  def test_invalid_binary_membership_raises_value_error(self):
    df = pd.DataFrame({"participant_id": ["P1"], "flag": ["maybe"]})
    with self.assertRaisesRegex(ValueError, "Expected binary indicator"):
      segment_synthesis.map_participants_to_segments(
          df, "participant_id", segment_columns=["flag"]
      )

  def test_view_dispatcher_requires_matching_precomputed_payload(self):
    specs = segment_synthesis.segment_specs_from_columns(["seg_a"])
    with self.assertRaisesRegex(ValueError, "segment_subtrees is required"):
      segment_synthesis.format_segment_conditioned_view(
          segment_synthesis.SegmentViewStructure.SUBTREE_ALL_QUOTES,
          specs,
          {},
          {"seg_a": 0},
      )
    with self.assertRaisesRegex(ValueError, "worldviews is required"):
      segment_synthesis.format_segment_conditioned_view(
          segment_synthesis.SegmentViewStructure.WORLDVIEW_SYNTHESIS,
          specs,
          {},
          {"seg_a": 0},
      )


if __name__ == "__main__":
  unittest.main()
