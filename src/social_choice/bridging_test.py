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

"""Unit tests for src.social_choice.bridging."""

import argparse
import math
import unittest

import numpy as np
import pandas as pd

from src.social_choice import bridging


class BridgingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.participants_df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "region": ["North", "North", "South", "South", "East", "East"],
        "urban": [1, 0, 1, 0, 1, 0],
        "renter": [True, True, False, True, False, False],
    })
    # Polarizing has high overall approval (4/6) but 0% in East.
    # Bridging has lower overall approval (3/6) but 50% in every region.
    # Broad has 100% North, 50% South, 50% East (4/6 overall, 50% MinMax).
    self.approval_matrix = pd.DataFrame(
        {
            "Polarizing": [True, True, True, True, False, False],
            "Bridging": [True, False, True, False, True, False],
            "Broad": [True, True, True, False, True, False],
        },
        index=["p1", "p2", "p3", "p4", "p5", "p6"],
    )

  def test_minmax_and_bottleneck_categorical_segments(self):
    memberships = bridging.extract_segment_memberships(
        self.participants_df, segment_column="region"
    )
    scores = bridging.compute_bridging_scores(self.approval_matrix, memberships)

    polarizing = scores["Polarizing"]
    self.assertAlmostEqual(polarizing.minmax_approval, 0.0)
    self.assertEqual(polarizing.bottleneck_segment, "East")
    self.assertAlmostEqual(polarizing.overall_approval, 4.0 / 6.0)
    self.assertEqual(
        polarizing.segment_approvals,
        {"East": 0.0, "North": 1.0, "South": 1.0},
    )
    self.assertEqual(
        polarizing.segment_sizes,
        {"East": 2, "North": 2, "South": 2},
    )

    broad = scores["Broad"]
    self.assertAlmostEqual(broad.minmax_approval, 0.5)
    # East and South tie at 0.5; East wins deterministically by name.
    self.assertEqual(broad.bottleneck_segment, "East")
    self.assertAlmostEqual(broad.overall_approval, 4.0 / 6.0)

  def test_bottleneck_tie_breaks_deterministically_by_segment_name(self):
    matrix = pd.DataFrame(
        {"Cand": [True, False, True, False]},
        index=["p1", "p2", "p3", "p4"],
    )
    # Insert Zeta before Alpha to verify independence from insertion order.
    memberships = {"Zeta": {"p1", "p2"}, "Alpha": {"p3", "p4"}}
    scores = bridging.compute_bridging_scores(matrix, memberships)

    self.assertAlmostEqual(scores["Cand"].minmax_approval, 0.5)
    self.assertEqual(scores["Cand"].bottleneck_segment, "Alpha")

  def test_gic_product_and_geomean_unsmoothed(self):
    memberships = bridging.extract_segment_memberships(
        self.participants_df, segment_column="region"
    )
    scores = bridging.compute_bridging_scores(self.approval_matrix, memberships)

    # Polarizing has East = 0.0, so unsmoothed GIC collapses to 0.0.
    self.assertEqual(scores["Polarizing"].gic_product, 0.0)
    self.assertEqual(scores["Polarizing"].gic_geomean, 0.0)

    # Broad has North = 1.0, South = 0.5, East = 0.5 -> prod = 0.25.
    self.assertAlmostEqual(scores["Broad"].gic_product, 0.25)
    self.assertAlmostEqual(scores["Broad"].gic_geomean, 0.25 ** (1.0 / 3.0))

    # Bridging has 0.5 in all 3 regions -> prod = 0.125, geomean = 0.5.
    self.assertAlmostEqual(scores["Bridging"].gic_product, 0.125)
    self.assertAlmostEqual(scores["Bridging"].gic_geomean, 0.5)

  def test_gic_smoothing_prevents_zero_collapse_without_altering_minmax(self):
    memberships = bridging.extract_segment_memberships(
        self.participants_df, segment_column="region"
    )
    scores = bridging.compute_bridging_scores(
        self.approval_matrix, memberships, smoothing=1.0
    )

    polarizing = scores["Polarizing"]
    # Raw MinMax and segment_approvals remain unsmoothed.
    self.assertEqual(polarizing.minmax_approval, 0.0)
    self.assertEqual(polarizing.segment_approvals["East"], 0.0)

    # Smoothed rates: North (2+1)/(2+2)=3/4, South 3/4, East (0+1)/(2+2)=1/4.
    expected_product = 0.75 * 0.75 * 0.25
    self.assertAlmostEqual(polarizing.gic_product, expected_product)
    self.assertAlmostEqual(
        polarizing.gic_geomean, expected_product ** (1.0 / 3.0)
    )

  def test_gic_geomean_log_fallback_avoids_float_underflow(self):
    # 1200 segments each with approval 0.5 underflows 0.5**1200 to 0.0 in
    # IEEE-754 float64, while geometric mean should stay 0.5.
    matrix = pd.DataFrame({"Cand": [1.0, 0.0]}, index=["p1", "p2"])
    memberships = {f"seg_{i:04d}": {"p1", "p2"} for i in range(1200)}
    scores = bridging.compute_bridging_scores(matrix, memberships)

    self.assertEqual(scores["Cand"].gic_product, 0.0)
    self.assertAlmostEqual(scores["Cand"].gic_geomean, 0.5)

  def test_rank_by_bridging_tie_breaking_and_segment_columns(self):
    # HighOverallTie: North=0.5, South=1.0 -> minmax=0.5, overall=0.75.
    # TieAlpha / TieBeta: North=0.5, South=0.5 -> minmax=0.5, overall=0.5.
    # Polarized: North=0.0, South=1.0 -> minmax=0.0, overall=0.5.
    matrix = pd.DataFrame(
        {
            "TieBeta": [1, 0, 1, 0],
            "TieAlpha": [1, 0, 1, 0],
            "HighOverallTie": [1, 0, 1, 1],
            "Polarized": [0, 0, 1, 1],
        },
        index=["p1", "p2", "p3", "p4"],
    )
    memberships = {"North": {"p1", "p2"}, "South": {"p3", "p4"}}

    ranked = bridging.rank_by_bridging(
        matrix,
        memberships,
        metric="minmax",
        include_segment_rates=True,
    )

    self.assertEqual(
        ranked["candidate"].tolist(),
        ["HighOverallTie", "TieAlpha", "TieBeta", "Polarized"],
    )
    self.assertEqual(ranked["bridging_rank"].tolist(), [1, 2, 3, 4])
    self.assertAlmostEqual(ranked.loc[0, "segment_North"], 0.5)
    self.assertAlmostEqual(ranked.loc[0, "segment_South"], 1.0)
    self.assertAlmostEqual(ranked.loc[3, "minmax_approval"], 0.0)

  def test_rank_by_bridging_respects_chosen_metric(self):
    # CandA: S1=0.4, S2=1.0 -> minmax=0.4, gic_product=0.4, gic_geomean=~0.632.
    # CandB: S1=0.5, S2=0.5 -> minmax=0.5, gic_product=0.25, gic_geomean=0.5.
    matrix = pd.DataFrame(
        {
            "CandA": [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            "CandB": [1, 1, 0, 0, np.nan, 1, 1, 0, 0, np.nan],
        },
        index=[f"u{i}" for i in range(10)],
    )
    memberships = {
        "S1": {f"u{i}" for i in range(5)},
        "S2": {f"u{i}" for i in range(5, 10)},
    }

    ranked_minmax = bridging.rank_by_bridging(
        matrix, memberships, metric="minmax"
    )
    ranked_gic_prod = bridging.rank_by_bridging(
        matrix, memberships, metric="gic_product"
    )
    ranked_gic_geo = bridging.rank_by_bridging(
        matrix, memberships, metric="gic_geomean"
    )

    self.assertEqual(ranked_minmax["candidate"].tolist(), ["CandB", "CandA"])
    self.assertEqual(ranked_gic_prod["candidate"].tolist(), ["CandA", "CandB"])
    self.assertEqual(ranked_gic_geo["candidate"].tolist(), ["CandA", "CandB"])

  def test_nan_votes_and_min_segment_size_filtering(self):
    matrix = pd.DataFrame(
        {
            # SegA has 2 non-null votes (1/2 = 0.5), SegB has 1 non-null (0.0).
            "PartialSegB": [1.0, 0.0, 0.0, np.nan],
            # All NaN votes -> no segment meets min_segment_size.
            "AllMissing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=["p1", "p2", "p3", "p4"],
    )
    memberships = {"SegA": {"p1", "p2"}, "SegB": {"p3", "p4"}}

    ranked = bridging.rank_by_bridging(
        matrix, memberships, metric="minmax", min_segment_size=2
    )

    self.assertEqual(
        ranked["candidate"].tolist(), ["PartialSegB", "AllMissing"]
    )
    # SegB (n=1 < 2) is excluded for PartialSegB, leaving only SegA (0.5).
    self.assertAlmostEqual(ranked.loc[0, "minmax_approval"], 0.5)
    self.assertEqual(ranked.loc[0, "bottleneck_segment"], "SegA")
    self.assertAlmostEqual(ranked.loc[0, "overall_approval"], 1.0 / 3.0)
    self.assertEqual(ranked.loc[0, "bridging_rank"], 1)

    # AllMissing has NaN metrics and missing rank, sorted last.
    scores = bridging.compute_bridging_scores(
        matrix, memberships, min_segment_size=2
    )
    self.assertIsNone(scores["AllMissing"].bottleneck_segment)
    self.assertTrue(math.isnan(ranked.loc[1, "minmax_approval"]))
    self.assertTrue(pd.isna(ranked.loc[1, "bottleneck_segment"]))
    self.assertTrue(pd.isna(ranked.loc[1, "bridging_rank"]))

  def test_extract_segment_memberships_categorical_handles_missing_and_ints(
      self,
  ):
    df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4", None, "p6"],
        "cluster": [0, 1, None, "  ", 0, 1],
    })
    memberships = bridging.extract_segment_memberships(
        df, segment_column="cluster"
    )

    self.assertCountEqual(memberships.keys(), ["0", "1"])
    self.assertEqual(memberships["0"], {"p1"})
    self.assertEqual(memberships["1"], {"p2", "p6"})

  def test_extract_segment_memberships_overlapping_binary_columns(self):
    df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "urban": [1, "yes", "False", np.nan],
        "renter": [True, 0, "true", "0"],
        "empty_seg": [False, 0, "no", ""],
    })
    memberships = bridging.extract_segment_memberships(
        df, segment_columns=["urban", "renter", "empty_seg"]
    )

    # p1 belongs to both urban and renter (overlapping).
    self.assertEqual(memberships["urban"], {"p1", "p2"})
    self.assertEqual(memberships["renter"], {"p1", "p3"})
    self.assertEqual(memberships["empty_seg"], set())

    # Verify overlapping memberships compute accurate bridging scores.
    matrix = pd.DataFrame(
        {"Cand": [True, False, True, False]},
        index=["p1", "p2", "p3", "p4"],
    )
    scores = bridging.compute_bridging_scores(
        matrix, memberships, min_segment_size=1
    )
    # urban: p1=1, p2=0 -> 0.5; renter: p1=1, p3=1 -> 1.0; empty_seg skipped.
    self.assertEqual(
        scores["Cand"].segment_approvals, {"renter": 1.0, "urban": 0.5}
    )
    self.assertAlmostEqual(scores["Cand"].minmax_approval, 0.5)
    self.assertEqual(scores["Cand"].bottleneck_segment, "urban")
    self.assertAlmostEqual(scores["Cand"].overall_approval, 0.5)

  def test_parse_segment_columns(self):
    self.assertIsNone(bridging.parse_segment_columns(None))
    self.assertEqual(
        bridging.parse_segment_columns(["seg_a", "seg_b"]), ["seg_a", "seg_b"]
    )
    self.assertEqual(
        bridging.parse_segment_columns(["seg_a,seg_b", "seg_b, seg_c"]),
        ["seg_a", "seg_b", "seg_c"],
    )
    self.assertEqual(
        bridging.parse_segment_columns("seg_x, seg_y"), ["seg_x", "seg_y"]
    )
    with self.assertRaises(ValueError):
      bridging.parse_segment_columns([" , "])

  def test_empty_approval_matrix_returns_empty_dataframe_with_schema(self):
    empty_matrix = pd.DataFrame(index=["p1", "p2"])
    ranked = bridging.rank_by_bridging(
        empty_matrix,
        {"North": {"p1"}, "South": {"p2"}},
        include_segment_rates=True,
    )
    self.assertTrue(ranked.empty)
    self.assertEqual(
        ranked.columns.tolist(),
        [
            "candidate",
            "minmax_approval",
            "bottleneck_segment",
            "gic_product",
            "gic_geomean",
            "overall_approval",
            "bridging_rank",
            "segment_North",
            "segment_South",
        ],
    )

  def test_extract_segment_memberships_validation_errors(self):
    with self.assertRaisesRegex(ValueError, "either segment_column"):
      bridging.extract_segment_memberships(self.participants_df)

    with self.assertRaisesRegex(ValueError, "not both"):
      bridging.extract_segment_memberships(
          self.participants_df,
          segment_column="region",
          segment_columns=["urban"],
      )

    with self.assertRaisesRegex(ValueError, "not a string"):
      bridging.extract_segment_memberships(
          self.participants_df, segment_columns="urban"
      )

    with self.assertRaisesRegex(ValueError, "Participant ID column"):
      bridging.extract_segment_memberships(
          self.participants_df,
          participant_id_col="missing_id",
          segment_column="region",
      )

    with self.assertRaisesRegex(ValueError, "Segment column 'missing'"):
      bridging.extract_segment_memberships(
          self.participants_df, segment_column="missing"
      )

    with self.assertRaisesRegex(ValueError, "not found"):
      bridging.extract_segment_memberships(
          self.participants_df, segment_columns=["urban", "missing_col"]
      )

    non_binary_df = pd.DataFrame({
        "participant_id": ["p1", "p2"],
        "cluster": [0, 2],
    })
    with self.assertRaisesRegex(ValueError, "Expected binary indicator"):
      bridging.extract_segment_memberships(
          non_binary_df, segment_columns=["cluster"]
      )

  def test_compute_and_rank_validation_errors(self):
    memberships = {"North": {"p1", "p2"}}
    with self.assertRaisesRegex(ValueError, "min_segment_size"):
      bridging.compute_bridging_scores(
          self.approval_matrix, memberships, min_segment_size=0
      )
    with self.assertRaisesRegex(ValueError, "smoothing"):
      bridging.compute_bridging_scores(
          self.approval_matrix, memberships, smoothing=-0.5
      )
    with self.assertRaisesRegex(ValueError, "at least one segment"):
      bridging.compute_bridging_scores(self.approval_matrix, {})
    with self.assertRaisesRegex(ValueError, "Unknown bridging metric"):
      bridging.rank_by_bridging(
          self.approval_matrix, memberships, metric="invalid_metric"
      )

    dup_index_matrix = pd.DataFrame({"C": [1, 0]}, index=["p1", "p1"])
    with self.assertRaisesRegex(ValueError, "duplicate participant IDs"):
      bridging.compute_bridging_scores(dup_index_matrix, memberships)

    trinary_matrix = pd.DataFrame({"C": [1, -1]}, index=["p1", "p2"])
    with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
      bridging.compute_bridging_scores(trinary_matrix, memberships)

    string_matrix = pd.DataFrame(
        {"C": ["Agree", "Disagree"]}, index=["p1", "p2"]
    )
    with self.assertRaisesRegex(ValueError, "boolean or numeric"):
      bridging.compute_bridging_scores(string_matrix, memberships)

    with self.assertRaisesRegex(ValueError, "collides with a core result"):
      bridging.rank_by_bridging(
          self.approval_matrix,
          {"minmax_approval": {"p1", "p2"}},
          include_segment_rates=True,
          segment_column_prefix="",
      )

    with self.assertRaises(argparse.ArgumentTypeError):
      bridging.min_segment_size_arg("0")
    with self.assertRaises(argparse.ArgumentTypeError):
      bridging.min_segment_size_arg("abc")


if __name__ == "__main__":
  unittest.main()
