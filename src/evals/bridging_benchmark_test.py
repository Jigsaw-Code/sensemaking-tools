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
"""Unit tests for the Sensemaking bridging scores benchmark and evaluator."""

import json
import pathlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.evals import bridging_benchmark


def _make_oracle_scored_df(gt_df: pd.DataFrame) -> pd.DataFrame:
  """Constructs an idealized scored DataFrame from ground-truth labels."""
  df = gt_df.copy()
  df["CURIOSITY_EXPERIMENTAL"] = np.where(
      df["expected_curiosity"] == 1, 0.85, 0.10
  )
  df["PERSONAL_STORY_EXPERIMENTAL"] = np.where(
      df["expected_personal_story"] == 1, 0.85, 0.10
  )
  df["REASONING_EXPERIMENTAL"] = np.where(
      df["expected_reasoning"] == 1, 0.85, 0.10
  )
  df["AVERAGE_OF_3_BRIDGING"] = df[
      [
          "CURIOSITY_EXPERIMENTAL",
          "PERSONAL_STORY_EXPERIMENTAL",
          "REASONING_EXPERIMENTAL",
      ]
  ].mean(axis=1)
  df["BRIDGING_SKIP_REASON"] = ""
  return df


class TestLoadBridgingBenchmarkDataset:
  """Tests for loading and validating the ground-truth benchmark dataset."""

  def test_default_wedding_music_bridging_dataset_structure(self):
    """Verifies the default 70-quote wedding music bridging dataset."""
    raw_df = pd.read_csv(bridging_benchmark.DEFAULT_BENCHMARK_CSV)
    assert list(raw_df.columns) == [
        "quote_id",
        "participant_id",
        "topic",
        "opinion",
        "archetype",
        "expected_curiosity",
        "expected_personal_story",
        "expected_reasoning",
        "quote",
    ]

    df = bridging_benchmark.load_bridging_benchmark_dataset()
    assert len(df) == 70
    assert df["topic"].nunique() == 3
    assert df["opinion"].nunique() == 7
    expected_tier_sum = (
        df["expected_curiosity"]
        + df["expected_personal_story"]
        + df["expected_reasoning"]
    )
    assert (df["expected_bridging_tier"] == expected_tier_sum).all()

    # Every opinion cluster should have exactly 10 quotes spanning all 7
    # archetypes in the expected proportions.
    expected_archetype_counts = {
        "bare_assertion": 2,
        "pure_curiosity": 1,
        "pure_personal_story": 1,
        "pure_reasoning": 2,
        "story_and_reasoning": 1,
        "curiosity_and_reasoning": 1,
        "full_bridging": 2,
    }
    for (_, _), group in df.groupby(["topic", "opinion"]):
      assert len(group) == 10
      counts = group["archetype"].value_counts().to_dict()
      assert counts == expected_archetype_counts

    # Verify balanced positive counts across the 70 quotes.
    assert int(df["expected_curiosity"].sum()) == 28
    assert int(df["expected_personal_story"].sum()) == 28
    assert int(df["expected_reasoning"].sum()) == 42

  def test_missing_columns_raises_value_error(self, tmp_path: pathlib.Path):
    """Verifies missing required columns raise ValueError."""
    bad_csv = tmp_path / "missing_cols.csv"
    pd.DataFrame({"quote_id": ["q1"], "quote": ["hello"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(ValueError, match="missing required columns"):
      bridging_benchmark.load_bridging_benchmark_dataset(bad_csv)

  def test_duplicate_quote_id_raises_value_error(self, tmp_path: pathlib.Path):
    """Verifies duplicate quote_id rows raise ValueError."""
    df = bridging_benchmark.load_bridging_benchmark_dataset().iloc[:2].copy()
    df["quote_id"] = ["q001", "q001"]
    bad_csv = tmp_path / "dup_ids.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="duplicate quote_id"):
      bridging_benchmark.load_bridging_benchmark_dataset(bad_csv)

  def test_non_binary_flag_raises_value_error(self, tmp_path: pathlib.Path):
    """Verifies non-binary ground-truth attribute labels raise ValueError."""
    df = bridging_benchmark.load_bridging_benchmark_dataset().iloc[:2].copy()
    df.loc[df.index[0], "expected_curiosity"] = 2
    bad_csv = tmp_path / "bad_binary.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="only binary values"):
      bridging_benchmark.load_bridging_benchmark_dataset(bad_csv)

  def test_inconsistent_tier_raises_value_error(self, tmp_path: pathlib.Path):
    """Verifies tier mismatch with sum of binary flags raises ValueError."""
    df = bridging_benchmark.load_bridging_benchmark_dataset().iloc[:2].copy()
    df.loc[df.index[0], "expected_bridging_tier"] = 3
    bad_csv = tmp_path / "bad_tier.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="does not equal the sum"):
      bridging_benchmark.load_bridging_benchmark_dataset(bad_csv)

  def test_empty_quote_raises_value_error(self, tmp_path: pathlib.Path):
    """Verifies blank or whitespace-only quotes raise ValueError."""
    df = bridging_benchmark.load_bridging_benchmark_dataset().iloc[:2].copy()
    df.loc[df.index[0], "quote"] = "   "
    bad_csv = tmp_path / "empty_quote.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="empty quote texts"):
      bridging_benchmark.load_bridging_benchmark_dataset(bad_csv)


class TestEvaluateBridgingOutput:
  """Tests for discrimination, orthogonality, ranking, and summary metrics."""

  def test_oracle_predictions_achieve_perfect_metrics(self):
    """Verifies an ideal scorer achieves 1.0 on discrimination and ranking."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    scored_df = _make_oracle_scored_df(gt_df)

    metrics = bridging_benchmark.evaluate_bridging_output(scored_df)

    summary = metrics["summary"]
    assert summary["total_quotes"] == 70
    assert summary["scored_quotes"] == 70
    assert summary["scored_rate"] == pytest.approx(1.0)
    assert summary["scoring_failed_count"] == 0
    assert summary["scoring_failed_rate"] == pytest.approx(0.0)

    macro = metrics["attribute_metrics"]["macro_average"]
    assert macro["roc_auc"] == pytest.approx(1.0)
    assert macro["average_precision"] == pytest.approx(1.0)
    assert macro["mean_positive"] == pytest.approx(0.85)
    assert macro["mean_negative"] == pytest.approx(0.10)
    assert macro["mean_separation"] == pytest.approx(0.75)
    assert macro["low_band_compliance_rate"] == pytest.approx(1.0)
    assert macro["high_band_compliance_rate"] == pytest.approx(1.0)
    assert macro["off_target_leakage_mean"] == pytest.approx(0.10)
    assert macro["off_target_high_rate"] == pytest.approx(0.0)

    ortho = metrics["orthogonality_metrics"]
    assert ortho["single_attribute_top1_accuracy"] == pytest.approx(1.0)
    assert ortho["tier_means_strictly_increasing"]

    ranking = metrics["ranking_metrics"]
    assert ranking["opinion_count"] == 7
    assert ranking["mean_spearman_rho"] == pytest.approx(1.0)
    assert ranking["mean_kendall_tau"] == pytest.approx(1.0)
    assert ranking["mean_pairwise_tier_concordance"] == pytest.approx(1.0)
    assert ranking["mean_ndcg_at_3"] == pytest.approx(1.0)
    assert ranking["top1_is_max_tier_rate"] == pytest.approx(1.0)
    assert ranking["bottom1_is_min_tier_rate"] == pytest.approx(1.0)

  def test_halo_effect_leakage_is_detected(self):
    """Verifies cross-attribute leakage metrics catch halo-effect inflation."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    df = gt_df.copy()

    # Simulate a model that gives 0.90 to target attributes, 0.10 to bare
    # assertions, but leaks 0.75 onto off-target attributes whenever ANY
    # bridging attribute is present.
    has_any = df["expected_bridging_tier"] >= 1
    df["CURIOSITY_EXPERIMENTAL"] = np.where(
        df["expected_curiosity"] == 1,
        0.90,
        np.where(has_any, 0.75, 0.10),
    )
    df["PERSONAL_STORY_EXPERIMENTAL"] = np.where(
        df["expected_personal_story"] == 1,
        0.90,
        np.where(has_any, 0.75, 0.10),
    )
    df["REASONING_EXPERIMENTAL"] = np.where(
        df["expected_reasoning"] == 1,
        0.90,
        np.where(has_any, 0.75, 0.10),
    )
    df["AVERAGE_OF_3_BRIDGING"] = df[
        [
            "CURIOSITY_EXPERIMENTAL",
            "PERSONAL_STORY_EXPERIMENTAL",
            "REASONING_EXPERIMENTAL",
        ]
    ].mean(axis=1)
    df["BRIDGING_SKIP_REASON"] = ""

    metrics = bridging_benchmark.evaluate_bridging_output(df)
    macro = metrics["attribute_metrics"]["macro_average"]

    # Off-target leakage mean should be 0.75 and 100% of off-target rows exceed
    # the 0.60 high-score threshold.
    assert macro["off_target_leakage_mean"] == pytest.approx(0.75)
    assert macro["off_target_high_rate"] == pytest.approx(1.0)
    assert macro["low_band_compliance_rate"] < 0.50

  def test_constant_scores_are_order_invariant(self):
    """Verifies constant predictions yield tie-aware, order-invariant scores."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    df_forward = gt_df.copy()
    for col in [
        "CURIOSITY_EXPERIMENTAL",
        "PERSONAL_STORY_EXPERIMENTAL",
        "REASONING_EXPERIMENTAL",
        "AVERAGE_OF_3_BRIDGING",
    ]:
      df_forward[col] = 0.50
    df_forward["BRIDGING_SKIP_REASON"] = ""

    # Reverse the row order to ensure ties do not depend on input row ordering.
    df_reverse = df_forward.iloc[::-1].reset_index(drop=True)

    m_fwd = bridging_benchmark.evaluate_bridging_output(df_forward)
    m_rev = bridging_benchmark.evaluate_bridging_output(df_reverse)

    assert m_fwd["attribute_metrics"]["macro_average"][
        "roc_auc"
    ] == pytest.approx(0.5)
    assert m_fwd["attribute_metrics"]["macro_average"][
        "average_precision"
    ] == pytest.approx(
        m_rev["attribute_metrics"]["macro_average"]["average_precision"]
    )
    assert m_fwd["orthogonality_metrics"][
        "single_attribute_top1_accuracy"
    ] == pytest.approx(0.0)

    rank_fwd = m_fwd["ranking_metrics"]
    rank_rev = m_rev["ranking_metrics"]
    assert rank_fwd["mean_spearman_rho"] == pytest.approx(0.0)
    assert rank_fwd["mean_kendall_tau"] == pytest.approx(0.0)
    assert rank_fwd["mean_pairwise_tier_concordance"] == pytest.approx(0.0)
    assert rank_fwd[
        "mean_pairwise_concordance_with_half_ties"
    ] == pytest.approx(0.5)
    assert rank_fwd["mean_pairwise_tie_rate"] == pytest.approx(1.0)
    # 2 out of 10 quotes per opinion are tier 3, and 2 out of 10 are tier 0.
    assert rank_fwd["top1_is_max_tier_rate"] == pytest.approx(0.2)
    assert rank_fwd["bottom1_is_min_tier_rate"] == pytest.approx(0.2)
    assert rank_fwd["mean_ndcg_at_3"] == pytest.approx(
        rank_rev["mean_ndcg_at_3"]
    )

  def test_partial_scoring_failures_are_tracked(self):
    """Verifies failed/NaN rows are counted in summary operational metrics."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    scored_df = _make_oracle_scored_df(gt_df)

    # Mark the first 7 rows as failed scoring.
    scored_df.loc[:6, "CURIOSITY_EXPERIMENTAL"] = np.nan
    scored_df.loc[:6, "PERSONAL_STORY_EXPERIMENTAL"] = np.nan
    scored_df.loc[:6, "REASONING_EXPERIMENTAL"] = np.nan
    scored_df.loc[:6, "AVERAGE_OF_3_BRIDGING"] = np.nan
    scored_df.loc[:6, "BRIDGING_SKIP_REASON"] = "scoring_failed"

    metrics = bridging_benchmark.evaluate_bridging_output(scored_df)
    summary = metrics["summary"]
    assert summary["total_quotes"] == 70
    assert summary["scored_quotes"] == 63
    assert summary["scored_rate"] == pytest.approx(0.9)
    assert summary["scoring_failed_count"] == 7
    assert summary["scoring_failed_rate"] == pytest.approx(0.1)
    assert summary["skip_reason_counts"]["scoring_failed"] == 7

  def test_evaluate_computes_tier_when_absent_from_dataframe(self):
    """Verifies evaluate_bridging_output derives expected_bridging_tier."""
    raw_df = pd.read_csv(bridging_benchmark.DEFAULT_BENCHMARK_CSV)
    assert "expected_bridging_tier" not in raw_df.columns
    scored_df = _make_oracle_scored_df(raw_df)

    metrics = bridging_benchmark.evaluate_bridging_output(scored_df)
    assert metrics["ranking_metrics"]["mean_spearman_rho"] == pytest.approx(1.0)
    assert metrics["ranking_metrics"]["mean_kendall_tau"] == pytest.approx(1.0)
    assert metrics["ranking_metrics"]["mean_ndcg_at_3"] == pytest.approx(1.0)


class TestRunnerAndCLI:
  """Tests for environment configuration, artifact export, and CLI execution."""

  def test_configure_model_endpoint_env_sets_gemma_ollama_endpoint(self):
    """Verifies gemma* models set openai_api_compatible and Ollama /v1/ URL."""
    env = bridging_benchmark.configure_model_endpoint_env(
        model_name="gemma4:26b",
        env={"OLLAMA_HOST": "10.128.0.5:11434"},
    )
    assert env["MODEL_ENDPOINT_TYPE"] == "openai_api_compatible"
    assert env["OPENAI_API_ENDPOINT_URL"] == "http://10.128.0.5:11434/v1/"

  def test_cli_validate_only_succeeds(self, capsys):
    """Verifies --validate_only validates the default dataset and exits 0."""
    rc = bridging_benchmark.main(["--validate_only"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Validated bridging benchmark dataset: 70 quotes" in captured.out

  def test_cli_evaluate_and_save_artifacts(self, tmp_path: pathlib.Path):
    """Verifies evaluating a scored CSV writes all report artifacts."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    scored_df = _make_oracle_scored_df(gt_df)
    scored_csv = tmp_path / "scored.csv"
    scored_df.to_csv(scored_csv, index=False)

    out_dir = tmp_path / "reports"
    rc = bridging_benchmark.main(
        ["--scored_csv", str(scored_csv), "--output_dir", str(out_dir)]
    )
    assert rc == 0

    report_json = out_dir / "bridging_evaluation_report.json"
    summary_txt = out_dir / "bridging_evaluation_summary.txt"
    attr_csv = out_dir / "bridging_attribute_metrics.csv"
    arch_csv = out_dir / "bridging_archetype_breakdown.csv"
    opinion_csv = out_dir / "bridging_opinion_ranking.csv"

    assert report_json.exists()
    assert summary_txt.exists()
    assert attr_csv.exists()
    assert arch_csv.exists()
    assert opinion_csv.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["ranking_metrics"]["mean_spearman_rho"] == pytest.approx(1.0)

  @mock.patch("src.get_bridging_scores.get_bridging_scores")
  def test_cli_run_bridging_invokes_get_bridging_scores_library(
      self, mock_get_bridging_scores: mock.MagicMock, tmp_path: pathlib.Path
  ):
    """Verifies --run_bridging calls get_bridging_scores in-process."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset()
    scored_df = _make_oracle_scored_df(gt_df)
    mock_get_bridging_scores.return_value = scored_df
    out_dir = tmp_path / "run_out"

    with mock.patch.dict("os.environ", {}, clear=False):
      rc = bridging_benchmark.main([
          "--run_bridging",
          "--model_name",
          "gemma4:e4b",
          "--output_dir",
          str(out_dir),
      ])
    assert rc == 0
    mock_get_bridging_scores.assert_called_once()
    call_kwargs = mock_get_bridging_scores.call_args.kwargs
    assert call_kwargs["model_name"] == "gemma4:e4b"
    assert call_kwargs["scorer_type"] == "GEMINI"
    assert call_kwargs["force_rerun"]
    assert (out_dir / "scored_bridging_quotes.csv").exists()
    assert (out_dir / "bridging_evaluation_report.json").exists()

  @mock.patch("src.get_bridging_scores.ContentScorer")
  def test_run_bridging_pipeline_in_memory_without_csv(
      self, mock_scorer_cls: mock.MagicMock
  ):
    """Verifies run_bridging_pipeline scores a DataFrame directly in memory."""
    gt_df = bridging_benchmark.load_bridging_benchmark_dataset().iloc[:3].copy()
    mock_scorer = mock_scorer_cls.return_value
    mock_scorer.score.side_effect = lambda items, attrs: [
        {"row_id": item["row_id"], "scores": {attr: 0.75 for attr in attrs}}
        for item in items
    ]

    result_df = bridging_benchmark.run_bridging_pipeline(
        dataset=gt_df,
        model_name="gemini-2.5-flash",
        gemini_api_key="test_key",
    )
    assert len(result_df) == 3
    assert (result_df["AVERAGE_OF_3_BRIDGING"] == 0.75).all()
    assert result_df["BRIDGING_SKIP_REASON"].eq("").all()
