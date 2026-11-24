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

"""Unit tests for the modular categorization benchmark dataset and evaluator."""

import json
import os
import pandas as pd
import pytest

from src import categorization_runner
from src.evals import categorization_benchmark
from src.models import custom_types


def test_wedding_music_benchmark_dataset_invariants():
  """Verifies the 100-comment wedding music dataset structure and balance."""
  df = categorization_benchmark.load_benchmark_dataset()

  assert list(df.columns) == [
      "participant_id",
      "expected_topic",
      "expected_opinion",
      "expected_stance",
      "expected_keywords",
      "survey_text",
  ]
  assert len(df) == 100
  assert df["participant_id"].nunique() == 100
  assert df["survey_text"].nunique() == 100

  stance_counts = df["expected_stance"].value_counts().to_dict()
  assert stance_counts["house_music"] == 20
  assert stance_counts["80s_pop"] == 20
  assert stance_counts["disco"] == 20
  assert stance_counts["scheduled_mix"] == 12
  assert stance_counts["dinner_acoustic"] == 10
  assert stance_counts["dj_beatmatching"] == 10
  assert stance_counts["live_band_energy"] == 8

  assert df["expected_topic"].nunique() == 3
  assert df["expected_opinion"].nunique() == 7

  context_path = categorization_benchmark.DEFAULT_WEDDING_CONTEXT_FILE
  assert context_path.endswith("wedding_music_context.md")
  assert os.path.exists(context_path)
  with open(context_path, "r", encoding="utf-8") as f:
    context_text = f.read()
  assert "wedding reception" in context_text.lower()
  assert "GEMINI_API_KEY" not in context_text
  assert "python -m" not in context_text


def test_compatibility_with_categorization_runner_io():
  """Verifies `categorization_runner` reads and preserves benchmark columns."""
  csv_rows = categorization_runner._read_csv_to_dicts(
      categorization_benchmark.DEFAULT_WEDDING_BENCHMARK_CSV
  )
  statements = categorization_runner._convert_csv_rows_to_statements(csv_rows)

  assert len(statements) == 100
  # Ground-truth columns must not be mistaken for pre-assigned topics.
  assert all(not s.topics for s in statements)
  assert all(not s.quotes for s in statements)

  # Simulate categorization_runner assigning each statement its ground-truth
  # topic and opinion as a single full-statement quote.
  rows_by_id = {r["participant_id"]: r for r in csv_rows}
  for stmt in statements:
    gt_row = rows_by_id[stmt.id]
    nested_topic = custom_types.NestedTopic(
        name=gt_row["expected_topic"],
        subtopics=[custom_types.FlatTopic(name=gt_row["expected_opinion"])],
    )
    stmt.quotes = [
        custom_types.Quote(
            id=f"{stmt.id}-q1",
            text=f"[{stmt.text}]",
            topic=nested_topic,
        )
    ]

  output_rows = categorization_runner._set_topics_on_csv_rows(
      csv_rows, statements
  )
  assert len(output_rows) == 100
  output_df = pd.DataFrame(output_rows)

  # Verify ground-truth columns are preserved on categorized_with_other rows.
  for col in (
      "participant_id",
      "expected_topic",
      "expected_opinion",
      "expected_stance",
      "expected_keywords",
      "topic",
      "opinion",
  ):
    assert col in output_df.columns

  metrics = categorization_benchmark.evaluate_categorization_output(output_df)
  assert metrics["summary"]["statement_coverage_rate"] == pytest.approx(1.0)
  assert metrics["summary"][
      "non_other_opinion_statement_rate"
  ] == pytest.approx(1.0)
  assert metrics["topic_metrics"]["cluster_purity"] == pytest.approx(1.0)
  assert metrics["topic_metrics"]["inverse_purity_recall"] == pytest.approx(1.0)
  assert metrics["topic_metrics"]["pairwise_f1"] == pytest.approx(1.0)
  assert metrics["topic_metrics"]["v_measure"] == pytest.approx(1.0)
  assert metrics["topic_metrics"]["exact_match_rate"] == pytest.approx(1.0)

  assert metrics["opinion_metrics"]["cluster_purity"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["inverse_purity_recall"] == pytest.approx(
      1.0
  )
  assert metrics["opinion_metrics"]["pairwise_f1"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["v_measure"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["exact_match_rate"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["group_keyword_coverage"] == pytest.approx(
      1.0
  )


def test_export_guided_topics_and_taxonomy_csv(tmp_path):
  """Verifies guided topics and `--topic_and_opinion_csv` export cleanly."""
  df = categorization_benchmark.load_benchmark_dataset()
  topics_arg = categorization_benchmark.export_guided_topics_arg(df)
  parsed_flat_topics = categorization_runner._get_topics_from_arg(topics_arg)
  assert len(parsed_flat_topics) == 3

  taxonomy_csv = os.path.join(str(tmp_path), "taxonomy.csv")
  exported_df = categorization_benchmark.export_topic_and_opinion_csv(
      df, taxonomy_csv
  )
  assert len(exported_df) == 7
  assert list(exported_df.columns) == ["topic", "opinion"]

  nested_topics = categorization_runner._get_topics_and_opinions_from_csv(
      taxonomy_csv
  )
  assert len(nested_topics) == 3
  total_subtopics = sum(len(t.subtopics) for t in nested_topics)
  assert total_subtopics == 7


def test_unconstrained_discovery_paraphrased_labels_score_perfect_clustering():
  """Verifies discovered labels with different wording get 1.0 cluster scores."""
  df = categorization_benchmark.load_benchmark_dataset()

  paraphrased_topics = {
      "Dance Floor Music Genres": "Choosing Dance Music Styles",
      "Reception Timeline and Atmosphere": "Pacing the Reception Schedule",
      "DJ Mixing and Live Performance": "Live Musicians vs. DJ Mixing",
  }
  paraphrased_opinions = {
      "house_music": "Electronic house beats keep the dance floor moving.",
      "80s_pop": "Eighties pop anthems get everyone singing the lyrics.",
      "disco": "Seventies disco funk brings all generations together.",
      "scheduled_mix": (
          "Play family hits early and transition to late club sets."
      ),
      "dinner_acoustic": "Keep dinner mellow with jazz and acoustic songs.",
      "dj_beatmatching": "Hire a DJ who beatmatches seamless transitions.",
      "live_band_energy": "Book a live band with horns for stage excitement.",
  }

  categorized_df = df.copy()
  categorized_df["topic"] = categorized_df["expected_topic"].map(
      paraphrased_topics
  )
  categorized_df["opinion"] = categorized_df["expected_stance"].map(
      paraphrased_opinions
  )

  metrics = categorization_benchmark.evaluate_categorization_output(
      categorized_df
  )
  op = metrics["opinion_metrics"]
  assert op["exact_match_rate"] == pytest.approx(0.0)
  assert op["cluster_purity"] == pytest.approx(1.0)
  assert op["inverse_purity_recall"] == pytest.approx(1.0)
  assert op["purity_f1"] == pytest.approx(1.0)
  assert op["pairwise_f1"] == pytest.approx(1.0)
  assert op["v_measure"] == pytest.approx(1.0)
  assert op["statement_keyword_hit_rate"] == pytest.approx(1.0)
  assert op["group_keyword_coverage"] == pytest.approx(1.0)
  # Keywords are opinion-level domain checks, not applied at topic level.
  assert metrics["topic_metrics"]["group_keyword_coverage"] == pytest.approx(
      0.0
  )

  # Opaque placeholder labels keep structural purity=1.0 but fail keyword check.
  placeholder_df = df.copy()
  placeholder_df["topic"] = placeholder_df["expected_topic"].map({
      "Dance Floor Music Genres": "Topic A",
      "Reception Timeline and Atmosphere": "Topic B",
      "DJ Mixing and Live Performance": "Topic C",
  })
  placeholder_df["opinion"] = placeholder_df["expected_stance"].map({
      "house_music": "Opinion 1",
      "80s_pop": "Opinion 2",
      "disco": "Opinion 3",
      "scheduled_mix": "Opinion 4",
      "dinner_acoustic": "Opinion 5",
      "dj_beatmatching": "Opinion 6",
      "live_band_energy": "Opinion 7",
  })
  placeholder_metrics = categorization_benchmark.evaluate_categorization_output(
      placeholder_df
  )
  assert placeholder_metrics["opinion_metrics"][
      "cluster_purity"
  ] == pytest.approx(1.0)
  assert placeholder_metrics["opinion_metrics"][
      "group_keyword_coverage"
  ] == pytest.approx(0.0)
  assert placeholder_metrics["opinion_metrics"][
      "statement_keyword_hit_rate"
  ] == pytest.approx(0.0)


def test_merged_clusters_and_other_penalties(tmp_path):
  """Verifies merged stances and 'Other' rows reduce purity and recall."""
  df = categorization_benchmark.load_benchmark_dataset()
  categorized_df = df.copy()
  categorized_df["topic"] = categorized_df["expected_topic"]
  categorized_df["opinion"] = categorized_df["expected_opinion"]

  # Merge all 20 'disco' comments into the 'house_music' opinion label.
  house_opinion = df.loc[
      df["expected_stance"] == "house_music", "expected_opinion"
  ].iloc[0]
  categorized_df.loc[
      categorized_df["expected_stance"] == "disco", "opinion"
  ] = house_opinion

  # Assign all 10 'dinner_acoustic' comments to 'Other'.
  categorized_df.loc[
      categorized_df["expected_stance"] == "dinner_acoustic", "opinion"
  ] = "Other"

  metrics = categorization_benchmark.evaluate_categorization_output(
      categorized_df
  )
  summary = metrics["summary"]
  op = metrics["opinion_metrics"]

  assert summary["non_other_opinion_statement_rate"] == pytest.approx(0.90)
  assert summary["opinion_other_row_rate"] == pytest.approx(0.10)
  # 20 of 90 non-Other rows are impure (disco merged into house_music: 70/90).
  assert op["cluster_purity"] == pytest.approx(70.0 / 90.0)
  # 90 out of 100 total statements are in their group's dominant cluster.
  assert op["inverse_purity_recall"] == pytest.approx(0.90)
  assert op["pairwise_precision"] < 1.0

  report_text = categorization_benchmark.format_evaluation_report(metrics)
  assert "CATEGORIZATION BENCHMARK EVALUATION REPORT" in report_text

  json_path, csv_path = categorization_benchmark.save_evaluation_artifacts(
      metrics, str(tmp_path)
  )
  assert os.path.exists(json_path)
  assert os.path.exists(csv_path)
  with open(json_path, "r") as f:
    loaded_json = json.load(f)
  assert loaded_json["summary"]["total_ground_truth_statements"] == 100
  mapping_df = pd.read_csv(csv_path)
  assert len(mapping_df) == 7


def test_modular_custom_benchmark_config(tmp_path):
  """Verifies custom datasets with arbitrary column names can be swapped in."""
  custom_gt = pd.DataFrame([
      {
          "comment_id": "u1",
          "body": "We need more bike lanes downtown.",
          "gold_theme": "Transit",
          "gold_claim": "Expand protected bike lanes.",
          "gold_keywords": "bike;cycling;lanes",
      },
      {
          "comment_id": "u2",
          "body": "Protected cycling lanes make commuting safe.",
          "gold_theme": "Transit",
          "gold_claim": "Expand protected bike lanes.",
          "gold_keywords": "bike;cycling;lanes",
      },
      {
          "comment_id": "u3",
          "body": "Plant more shade trees in neighborhood parks.",
          "gold_theme": "Parks",
          "gold_claim": "Increase tree canopy in parks.",
          "gold_keywords": "trees;shade;parks",
      },
  ])
  gt_path = os.path.join(str(tmp_path), "custom_gt.csv")
  custom_gt.to_csv(gt_path, index=False)

  config = categorization_benchmark.BenchmarkConfig(
      id_col="comment_id",
      text_col="body",
      expected_topic_col="gold_theme",
      expected_opinion_col="gold_claim",
      expected_stance_col=None,
      expected_keywords_col="gold_keywords",
      predicted_topic_col="pred_topic",
      predicted_opinion_col="pred_opinion",
  )

  loaded_gt = categorization_benchmark.load_benchmark_dataset(
      gt_path, config=config
  )
  assert len(loaded_gt) == 3

  pred_df = pd.DataFrame([
      {
          "comment_id": "u1",
          "pred_topic": "Urban Mobility",
          "pred_opinion": "Build protected cycling lanes.",
      },
      {
          "comment_id": "u2",
          "pred_topic": "Urban Mobility",
          "pred_opinion": "Build protected cycling lanes.",
      },
      {
          "comment_id": "u3",
          "pred_topic": "Green Spaces",
          "pred_opinion": "Plant shade trees in city parks.",
      },
  ])

  metrics = categorization_benchmark.evaluate_categorization_output(
      categorized_df=pred_df,
      ground_truth_df=loaded_gt,
      config=config,
  )
  assert metrics["opinion_metrics"]["cluster_purity"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["pairwise_f1"] == pytest.approx(1.0)
  assert metrics["opinion_metrics"]["group_keyword_coverage"] == pytest.approx(
      1.0
  )


def test_validation_rejects_duplicate_ids_and_reserved_columns(tmp_path):
  """Verifies `load_benchmark_dataset` catches schema errors."""
  bad_dup_df = pd.DataFrame([
      {
          "participant_id": "c001",
          "survey_text": "First comment",
          "expected_topic": "T1",
          "expected_opinion": "O1",
      },
      {
          "participant_id": "c001",
          "survey_text": "Second comment with duplicate ID",
          "expected_topic": "T1",
          "expected_opinion": "O1",
      },
  ])
  dup_path = os.path.join(str(tmp_path), "dup.csv")
  bad_dup_df.to_csv(dup_path, index=False)
  with pytest.raises(ValueError, match="duplicate"):
    categorization_benchmark.load_benchmark_dataset(dup_path)

  bad_reserved_df = pd.DataFrame([
      {
          "participant_id": "c001",
          "survey_text": "First comment",
          "topics": "Preassigned Topic",
          "expected_topic": "T1",
          "expected_opinion": "O1",
      },
  ])
  reserved_path = os.path.join(str(tmp_path), "reserved.csv")
  bad_reserved_df.to_csv(reserved_path, index=False)
  with pytest.raises(ValueError, match="reserved"):
    categorization_benchmark.load_benchmark_dataset(reserved_path)


def test_build_runner_env_configures_gemma_and_gce_tunnel(tmp_path):
  """Verifies `build_runner_env` sets OpenAICompatibleModel env vars for Gemma."""
  host_file = os.path.join(str(tmp_path), "gce_ollama_active_host")
  with open(host_file, "w") as f:
    f.write("127.0.0.1:11435\n")

  env_26b = categorization_benchmark.build_runner_env(
      model_name="gemma4:26b",
      base_env={},
      active_host_file=host_file,
  )
  assert env_26b["MODEL_ENDPOINT_TYPE"] == "openai_api_compatible"
  assert env_26b["OLLAMA_HOST"] == "127.0.0.1:11435"
  assert env_26b["OPENAI_API_ENDPOINT_URL"] == "http://127.0.0.1:11435/v1"
  assert env_26b["OPENAI_MAX_TOKENS"] == "8192"
  assert env_26b["OPENAI_REASONING_EFFORT"] == "none"
  assert env_26b["MAX_BATCH_TOKENS"] == "3000"
  assert env_26b["MAX_ITEMS_PER_BATCH"] == "5"

  env_270m = categorization_benchmark.build_runner_env(
      model_name="gemma3:270m",
      base_env={},
      active_host_file=os.path.join(str(tmp_path), "nonexistent"),
  )
  assert env_270m["MODEL_ENDPOINT_TYPE"] == "openai_api_compatible"
  assert env_270m["OLLAMA_HOST"] == "127.0.0.1:11434"
  assert env_270m["OPENAI_MAX_TOKENS"] == "1024"
  assert env_270m["MAX_ITEMS_PER_BATCH"] == "2"

  env_gemini = categorization_benchmark.build_runner_env(
      model_name="gemini-3.5-flash",
      base_env={},
      active_host_file=host_file,
  )
  assert "MODEL_ENDPOINT_TYPE" not in env_gemini
