# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines metrics used for evaluating tasks.

from typing import Dict, List, Optional

import pandas as pd
from src import prompts
from src.quote_extraction import quote_extractor


# The criteria to define a successful quote extraction.
def _format_list_as_string(items: List[str]) -> str:
  """Formats a list of strings into a multi-line string with bullet points."""
  return "\n".join([f"- {item}" for item in items])


class GroupByConfig:
  """Configuration for grouping data before evaluation."""

  def __init__(self, group_by_col: str, agg_col: str):
    self.group_by_col = group_by_col
    self.agg_col = agg_col


def _get_aggregated_col_name(name: str) -> str:
  return f"{name}_aggregated"


def _group_by_df(
    df: pd.DataFrame, group_by_col: str, agg_col: str
) -> pd.DataFrame:
  """Helper function to group a dataframe."""
  if group_by_col not in df.columns:
    raise ValueError(f"Column '{group_by_col}' not found for grouping.")
  if agg_col not in df.columns:
    raise ValueError(f"Column '{agg_col}' not found for aggregation.")

  agg_col_new = _get_aggregated_col_name(agg_col)
  grouped_agg = (
      df.groupby(group_by_col)[agg_col]
      .apply(lambda x: "\n".join([f"{i+1}. {v}" for i, v in enumerate(x)]))
      .reset_index(name=agg_col_new)
  )

  # Get other columns from the first entry of each group
  other_cols = df.drop(columns=[agg_col]).drop_duplicates(subset=[group_by_col])

  # Merge them back
  result_df = pd.merge(other_cols, grouped_agg, on=group_by_col, how="left")

  return result_df


class EvaluationMetrics:
  """Base class that contains Pointwise and Pairwise evaluation metrics."""

  def __init__(
      self,
      name: str,
      criteria: Dict[str, str],
      additional_input_variables: List[str],
      response_name: str,
      group_by_config: Optional[GroupByConfig] = None,
      pointwise_metric: Optional["PointwiseEvaluationMetric"] = None,
  ):
    """ "Creates an EvaluationMetric object.

    Args:
        name: the name of the evaluation
        criteria: the criteria for evaluating how well the task was done
        additional_input_variables: other context that should be included during
          evals. These should correspond to columns in the input data. An
          example use case for this would be to include additional data like the
          topic so one part of the criteria is that the task is topic-aware.
        response_name: the name of the field that contains the task result, for
          example "extracted text" for evaluating quote extraction
        group_by_config: an optional dictionary that defines how to group the
          data before running evals.
        pointwise_metric: An optional, pre-configured PointwiseEvaluationMetric
          instance. If not provided, a default PointwiseEvaluationMetric will be
          created.
    """
    self.name = name
    self.group_by_config = group_by_config
    self.response_name = response_name
    self.additional_input_variables = additional_input_variables
    # Use the newly aggregated column name if applicable
    self._update_variable_names()

    if pointwise_metric:
      self.pointwise_metric = pointwise_metric
    else:
      self.pointwise_metric = PointwiseEvaluationMetric(
          name,
          criteria,
          self.additional_input_variables,
          self.response_name,
          group_by_config,
      )
    self.pairwise_metric = PairwiseEvaluationMetric(
        name,
        criteria,
        self.additional_input_variables,
        self.response_name,
        group_by_config,
    )

  def _update_variable_names(self):
    if not self.group_by_config:
      return

    agg_col = self.group_by_config.agg_col
    if agg_col == self.response_name:
      self.response_name = _get_aggregated_col_name(self.response_name)


class PointwiseEvaluationMetric:
  """Class for pointwise evaluation metrics."""

  rating_rubric = {
      "4": "The response performs well on all criteria.",
      "3": "The response performs well on most criteria.",
      "2": "The response performs well on some criteria.",
      "1": "The response is somewhat aligned with the criteria",
      "0": "The response falls short on all criteria",
  }

  def __init__(
      self,
      name: str,
      criteria: Dict[str, str],
      input_variables: List[str],
      response_name: str,
      group_by_config: Optional[GroupByConfig] = None,
  ):
    self.criteria = criteria
    self.input_variables = input_variables
    self.name = name + "Pointwise"
    self.response_name = response_name
    self.group_by_config = group_by_config

  def get_evaluation_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
    """Creates an evaluation dataset based on the input data and input variables."""
    processed_data = input_data
    if self.name == "QuoteExtractionPointwise":
      processed_data.drop_duplicates(
          subset=[self.response_name], inplace=True
      )  # response_name contains the quotes
    if self.name == "OpinionQualityPointwise":
      processed_data = processed_data[
          processed_data[self.response_name] != "Other"
      ]
    if self.group_by_config:
      processed_data = _group_by_df(
          processed_data,
          self.group_by_config.group_by_col,
          self.group_by_config.agg_col,
      )

    evaluation_data = pd.DataFrame()
    for input_variable in self.input_variables:
      if input_variable not in processed_data.columns:
        raise RuntimeError(
            f"Input data is expected to have the column '{input_variable}'. Got"
            f" the following columns: {processed_data.columns}"
        )
      else:
        evaluation_data[input_variable] = processed_data[input_variable]

    if self.response_name not in processed_data.columns:
      raise RuntimeError(
          f"Response column '{self.response_name}' not in processed data."
      )
    evaluation_data["response"] = processed_data[self.response_name]
    return evaluation_data


class PairwiseEvaluationMetric:
  """Class for pairwise evaluation metrics."""

  def __init__(
      self,
      name: str,
      criteria: Dict[str, str],
      input_variables: List[str],
      response_name: str,
      group_by_config: Optional[GroupByConfig] = None,
  ):
    self.criteria = criteria
    self.input_variables = input_variables
    self.name = name + "Pairwise"
    self.response_name = response_name
    self.group_by_config = group_by_config
    self.rating_rubric = {
        "A": (
            "Response A demonstrates significantly better quality than"
            " Response B across multiple criteria, including"
            f" {self.criteria.keys()}."
        ),
        "SAME": (
            "Response A and Response B demonstrate comparable quality, with no"
            " significant differences across the evaluated criteria."
        ),
        "B": (
            "Response B demonstrates significantly better quality than"
            " Response A across multiple criteria, including"
            f" {self.criteria.keys()}."
        ),
    }

  def get_evaluation_data(
      self, baseline_data: pd.DataFrame, candidate_data: pd.DataFrame
  ) -> pd.DataFrame:
    """Creates an evaluation dataset for pairwise evals comparing baseline and candidate data outputs."""
    processed_baseline = baseline_data
    processed_candidate = candidate_data
    if self.name == "QuoteExtractionPairwise":
      processed_baseline.drop_duplicates(
          subset=[self.response_name], inplace=True
      )  # response_name contains the quotes
      processed_candidate.drop_duplicates(
          subset=[self.response_name], inplace=True
      )
    if self.name == "OpinionQualityPairwise":
      processed_baseline = processed_baseline[
          processed_baseline[self.response_name] != "Other"
      ]
      processed_candidate = processed_candidate[
          processed_candidate[self.response_name] != "Other"
      ]
    if self.group_by_config:
      processed_baseline = _group_by_df(
          processed_baseline,
          self.group_by_config.group_by_col,
          self.group_by_config.agg_col,
      )
      processed_candidate = _group_by_df(
          processed_candidate,
          self.group_by_config.group_by_col,
          self.group_by_config.agg_col,
      )
    evaluation_data = pd.DataFrame()
    for input_variable in self.input_variables:
      if input_variable not in processed_baseline.columns:
        raise RuntimeError(
            f"Input data is expected to have the column '{input_variable}'"
        )
      else:
        evaluation_data[input_variable] = processed_baseline[input_variable]

    if self.response_name not in processed_baseline.columns:
      raise RuntimeError(
          "Response column"
          f" '{self.response_name}' not in processed baseline data."
      )
    evaluation_data["baseline_model_response"] = processed_baseline[
        self.response_name
    ]

    if self.response_name not in processed_candidate.columns:
      raise RuntimeError(
          "Response column"
          f" '{self.response_name}' not in processed candidate data."
      )
    evaluation_data["response"] = processed_candidate[self.response_name]
    return evaluation_data


class OtherOpinionPointwiseEvaluationMetric(PointwiseEvaluationMetric):
  """Custom pointwise metric to evaluate 'Other' opinions."""

  def get_evaluation_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
    """Creates a dataset to check if "Other" quotes fit in other opinions."""
    evaluation_rows = []
    if quote_extractor.QUOTE_COL not in input_data.columns:
      raise ValueError(
          f"Column '{quote_extractor.QUOTE_COL}' not found in"
          " input data."
      )
    if quote_extractor.TOPIC_COL not in input_data.columns:
      raise ValueError(
          f"Column '{quote_extractor.TOPIC_COL}' not found in input data."
      )
    if "opinion" not in input_data.columns:
      raise ValueError("Column 'opinion' not found in input data.")

    topics = input_data[quote_extractor.TOPIC_COL].unique()

    for topic in topics:
      topic_data = input_data[input_data[quote_extractor.TOPIC_COL] == topic]
      is_other_opinion = topic_data["opinion"] == "Other"
      other_quotes = (
          topic_data[is_other_opinion][quote_extractor.QUOTE_COL]
          .unique()
          .tolist()
      )

      if not other_quotes:
        continue

      opinions_to_check = (
          topic_data[~is_other_opinion]["opinion"].unique().tolist()
      )

      if not opinions_to_check:
        continue

      opinions_str = _format_list_as_string(opinions_to_check)

      for quote in other_quotes:
        evaluation_rows.append([topic, opinions_str, quote])

    return pd.DataFrame(
        evaluation_rows,
        columns=[quote_extractor.TOPIC_COL, "existing_opinions", "response"],
    )


class OtherOpinionEvaluationMetrics(EvaluationMetrics):
  """Metrics for evaluating if 'Other' opinions are correctly categorized."""

  def __init__(self):
    name = "OtherOpinion"
    criteria = prompts.eval_other_opinion_criteria
    additional_input_variables = [
        quote_extractor.TOPIC_COL,
        "existing_opinions",
    ]
    # The response to evaluate is the quote text itself.
    response_name = quote_extractor.QUOTE_COL

    custom_pointwise_metric = OtherOpinionPointwiseEvaluationMetric(
        name=name,
        criteria=criteria,
        input_variables=additional_input_variables,
        # The response_name for the pointwise metric is not used by our
        # custom get_evaluation_data, but we pass it for consistency.
        response_name=response_name,
    )

    super().__init__(
        name=name,
        criteria=criteria,
        additional_input_variables=additional_input_variables,
        response_name=response_name,
        group_by_config=None,
        pointwise_metric=custom_pointwise_metric,
    )
    # This autorater does not support pairwise comparison.
    self.pairwise_metric = None


class OpinionCategorizationCorrectnessPointwiseEvaluationMetric(
    PointwiseEvaluationMetric
):
  """Custom pointwise metric to evaluate opinion categorization correctness."""

  def get_evaluation_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
    """Creates a dataset to check for opinion categorization correctness."""
    if quote_extractor.QUOTE_COL not in input_data.columns:
      raise ValueError(
          f"Column '{quote_extractor.QUOTE_COL}' not found in"
          " input data."
      )
    if quote_extractor.TOPIC_COL not in input_data.columns:
      raise ValueError(
          f"Column '{quote_extractor.TOPIC_COL}' not found in input data."
      )
    if "opinion" not in input_data.columns:
      raise ValueError("Column 'opinion' not found in input data.")

    # Group by topic and quote to find all opinions assigned to each quote.
    opinions_by_quote = (
        input_data.groupby(
            [quote_extractor.TOPIC_COL, quote_extractor.QUOTE_COL]
        )["opinion"]
        .apply(list)
        .reset_index()
    )

    if opinions_by_quote.empty:
      return pd.DataFrame(
          columns=[
              quote_extractor.TOPIC_COL,
              quote_extractor.QUOTE_COL,
              "all_opinions",
              "response",
          ]
      )

    # Get all opinions for each topic
    topic_opinions = (
        input_data.groupby(quote_extractor.TOPIC_COL)["opinion"]
        .unique()
        .to_dict()
    )

    evaluation_rows = []
    for _, row in opinions_by_quote.iterrows():
      topic = row[quote_extractor.TOPIC_COL]
      quote = row[quote_extractor.QUOTE_COL]
      assigned_opinions = row["opinion"]

      all_opinions_in_topic = topic_opinions.get(topic, [])
      all_opinions_str = _format_list_as_string(all_opinions_in_topic)
      assigned_opinions_str = _format_list_as_string(list(assigned_opinions))

      evaluation_rows.append({
          quote_extractor.TOPIC_COL: topic,
          quote_extractor.QUOTE_COL: quote,
          "all_opinions": all_opinions_str,
          "response": assigned_opinions_str,
      })

    return pd.DataFrame(evaluation_rows)


class OpinionCategorizationEvaluationMetrics(EvaluationMetrics):
  """Metrics for evaluating opinion categorization correctness."""

  def __init__(self):
    name = "OpinionCategorizationCorrectness"
    criteria = prompts.eval_opinion_categorization_correctness_criteria
    additional_input_variables = [
        quote_extractor.TOPIC_COL,
        quote_extractor.QUOTE_COL,
        "all_opinions",
    ]
    # The response to evaluate is the list of assigned opinions.
    response_name = "assigned_opinions"

    custom_pointwise_metric = OpinionCategorizationCorrectnessPointwiseEvaluationMetric(
        name=name,
        criteria=criteria,
        input_variables=additional_input_variables,
        # The response_name for the pointwise metric is not used by our
        # custom get_evaluation_data, but we pass it for consistency.
        response_name=response_name,
    )

    super().__init__(
        name=name,
        criteria=criteria,
        additional_input_variables=additional_input_variables,
        response_name=response_name,
        group_by_config=None,
        pointwise_metric=custom_pointwise_metric,
    )
    # This autorater does not support pairwise comparison.
    self.pairwise_metric = None


QUOTE_EXTRACTION_METRICS = EvaluationMetrics(
    name="QuoteExtraction",
    criteria=prompts.get_eval_quote_extraction_criteria(
        survey_text_col=quote_extractor.SURVEY_TEXT_COL,
        topic_col=quote_extractor.TOPIC_COL,
    ),
    additional_input_variables=[
        quote_extractor.TOPIC_COL,
        quote_extractor.SURVEY_TEXT_COL,
    ],
    response_name="quote_with_brackets",
)

INPUT_EVAL_METRICS = EvaluationMetrics(
    name="InputCriteria",
    criteria=prompts.eval_input_data_criteria,
    additional_input_variables=[],
    response_name=quote_extractor.SURVEY_TEXT_COL,
)

OPINION_QUALITY_METRICS = EvaluationMetrics(
    name="OpinionQuality",
    criteria=prompts.eval_opinion_quality_criteria,
    additional_input_variables=["quote_aggregated"],
    # The aggregated version of this column is used.
    response_name="opinion",
    group_by_config=GroupByConfig(
        group_by_col="opinion",
        agg_col=quote_extractor.QUOTE_COL,
    ),
)

OTHER_OPINION_METRICS = OtherOpinionEvaluationMetrics()

OPINION_CATEGORIZATION_METRICS = OpinionCategorizationEvaluationMetrics()

AGREEMENT_METRICS = EvaluationMetrics(
    name="Agreement",
    criteria=prompts.eval_agreement_criteria,
    additional_input_variables=["question"],
    response_name="answer",
)

PROPOSITION_TOPIC_METRICS = EvaluationMetrics(
    name="PropositionTopic",
    criteria=prompts.eval_proposition_topic_criteria,
    additional_input_variables=["topic"],
    response_name="proposition",
)

PROPOSITION_OPINION_METRICS = EvaluationMetrics(
    name="PropositionOpinion",
    criteria=prompts.eval_proposition_opinion_criteria,
    additional_input_variables=["opinion"],
    response_name="proposition",
)
