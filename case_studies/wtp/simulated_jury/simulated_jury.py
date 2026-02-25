"""
Library for running simulated juries.
"""

import functools
import time
from enum import Enum
from typing import List
# try:
#     from itertools import batched
# except ImportError:
from itertools import islice
from google.genai import types as genai_types


# Backport of itertools.batched for python<3.12
def batched(iterable, n):
  "batched('ABCDEFG', 3) --> ABC DEF G"
  if n < 1:
    raise ValueError("n must be at least one")
  it = iter(iterable)
  while batch := tuple(islice(it, n)):
    yield batch


import logging
import os
import random
import json
from case_studies.wtp.models import genai_model
from case_studies.wtp import participation
import pandas as pd
import re

# Votes within ApprovalScale options that are considered "positive" approvals.
POSITIVE_APPROVAL_VOTES = frozenset([
    "Agree",
    "Strongly Agree",
    "Somewhat Agree",
])


class VotingMode(Enum):
  RANK = 1
  APPROVAL = 2


class ApprovalScale(Enum):
  AGREE_DISAGREE = "agree_disagree"
  AGREE_DISAGREE_NEITHER = "agree_disagree_neither"
  LIKERT_5 = "likert_5"
  LIKERT_5_SOMEWHAT = "likert_5_somewhat"
  LIKERT_4 = "likert_4"
  LIKERT_4_SOMEWHAT = "likert_4_somewhat"

  _mapping = {
      "agree_disagree": ["Agree", "Disagree"],
      "agree_disagree_neither": [
          "Agree",
          "Disagree",
          "Neither Agree nor Disagree",
      ],
      "likert_5": [
          "Strongly Agree",
          "Agree",
          "Neither Agree nor Disagree",
          "Disagree",
          "Strongly Disagree",
      ],
      "likert_5_somewhat": [
          "Strongly Agree",
          "Somewhat Agree",
          "Neither Agree nor Disagree",
          "Somewhat Disagree",
          "Strongly Disagree",
      ],
      "likert_4": [
          "Strongly Agree",
          "Agree",
          "Disagree",
          "Strongly Disagree",
      ],
      "likert_4_somewhat": [
          "Strongly Agree",
          "Somewhat Agree",
          "Somewhat Disagree",
          "Strongly Disagree",
      ],
  }

  @classmethod
  def get_options(cls, scale: "ApprovalScale") -> List[str]:
    return cls._mapping.value[scale.value]


def _compute_stats_summary(
    llm_response_df: pd.DataFrame,
    llm_response_stats_df: pd.DataFrame,
    jobs: List[genai_model.Job],
    duration_seconds: float,
    voting_mode: VotingMode,
    topic_name: str = "",
    opinion_name: str = "",
) -> dict:
  """Computes a summary of statistics from a simulated jury run."""
  total_jobs = len(jobs)
  if llm_response_stats_df.empty and total_jobs == 0:
    return {}

  n_complete_fails = int(llm_response_stats_df["is_complete_failure"].sum())
  percent_complete_fails = (
      (n_complete_fails / total_jobs) * 100 if total_jobs > 0 else 0
  )
  n_at_least_one_fail = llm_response_stats_df[
      llm_response_stats_df["non_quota_failures"] > 0
  ].shape[0]
  total_non_quota_fails = llm_response_stats_df["non_quota_failures"].sum()
  average_fails = total_non_quota_fails / total_jobs if total_jobs > 0 else 0
  avg_time_per_job = duration_seconds / total_jobs if total_jobs > 0 else 0
  prompt_char_counts = [job.get("prompt_char_count", 0) for job in jobs if job]
  total_prompt_char_count = sum(prompt_char_counts)

  # Sum up token counts from the main results DataFrame, handling cases where the
  # DataFrame is empty or columns are missing.
  total_tokens_used = int(
      llm_response_df["total_token_used"].sum()
      if "total_token_used" in llm_response_df
      else 0
  )
  total_prompt_tokens = int(
      llm_response_df["prompt_token_count"].sum()
      if "prompt_token_count" in llm_response_df
      else 0
  )
  total_candidates_tokens = int(
      llm_response_df["candidates_token_count"].sum()
      if "candidates_token_count" in llm_response_df
      else 0
  )
  total_tool_use_prompt_tokens = int(
      llm_response_df["tool_use_prompt_token_count"].sum()
      if "tool_use_prompt_token_count" in llm_response_df
      else 0
  )
  total_thoughts_tokens = int(
      llm_response_df["thoughts_token_count"].sum()
      if "thoughts_token_count" in llm_response_df
      else 0
  )

  return {
      "jury_type": voting_mode.name.lower(),
      "topic": topic_name if topic_name else "Nuanced",
      "opinion": opinion_name if opinion_name else None,
      "total_jobs": total_jobs,
      "n_complete_fails": n_complete_fails,
      "percent_complete_fails": percent_complete_fails,
      "n_at_least_one_fail": n_at_least_one_fail,
      "average_fails": average_fails,
      "total_duration_seconds": duration_seconds,
      "avg_time_per_job_seconds": avg_time_per_job,
      "total_prompt_char_count": total_prompt_char_count,
      "total_tokens_used": total_tokens_used,
      "total_prompt_tokens": total_prompt_tokens,
      "total_candidates_tokens": total_candidates_tokens,
      "total_tool_use_prompt_tokens": total_tool_use_prompt_tokens,
      "total_thoughts_tokens": total_thoughts_tokens,
  }


class StatementMapper:
  """Maps characters to statements for parsing LLM responses."""

  CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

  def __init__(self, statements: list[str]):
    if len(statements) > len(self.CHARS):
      raise ValueError(f"Cannot map more than {len(self.CHARS)} statements.")
    self.statements = statements
    self.letter_to_statement = {
        self.CHARS[i]: statement for i, statement in enumerate(self.statements)
    }
    self.statement_to_letter = {
        v: k for k, v in self.letter_to_statement.items()
    }

  def get_formatted_statements(self) -> str:
    """Returns the statements formatted with letter prefixes."""
    return "".join([
        f"{self.statement_to_letter[statement]}. {statement}\n"
        for statement in self.statements
    ])

  def convert_letters_to_statements(self, letters: list[str]) -> list[str]:
    """Converts a list of letters to a list of statements."""
    return [
        self.letter_to_statement[letter]
        for letter in letters
        if letter in self.letter_to_statement
    ]


def parse_llm_ranking_response(resp: dict, job: dict) -> dict:
  """Parses a single result from a simulated jury and returns a dict with ranking and reasoning."""
  statements = job["shuffled_statements"]
  mapper = StatementMapper(statements)
  raw_response_for_logging = resp["text"]

  try:
    data = json.loads(raw_response_for_logging)
    ranking_letters = data.get("ranking", [])
    reasoning = data.get("reasoning", "")

    ranking = mapper.convert_letters_to_statements(ranking_letters)

    if len(ranking) != len(statements):
      # On the final attempt, accept a partial ranking to avoid failing the job.
      is_last_attempt = (
          job.get("current_attempt", 0)
          >= job.get("retry_attempts", genai_model.MAX_LLM_RETRIES) - 1
      )
      if is_last_attempt:
        logging.warning(
            "Accepting partial ranking on final attempt (%d of %d statements).",
            len(ranking),
            len(statements),
        )
      else:
        raise ValueError(
            "Incomplete ranking. Expected"
            f" {len(statements)}, got {len(ranking)}."
        )

  except json.JSONDecodeError as e:
    logging.warning(
        "Could not parse ranking from response:"
        f" {raw_response_for_logging}. Error: {e}"
    )
    raise ValueError(f"Parsing failed with JSONDecodeError: {e}") from e
  except (IndexError, KeyError, ValueError) as e:
    logging.warning(
        "Could not parse ranking from response:"
        f" {raw_response_for_logging}. Error: {e}"
    )
    raise ValueError(f"Parsing failed with error: {e}") from e

  return {
      "ranking": ranking,
      "reasoning": reasoning,
      "raw_response": raw_response_for_logging,
  }


def parse_llm_approval_response(resp: dict, job: dict) -> dict:
  """
  Parses a single JSON result from a simulated jury in approval mode.
  """
  statements = job["shuffled_statements"]
  mapper = StatementMapper(statements)
  raw_response_for_logging = resp["text"]

  try:
    data = json.loads(raw_response_for_logging)
    votes = data.get("votes", [])

    if len(votes) != len(statements):
      raise ValueError(
          f"Incomplete voting. Expected {len(statements)}, got {len(votes)}."
      )

    approval_dict = {}
    for vote in votes:
      statement_letter = vote.get("statement_letter")
      vote_value = vote.get("vote")
      if statement_letter in mapper.letter_to_statement:
        statement_text = mapper.letter_to_statement[statement_letter]
        # Count 'Agree', 'Strongly Agree', and 'Somewhat Agree' as agreement.
        approval_dict[statement_text] = vote_value in POSITIVE_APPROVAL_VOTES

    return approval_dict

  except json.JSONDecodeError as e:
    logging.warning(
        "Could not parse approval from response:"
        f" {raw_response_for_logging}. Error: {e}"
    )
    raise ValueError(f"Parsing failed with JSONDecodeError: {e}") from e
  except (IndexError, KeyError, ValueError) as e:
    logging.warning(
        "Could not parse approval from response:"
        f" {raw_response_for_logging}. Error: {e}"
    )
    raise ValueError(f"Parsing failed with error: {e}") from e


def generate_vote_prompt(
    participant_id: str,
    participation_record: str,
    statements: list[str],
    voting_mode: VotingMode,
    approval_scale: ApprovalScale = ApprovalScale.AGREE_DISAGREE,
) -> str:
  """Generates a prompt for a simulated jury vote."""
  prompt = f"""A participant in a public deliberation expressed the following opinions:

<participant id="{participant_id}">
{participation_record}
</participant>

Statements:
"""
  mapper = StatementMapper(statements)
  prompt += mapper.get_formatted_statements()

  if voting_mode == VotingMode.APPROVAL:
    options = ApprovalScale.get_options(approval_scale)
    options_str = ", ".join([f'"{option}"' for option in options])
    approval_prompt = f"""
Task: As an AI assistant, your job is to predict the participant's vote on each of the following statements based on their opinion.

Please think through this step-by-step:
1. Analyze the participant’s opinion, noting key points and sentiments.
2. For each statement, compare it to the participant’s opinion and decide on their vote.
3. Formulate a concise reasoning for your overall set of predictions.

Finally, provide your response as a JSON object containing your reasoning and a list of your votes. Each vote object in the list should include the statement's letter and your prediction. The prediction must be one of the following values: {options_str}.
"""
    prompt += approval_prompt
  elif voting_mode == VotingMode.RANK:
    prompt += """
Task: As an AI assistant, your job is to rank these statements in the order that the participant would most likely agree with them, based on their opinion.

Please think through this step-by-step:
1. Analyze the participant’s opinion, noting key points and sentiments.
2. Compare each statement to the participant’s opinion, considering how well it aligns with or supports their view.
3. Consider any nuances or implications in the statements that might appeal to or repel the participant based on their expressed opinion.
4. Rank the statements accordingly.

Finally, call the `submit_ranking` tool with your final ranking and a concise reasoning for your choice. The ranking should be an array of the statement letters, from most to least preferred. It is critical that every statement is included in the final ranking exactly once.
"""

  return prompt


async def run_simulated_jury(
    participants_df: pd.DataFrame,
    statements: list[str],
    voting_mode: VotingMode,
    model: genai_model.GenaiModel,
    topic_name: str = "",
    opinion_name: str = "",
    batch_size: int = None,
    approval_scale: ApprovalScale = ApprovalScale.AGREE_DISAGREE,
):
  """Runs a simulated jury for a set of participants and statements."""
  num_participants = len(participants_df)
  logging.info(
      "--- Running simulated jury with %d participants ---", num_participants
  )
  print(f"--- Running simulated jury with {num_participants} participants ---")
  jobs = []

  # Define the schema for the ranking tool
  rank_schema = {
      "type": "OBJECT",
      "properties": {
          "reasoning": {
              "type": "STRING",
              "description": (
                  "A concise summary of the reasoning for the ranking, based on"
                  " the participant's opinions."
              ),
          },
          "ranking": {
              "type": "ARRAY",
              "items": {"type": "STRING"},
              "description": (
                  "The final ranked list of statement letters, from most to"
                  " least preferred. Example: ['B', 'A', 'D', 'C']"
              ),
          },
      },
      "required": ["reasoning", "ranking"],
  }
  approval_schema = {
      "type": "OBJECT",
      "properties": {
          "reasoning": {
              "type": "STRING",
              "description": (
                  "A concise summary of the reasoning for the agree/disagree"
                  " choices, based on the participant's opinions."
              ),
          },
          "votes": {
              "type": "ARRAY",
              "items": {
                  "type": "OBJECT",
                  "properties": {
                      "statement_letter": {"type": "STRING"},
                      "vote": {
                          "type": "STRING",
                          "enum": ApprovalScale.get_options(approval_scale),
                      },
                  },
                  "required": ["statement_letter", "vote"],
              },
          },
      },
      "required": ["reasoning", "votes"],
  }

  # Determine if batching is needed
  use_batching = (
      batch_size
      and voting_mode == VotingMode.APPROVAL
      and len(statements) > batch_size
  )

  for i, (_, row) in enumerate(participants_df.iterrows()):
    # Create multiple jobs for this participant, one for each batch
    for j, batch_statements_tuple in enumerate(
        batched(statements, batch_size or len(statements))
    ):
      shuffled_batch = list(batch_statements_tuple)
      random.shuffle(shuffled_batch)

      prompt = generate_vote_prompt(
          row["rid"],
          participation.get_prompt_representation(row),
          shuffled_batch,
          voting_mode,
          approval_scale,
      )

      schema = None
      if voting_mode == VotingMode.RANK:
        schema = rank_schema
      elif voting_mode == VotingMode.APPROVAL:
        schema = approval_schema

      jobs.append({
          "job_id": f"{i}-{j}",
          "participant_rid": row["rid"],  # Add rid for later merging
          "topic": topic_name,
          "opinion": opinion_name,
          "prompt": prompt,
          "prompt_char_count": len(prompt),
          "data_row": row.to_dict(),
          "shuffled_statements": shuffled_batch,
          "thinking_level": genai_types.ThinkingLevel.HIGH,
          "response_schema": schema,
          "response_mime_type": "application/json",
      })

  # Choose the correct parser based on the voting mode.
  parser = None
  if voting_mode == VotingMode.RANK:
    parser = parse_llm_ranking_response
  elif voting_mode == VotingMode.APPROVAL:
    parser = parse_llm_approval_response
  else:
    raise ValueError(f"Unsupported voting mode: {voting_mode}")

  start_time = time.monotonic()
  llm_response_df, llm_response_stats_df, _, _ = (
      await model.process_prompts_concurrently(
          jobs,
          response_parser=parser,
          max_concurrent_calls=100,
          retry_attempts=4,
      )
  )
  end_time = time.monotonic()
  duration = end_time - start_time

  # --- Aggregate Failure Statistics ---
  stats_summary = _compute_stats_summary(
      llm_response_df,
      llm_response_stats_df,
      jobs,
      duration,
      voting_mode,
      topic_name,
      opinion_name,
  )

  # If we used batching, we need to merge the results back together
  if use_batching and not llm_response_df.empty:
    merged_results = []
    for rid, group in llm_response_df.groupby("participant_rid"):
      # Merge the 'result' dictionaries for this participant
      combined_result_dict = {}
      for res_dict in group["result"]:
        if isinstance(res_dict, dict):
          combined_result_dict.update(res_dict)

      # Create a new row representing the merged result for this participant
      new_row = group.iloc[0].copy()
      new_row["result"] = combined_result_dict
      merged_results.append(new_row)

    # Create a new DataFrame from the merged results
    final_df = pd.DataFrame(merged_results).reset_index(drop=True)
    return final_df, stats_summary

  # The 'result' column now contains dictionaries from the parsers.
  return llm_response_df, stats_summary


def build_approval_matrix(
    approval_results_df: pd.DataFrame,
    approval_scale: ApprovalScale = ApprovalScale.AGREE_DISAGREE,
) -> pd.DataFrame:
  """Builds a participant-by-proposition approval matrix from raw results."""
  approvals = []
  for _, row in approval_results_df.iterrows():
    participant_id = row["data_row"]["rid"]
    # The result is a dictionary mapping proposition to boolean approval
    for proposition, approved in row["result"].items():
      if proposition == "error":
        continue
      approvals.append({
          "participant_id": participant_id,
          "proposition": proposition,
          "approved": approved,
      })

  if not approvals:
    return pd.DataFrame()

  approval_df = pd.DataFrame(approvals)
  print(f"DEBUG: dtypes of approval_df before pivot:\n{approval_df.dtypes}")
  # Pivot to create the matrix: participants as index, propositions as columns
  approval_matrix = approval_df.pivot_table(
      index="participant_id",
      columns="proposition",
      values="approved",
      fill_value=False,
  )
  return approval_matrix
