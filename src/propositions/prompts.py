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

"""
This module contains functions for generating prompts and parsing responses
for the world model builder.
"""

from src.propositions import prompts_util
import pandas as pd
from typing import List


# Preamble prompt methods
def generate_preamble_prompt(
    opinion_list: List[str], additional_context: str | None = None
) -> str:
  """
  Generates the preamble for the proposition generation prompt.

  Args:
    opinion_list: Full list of opinions that were generated after R1.
    additional_context: Optional additional context to be added to the prompt.

  Returns:
    The prompt to be added to the context window.
  """
  prompt = f"""# Role and Objective

You are a qualitative data analyst. Your objective is to refine a single assigned `opinion` into one or more `statements` that better represent the common ground found in two sets of survey data. You are part of a parallel process where each instance is focused on only one opinion, but you have the full list for context.

# Key Definitions

  * A **topic** is a broad theme from Survey 1.
  * An **opinion** is a one-sentence summary of a specific viewpoint within a topic.
  * A **quote** is a direct excerpt from a participant's response in Survey 1.
  * A **statement** is the final, refined output you will generate. It represents a point of common ground, is written as a declarative fact, and is accessible at a 5th-grade reading level.
"""

  if additional_context:
    prompt += f"""
<additionalContext>
  {additional_context}
</additionalContext>
"""

  prompt += """
# Your Assigned Task Data

  * **Assigned Opinion to Evaluate:** `<opinion>`
  * **Full Opinion List for Context:**
    <full_opinion_list>"""
  for opinion in opinion_list:
    prompt += f"\n      <opinion>{opinion}</opinion>"
  prompt += """
    </full_opinion_list>
  * **Survey 1 Data (`<R1_DATA>`):** Contains the original quotes grouped by the opinions they informed.
  * **Survey 2 Data (`<R2_DATA>`):** Contains participant feedback on a specific quote related to the assigned opinion.
"""
  return prompt


def generate_instructions_prompt(
    number_of_propositions: int,
    reasoning: bool = False,
    include_opinion: bool = True,
):
  """
  Generates the instructions for the proposition generation prompt.

  Args:
    number_of_propositions: List of maximum amount of propositions to input into
      the prompt.
    reasoning: Bool flag indicating the proposition data includes reasoning.
    include_opinion: Bool flag indicating if the original opinion should be
      included in the final set of statements generated.

  Returns:
    The prompt to be added to the context window.
  """
  keep_original_step = (
      """
    a.  **Keep the Original:** Include the original opinion as the first statement in your output. Its reasoning should explain its value but also note its limitations."""
      if include_opinion
      else ""
  )

  new_statements_step = (
      f"""
    b.  **Draft New Statements:** Generate a maximum of `{number_of_propositions}` **new** statements to cover the identified gaps."""
      if include_opinion
      else f"""
    a.  **Draft Statements:** Generate a maximum of `{number_of_propositions}` statements to cover the identified gaps."""
  )

  sufficient_opinion_instruction = (
      "Return only the original opinion (in unedited form) as your statement"
      if include_opinion
      else (
          "You may return the original opinion in unedited or lightly edited"
          " form"
      )
  )

  return (
      f"""
# Step-by-Step Instructions

1.  **Synthesize Key Themes:** Analyze all participant `quotes` associated with your **Assigned Opinion** in `<R1_DATA>`. Then, analyze the additional participant feedback in `<R2_DATA>`. Identify the core ideas and points of agreement expressed across both datasets.

2.  **Evaluate the Assigned Opinion:** Compare the synthesized themes from Step 1 against the **Assigned Opinion**. Ask yourself: "Does the original opinion fully and accurately capture the main, agreed-upon sentiments from the participant data?"

3.  **Generate Statements:**

 * **IF the opinion is sufficient:** The original opinion perfectly captures the shared sentiment, while satisfying other instructions and requirements. {sufficient_opinion_instruction}, with an empty string for the reasoning.
  * **ELSE the opinion is insufficient:** The data reveals a significant, shared viewpoint that the original opinion misses, misrepresents, oversimplifies, or in some other way does not satisfy other instructions. In this case:{keep_original_step}{new_statements_step}
    c.  **Adhere to Statement Rules:**
      * Each statement must be a **single, declarative sentence** (e.g., "Economic opportunity is essential for equality."). Do not use "I think" or "We believe."
      * Write in simple, clear language at a **5th-grade reading level**.
      * Ensure new statements are **substantively different** from any opinion in the **Full Opinion List**. Your goal is to refine, not duplicate.

4.  **Provide Concise Reasoning:** For each statement you generate, provide a `reasoning` string. The reasoning must be 1-3 sentences and explain *why* the statement is necessary, referencing the survey data. For example: "The original opinion focused only on X, but analysis of R1 and R2 shows 15 of 27 participants also strongly agree that Y is a critical component."

# Anti-Redundancy Check

Focus strictly on the data related to your **Assigned Opinion**. Do not generate a statement if its core idea is more directly covered by another opinion in the **Full Opinion List**. For example, if your assigned opinion is about "economic equality" and you see themes about "political freedom," do not create a statement about freedom; trust that the parallel job assigned to the "freedom" opinion will handle it.

# Output Format

Provide your response as a single JSON array of objects, as shown in the example below.

[
  {{
    "statement": "The original opinion, presented here as a statement.",
    "reasoning": "This statement is the baseline, but it overlooks the recurring theme of equal access to resources, which was mentioned by over half the participants in R2."
  }},
  {{
    "statement": "A new, declarative statement capturing a missed theme.",
    "reasoning": "This new statement was created because the data showed strong agreement on the importance of resource access, a point not covered in the original opinion list."
  }}
]
"""
      if reasoning
      else f"""# Step-by-Step Instructions

1.  **Synthesize Key Themes:** Analyze all participant `quotes` associated with your **Assigned Opinion** in `<R1_DATA>`. Then, analyze the additional participant feedback in `<R2_DATA>`. Identify the core ideas and points of agreement expressed across both datasets.

2.  **Evaluate the Assigned Opinion:** Compare the synthesized themes from Step 1 against the **Assigned Opinion**. Ask yourself: "Does the original opinion fully and accurately capture the main, agreed-upon sentiments from the participant data?"

3.  **Generate Statements:**

 * **IF the opinion is sufficient:** The original opinion perfectly captures the shared sentiment, while satisfying other instructions and requirements. {sufficient_opinion_instruction}.
  * **ELSE the opinion is insufficient:** The data reveals a significant, shared viewpoint that the original opinion misses, misrepresents, oversimplifies, or in some other way does not satisfy other instructions. In this case:{keep_original_step.split(' Its reasoning')[0] if include_opinion else ''}{new_statements_step}
    c.  **Adhere to Statement Rules:**
      * Each statement must be a **single, declarative sentence** (e.g., "Economic opportunity is essential for equality."). Do not use "I think" or "We believe."
      * Write in simple, clear language at a **5th-grade reading level**.
      * Ensure new statements are **substantively different** from any opinion in the **Full Opinion List**. Your goal is to refine, not duplicate.

# Anti-Redundancy Check

Focus strictly on the data related to your **Assigned Opinion**. Do not generate a statement if its core idea is more directly covered by another opinion in the **Full Opinion List**. For example, if your assigned opinion is about "economic equality" and you see themes about "political freedom," do not create a statement about freedom; trust that the parallel job assigned to the "freedom" opinion will handle it.

# Output Format

Provide your response as a single JSON array of objects, as shown in the example below.

[
  "The original opinion, presented here as a statement.",
  "A new, declarative statement capturing a missed theme.",
  ...
]
"""
  )


# R1 methods.
def generate_r1_prompt_string(
    df: pd.DataFrame,
    user_id_column_name: str,
    topic_column_name: str,
    opinion_column_name: str,
    should_use_representative_text: bool = True,
    representative_text_column_name: str = "representative_text",
    should_use_opinion_sharding: bool = True,
) -> str:
  """
  Generates a prompt text from participants from R1 surveys on a given topic
  which will be used for Proposition generation.

  Args:
    df: A pandas DataFrame containing comments and topics.
    user_id_column_name: The name of the column containing user ID associated with the comments.
    topic_column_name: The name of the column containing the topic the row is associated with.
    opinion_column_name: The name of the column containing the opinion the row is associated with.
    should_use_representative_text: Bool flag that indicates if to use representative text or full survey data.
    representative_text_column_name: The name of the column containing representative text.
    should_use_opinion_sharding: Bool flag indicating whether to shard by opinion or topic.

  Returns:
    The prompt to be added to the context window.
  """

  if should_use_representative_text and (
      representative_text_column_name is None
      or representative_text_column_name not in df.columns
  ):
    raise ValueError(
        "Column name for representative text must not be empty and should"
        " exist in the DataFrame."
    )

  if user_id_column_name is None or user_id_column_name not in df.columns:
    raise ValueError("user_id_column_name must be present in the DataFrame.")
  elif topic_column_name is None or topic_column_name not in df.columns:
    raise ValueError(
        "Column name for topics must not be empty and should exist in the"
        " DataFrame."
    )
  elif opinion_column_name is None or opinion_column_name not in df.columns:
    raise ValueError(
        "Column name for opinions must not be empty and should exist in the"
        " DataFrame."
    )

  prompt = "<R1_DATA>\n"

  # Common part of the prompt. This is used for when we want toavoid having
  # repeated strings in the prompt.
  if should_use_opinion_sharding:
    prompt += f"""<topic>{df.iloc[0][topic_column_name]}</topic>
<opinion>{df.iloc[0][opinion_column_name]}</opinion>
"""

  # Construct the prompt in light XML format from itterating over rows.
  for _, row in df.iterrows():
    user_id_text = row[user_id_column_name]

    # Use representative text instead of full text.
    if should_use_representative_text:
      prompt += f"""<participant id={user_id_text}>"""

      # If the topic and opinion strings have not been moved to the top add
      # them here.
      if not should_use_opinion_sharding:
        prompt += f"""\n<topic>{row[topic_column_name]}</topic>
<opinion>{row[opinion_column_name]}</opinion>\n"""

      # Add representative text.
      newline = "\n"
      prompt += f"""{row[representative_text_column_name].replace(newline, " ").replace('"', '')}"""
    else:

      # Use full text instead of representative text.
      prompt += f"""<participant id={user_id_text}>"""

      # If the topic and opinion strings have not been moved to the top add
      # them here.
      if not should_use_opinion_sharding:
        prompt += f"""<topic>{row[topic_column_name]}</topic>
<opinion>{row[opinion_column_name]}</opinion>\n"""

      # Add the full text. Standard Q and QFU pairs
      for i in range(1, 11):  # Assuming up to 10 Q/A pairs
        q_text_col = f"Q{i}_Text"
        q_col = f"Q{i}"
        qfu_text_col = f"Q{i}FU_Text"
        qfu_col = f"Q{i}FU"

        if q_text_col in row and pd.notna(row[q_text_col]):
          prompt += f"<question_{i}>{row[q_text_col]}</question_{i}>\n"
        if q_col in row and pd.notna(row[q_col]):
          prompt += f"<answer_{i}>{row[q_col]}</answer_{i}>\n"
        if qfu_text_col in row and pd.notna(row[qfu_text_col]):
          prompt += f"<question_fu_{i}>{row[qfu_text_col]}</question_fu_{i}>\n"
        if qfu_col in row and pd.notna(row[qfu_col]):
          prompt += f"<answer_fu_{i}>{row[qfu_col]}</answer_fu_{i}>\n"

    prompt += "</participant>\n"

  prompt += "</R1_DATA>\n"
  return prompt


# R2 methods.
def generate_r2_prompt_string(
    df: pd.DataFrame,
    include_non_gov_sections: bool = False,
) -> str:
  """Generates the context for R2 surveys.

  Args:
    df: A pandas DataFrame containing R2 survey data.
    include_non_gov_sections: A boolean indicating whether to include non-GOV
      sections. Defaults to False.

  Returns:
    The prompt to be added to the context window.
  """
  r2_prompt_first_line = "<R2_DATA>\n"
  r2_prompt = r2_prompt_first_line
  r2_df = df.copy()

  # Extract opinions from GOV that is repeated to the top of the prompt with a
  # unique id.
  # This method finds the opinions and assignes them ids and creates a prompt
  # text to be added to the begining of the prompt block so the strings are not
  # repeated.
  free_text_prompt_header, free_text_opinions_map = (
      prompts_util.extract_reusable_strings(
          df=r2_df, question_type=prompts_util.QuestionType.FREE_TEXT
      )
  )

  # If there are any common opinions then add them to the begingin of the prompt.
  if free_text_prompt_header:
    r2_prompt += free_text_prompt_header

  # Extract opinions from ranking section to the top of the prompt with a
  # unique id.
  if include_non_gov_sections:
    ranking_prompt_header, ranking_opinions_map = (
        prompts_util.extract_reusable_strings(
            r2_df, prompts_util.QuestionType.RANKING
        )
    )
    if ranking_prompt_header:
      r2_prompt += ranking_prompt_header

  # Add the rest of the data by row.
  for _, row in r2_df.iterrows():
    user_id = row["rid"]
    # Note user's id.
    r2_prompt += f"<participant id={user_id}>"
    # Build the user data for prompt.
    free_form_prompt = prompts_util.build_free_text_response_prompt(
        row, free_text_opinions_map
    )
    if free_form_prompt:
      # If there are other sections then wrap this section with response tag.
      if include_non_gov_sections:
        r2_prompt += f"\n<response type='freetext'>\n"

      r2_prompt += free_form_prompt

      # If there are other sections then close this section tag.
      if include_non_gov_sections:
        r2_prompt += f"</response>\n"
    if include_non_gov_sections:
      ranking_prompt = prompts_util.build_ranking_response_prompt(
          row, ranking_opinions_map
      )
      if ranking_prompt:
        r2_prompt += (
            "Here is how this participant ranked the opinions "
            "that are listed above in order they agree with the most. "
            "After ranking they were asked a single followup question.\n"
        )
        r2_prompt += ranking_prompt
    r2_prompt += "</participant>\n"

  if r2_prompt != r2_prompt_first_line:
    r2_prompt += "</R2_DATA>\n"
    return r2_prompt
  else:
    return ""
