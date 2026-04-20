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

"""
This module is a centralized library for all LLM prompts used across the project,
including task instructions, evaluation criteria, and system messages.
"""

from typing import List

import pandas as pd
from src.propositions import prompts_util


def proposition_generation_generate_preamble_prompt(
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


def proposition_generation_generate_instructions_prompt(
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
def proposition_generation_generate_r1_prompt_string(
    df: pd.DataFrame,
    user_id_column_name: str,
    topic_column_name: str,
    opinion_column_name: str,
    should_use_quote: bool = True,
    quote_column_name: str = "quote",
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
    should_use_quote: Bool flag that indicates if to use quote or full survey data.
    quote_column_name: The name of the column containing quote.
    should_use_opinion_sharding: Bool flag indicating whether to shard by opinion or topic.

  Returns:
    The prompt to be added to the context window.
  """

  if should_use_quote and (
      quote_column_name is None
      or quote_column_name not in df.columns
  ):
    raise ValueError(
        "Column name for quote must not be empty and should"
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

    # Use quote instead of full text.
    if should_use_quote:
      prompt += f"""<participant id={user_id_text}>"""

      # If the topic and opinion strings have not been moved to the top add
      # them here.
      if not should_use_opinion_sharding:
        prompt += f"""\n<topic>{row[topic_column_name]}</topic>
<opinion>{row[opinion_column_name]}</opinion>\n"""

      # Add quote.
      newline = "\n"
      prompt += f"""{row[quote_column_name].replace(newline, " ").replace('"', '')}"""
    else:

      # Use full text instead of quote.
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
def proposition_generation_generate_r2_prompt_string(
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
    user_id = row["participant_id"]
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


topic_modeling_learn_topics_prompt = """
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
Your task is to analyze this entire data,
identify the topics according to the criteria below,
and then generate a JSON output with the identified topics.

Important Context:
There will be another round, where participants will be exploring topics, so it will be easier for them to tie everything together in their head, if they see topics being linked to the main subject.
To facilitate this connection, we want topic language to more explicitly connect to the main subject. Topics should be phrased like they are aspects of the main subject.

### **Criteria for Topics**

  * **Distinct:** Topics should be meaningfully different and cover separate conceptual areas.
  * **Substantive:** Topics should contain multiple, distinct opinions. Do not create single-opinion topics.
  * **Subject Linkage:** Topic names should be phrased as aspects of the main subject. E.g. "Freedom and Equality" subject could have the following topics: "Defining Freedom", "Defining Equality", "Barriers to Freedom and Equality", "Society's Role in Freedom and Equality", "The Individual's Role in Freedom and Equality", etc.
  * **Concise:** The topic name should be concise.
  * **Consistent Scope:** Topics should be at a similar level of abstraction (e.g., don't mix a very broad topic with a very narrow one).
  * **Efficiency:** Keep the number of topics as low as possible. Actively consolidate topics when their content can be logically grouped.


RESPONSE STRUCTURE:
Respond with a list of the identified topics only, nothing else.
The response should be in JSON format, that can be parse into the following class:
class FlatTopicList:
    topics: List[Topic]
class Topic:
    name: str

Do not include markdown code blocks around the JSON response, such as ```json or ```
Response example:
{"topics": [{"name": "Topic 1"}, {"name": "Topic 2"}]}
"""


def get_topic_modeling_opinions_prompt(parent_topic_name: str) -> str:
  return f"""
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
Your task is to analyze this entire dataset of quotes, identify the opinions on the following topic: "{parent_topic_name}" according to the criteria below, and then generate a JSON output with the identified opinions.

### **Criteria for Opinions**

1.  **Active Voice & Direct Phrasing (Crucial):**
    * Use strong, active verbs. Avoid passive voice (e.g., "It is believed that...").
    * Avoid abstract policy speak. Instead of "To improve economic opportunity, there needs to be investment in...", write "Our city must invest in..." or "Schools need better funding to..."
    * **Do not** use words like "perception" or "sentiment." State the opinion as a fact as viewed by the participant.

2.  **Avoid Repetitive Sentence Starters:**
    * **Do not** start every opinion with the same phrase (e.g., stop using "To strengthen the social safety net..." for every single item).
    * Ensure the list has syntactic variety while remaining thematically tight.

3.  **Simplify & Avoid Complex Parallelisms:**
    * **One idea per opinion.** Avoid complex lists (e.g., "We need X, Y, and Z, while also ensuring A and B").
    * Aim for a 5th-grade reading level. Keep it simple and punchy.

4.  **Distinct & Substantive:**
    * Opinions must represent unique viewpoints within the topic.
    * Merge overlaps: Actively consolidate opinions when their content can be logically grouped.
    * Do not create single-quote opinions or opinions with very few opinions compared to its peers. A long tail of opinions is extremely undesirable.
    * Overall, we want to tightly curate opinions to help the user understand the main perspective within a topic, but we do not want to overwhelm them with a laundry list.  **Keep the number of opinions as low as possible.**

5.  **Topic Linkage (Without Repetition):**
    * The opinion must be clearly relevant to "{parent_topic_name}", but it should not rigidly repeat the topic name in the text.
    * *Bad:* "A barrier to economic growth is the lack of jobs."
    * *Good:* "A lack of quality jobs prevents the economy from growing."

### **Response Structure**
Respond only with the identified opinions, where top level is the overarching topic, and opinions are subtopics.
The response should be in JSON format, that can be parse into the following class:
class Topic:
    name: str # This will be the overarching topic
    subtopics: List[Topic] # Where subtopic is class Topic {{ name: str }} (the opinions)

Do not include markdown code blocks around the JSON response, such as ```json or ```
For example:
{{
  "name": "{parent_topic_name}",
  "subtopics": [
      {{ "name": "Opinion 1" }},
      {{ "name": "Opinion 2" }}
  ]
}}
"""


def get_topic_modeling_merge_opinions_prompt(parent_topic_name: str) -> str:
  return f"""
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
You have been provided with multiple lists of opinions for the topic "{parent_topic_name}" that were generated by analyzing different chunks of the same dataset.
Your task is to synthesize and consolidate these lists by merging similar opinions, and generate a final, deduplicated list of opinions.

### **Consolidation Criteria**

  * **Distinct:** Within a single topic, opinions should represent unique viewpoints. Opinions should be meaningfully distinct and different within the topic.
  * **Substantive:** Opinions should be substantive (i.e. not single-quote opinions).
  * **Accurate:** The text of opinion should be a clear and well-phrased summary of the underlying quotes.
  * **Concise:** The opinions should be concise.
  * **Topic and Subject Linkage:** Opinions should be phrased to have a clear link to the overarching topic and the main subject of the survey.
  * **Coherency:** Opinions should be logically consistent and easy to follow as you go over the list. E.g. for "Barriers to Freedom and Equality" topic, all the opinions could start with: "A key barrier to freedom and equality is "; for "Defining Equality" topic: "Equality is ", etc.
  * **Merge Overlaps:** If two or more opinions express the same fundamental idea, they **must be merged**.
  * **Efficiency:** Keep the number of opinions as low as possible. Actively consolidate opinions when their content can be logically grouped.

When creating opinions, keep in mind that on later stages extracted quotes will be categorized into the identified opinions.
For that, the quote must holistically match the entire opinion. A partial match, where the quote only supports one piece of the opinion, is not sufficient.
To be a match, the quote must explicitly support every key concept within the opinion.
So avoid creating opinions that will be hard to completely match to quotes.

RESPONSE STRUCTURE:
Respond only with the identified opinions, where top level is the overarching topic, and opinions are subtopics.
The response should be in JSON format, that can be parse into the following class:
class Topic:
    name: str # This will be the overarching topic
    subtopics: List[Topic] # Where subtopic is class Topic {{ name: str }} (the opinions)

Do not include markdown code blocks around the JSON response, such as ```json or ```
For example:
{{
  "name": "{parent_topic_name}",
  "subtopics": [
      {{ "name": "Opinion 1" }},
      {{ "name": "Opinion 2" }}
  ]
}}
"""

categorization_opinion_main_rules = """
Main Rules:
- PRIORITY RULE: Select Only the Most Literal Match(es). You must evaluate all opinions. After evaluating all of them, select only the opinion or opinions that are the most literal, most explicit, and most holistic match and require the least inference.
- Keep the number of selections to the absolute minimum. Only select more than one opinion if they are both equally perfect, literal matches to the quote.
- PRIORITY RULE: The Main Thesis MUST Agree. You must first compare the primary claim of the quote and the opinion. If the primary claims contradict (e.g., 'equal opportunity' vs. 'same results'), you must not select that category, even if supporting examples (like 'housing') are similar.
- The quote must holistically match the entire opinion. A partial keyword match (like matching 'equality' but ignoring the rest of the quote) is not sufficient. To be a match, the quote must explicitly support every key concept within the opinion.
- The quote and the opinion must be making the same kind of claim. Do not match a personal definition (e.g., 'Freedom means...') to a conditional argument (e.g., 'Freedom isn't real until...').
- Be Aggressively Literal / No Inference. Do not make inferences or assumptions. Do not make logical leaps or semantic substitutions. If an opinion mentions a concept like 'dignity,' the quote must also mention 'dignity' or 'respect.' Do not infer that 'equal rights' is the same as 'dignity.' Do not infer 'regime' means 'corrupt government.'
- Match Abstraction Levels. Do not match specific examples (like 'targeting based on skin color') to a broad category (like 'culture of hate').
"""


def get_topic_categorization_prompt(topics_json_str: str) -> str:
  """Generates the prompt for categorizing statements into topics."""
  return f"""
For each of the following statements, identify any relevant topic from the list below.
Input Topics:
{topics_json_str}

Important Considerations:
- Ensure the assigned topic accurately reflects the meaning of the statement.
- If relevant and necessary (e.g. when a statement contains multiple disjoint claims), a statement can be assigned to multiple topics.
- Prioritize using the existing topics whenever possible. Keep the "Other" topic to minimum, ideally keep it empty.
- Use "Other" topic if the statement is completely off-topic and doesn't really fit any of the topics.
- All statements must be assigned at least one existing topic.
- Do not create any new topics that are not listed in the Input Topics.
- When generating the JSON output, minimize the size of the response. For example, prefer this compact format: {{"id": "5258", "topics": [{{"name": "Arts, Culture, And Recreation"}}]}} instead of adding unnecessary whitespace or newlines.

class StatementRecordList(BaseModel):
    items: list[StatementRecord]

class StatementRecord(BaseModel):
    id: str = Field(description="The unique identifier of the statement.")
    topics: list[Topic] = Field(description="A list of topics assigned to the statement.")

class Topic(BaseModel):
    name: str

You must follow the rules for the instructions strictly.
Pay close attention to Rules "Most Literal Match" and "Holistic Match". Do not select any opinion that is only a partial match or requires an inference if a more literal match is available.

Response must be a valid JSON object matching StatementRecordList schema. Example:
{{
  "items": [
    {{
      "id": "5258",
      "topics": [{{"name": "Arts, Culture, And Recreation"}}]
    }}
  ]
}}
"""


def get_categorization_opinion_prompt(opinions_json_str: str) -> str:
  return f"""
Categorize the following quotes based on the provided opinions.

Input Opinions:
{opinions_json_str}

{categorization_opinion_main_rules}

Other rules:
- Prioritize using the existing opinions whenever possible.
- Use "Other" opinion if the quote is completely off-topic and doesn't really fit any of the opinions. Keep the "Other" opinion to minimum, ideally keep it empty.
- All quotes must be assigned at least one existing opinion.
- Do not create any new opinions that are not listed in the Input Opinions.
- Respond with a JSON array of objects, each with "id", "quote_id" (which is the id of the quote), and "topics" (A list of opinions assigned to the quote, each with a "name" key).
- When generating the JSON output, minimize the size of the response. For example, prefer this compact format: {{"id": "5258", "quote_id": "q1", "topics": [{{"name": "Opinion from the Input list"}}]}} instead of adding unnecessary whitespace or newlines.

VERY IMPORTANT:
Double check to make sure that all quote ids and topic names are in the input. For example, if an input quote_id is '1183-Defining Freedom', then the output quote_id should be the same '1183-Defining Freedom', and not anything else like '1183' or '1188-Defining Freedom'.

class StatementRecordList(BaseModel):
    items: list[StatementRecord]

class StatementRecord(BaseModel):
    id: str = Field(description="The unique identifier of the statement.")
    quote_id: str = Field(description="The unique identifier of the quote.")
    topics: list[Topic] = Field(description="A list of opinions assigned to the quote.")

class Topic(BaseModel):
    name: str

You must follow the rules for the instructions strictly.
Pay close attention to Rules "Most Literal Match" and "Holistic Match". Do not select any opinion that is only a partial match or requires an inference if a more literal match is available.

Response must be a valid JSON object matching StatementRecordList schema. Example:
{{
  "items": [
     {{
       "id": "5258",
       "quote_id": "q1",
       "topics": [
          {{"name": "An opinion assigned to the quote."}}
       ]
     }}
  ]
}}
"""


scoring_system_instruction = """You are an expert analyst of online political discussion. You operate with a 'Maximum Inclusion' philosophy, assuming a context of robust, open, and sometimes heated democratic debate.

SCALING PRINCIPLES:
1. DYNAMIC RANGE: Use the full 0.0 - 1.0 spectrum. Avoid clumping scores.
2. CONTEXT AWARENESS: Distinguish between attacks on ideas (allowed) vs. attacks on people (penalized).
3. ANCHOR ALIGNMENT: Prioritize matching the specific attribute definitions provided."""

scoring_system_prompt_template = """{system_instruction}

Your task is to estimate the probability (0.0 to 1.0) that a group of human
annotators would agree the text belongs to the category: {label}.

Definition of {label}: {definition}
{additional_instr}
Calibrated Examples for {label}:
{calibrated_examples}

IMPORTANT: These examples are provided to clarify the boundaries of the
definition and calibrate your scoring. Do NOT overfit to the specific language,
subjects, or keywords used in these examples.

Respond ONLY with a valid JSON object: {{"score": <float>}}
"""


def get_eval_quote_extraction_criteria(survey_text_col: str, topic_col: str) -> dict[str, str]:
  return {
    "correctness": (
        "The quote accurately reflects the original meaning and sentiment from"
        f" the '{survey_text_col}'. It may be a single, exact"
        " substring or a combination of multiple substrings from the original"
        " text. Light edits are permissible to ensure coherence and"
        " readability, similar to journalistic quoting practices. These edits"
        " should not alter the original meaning, introduce new information, or"
        " include external commentary. Any additions or changes should be"
        " minimal and clearly serve to make the extracted quote flow naturally"
        " as a complete thought or statement."
    ),
    "conciseness": (
        "The quote extracts only the most essential words to convey the core"
        " idea or sentiment from "
        f"the '{survey_text_col}' relevant to the topic. It is"
        " as brief as possible without losing critical "
        + "meaning or context. Partial sentences are acceptable if they are the"
        " most concise way to "
        + "capture the point and remain coherent."
    ),
    "relevance": (
        "The quote is highly relevant to the specified"
        f" '{topic_col}'. It accurately represents the"
        f" '{survey_text_col}'s' main points, arguments, or"
        " sentiment concerning *that specific topic*. It captures a key and"
        f" insightful aspect of the '{survey_text_col}'"
        f" regarding the '{topic_col}'."
    ),
    "storytelling": (
        "The quote effectively conveys a personal experience, anecdote, or a "
        "thought-provoking opinion from the speaker. It should be engaging, "
        "insightful, or otherwise captivating, providing a glimpse into the "
        "individual's perspective or a specific event they describe."
    ),
}


eval_input_data_criteria = {
    "spamminess": (
        """
    You are evaluating a dialogue between an LLM pollster and a human respondent. Your task is to rate the human's response
    for non 'spamminess'.
    Rate the response low if it matches any of the following criteria.
    **Spam Criteria:**
    1.  **Irrelevant Self-Promotion:** Promoting one's own business or social media without relevance.
    2.  **Placeholder/Acknowledgement:** "ok", "got it", "I see".
    3.  **Meta-Commentary:** Commenting on the question or the poll itself (e.g., "that's a dumb question").
    4.  **Irrelevant Personal Content:** Sharing personal stories or information unrelated to the question.
    5.  **Promotional Content / Commercial Advertisement:** Advertising a product or service.
    6.  **Ambiguous/Non-Committal:** "maybe", "I guess", "I don't know".
    7.  **Phishing/Solicitation:** Asking for personal information or money.
    8.  **Vague/Insufficient Detail:** "It's good," "I like it."
    9.  **Gibberish:** Random characters or nonsensical words (e.g., "asdfghjkl").
    10. **Repetition:** Repeating phrases, sentences, words, characters, or patterns.
    11. **Generic Phrase Response:** Using clichés or stock phrases (e.g., "live, laugh, love").
    12. **Single-Word Response:** Answering with only one word.
    13. **Directly Copy-Pasted Content:** Pasting text from an external, non-AI source.
    14. **Irrelevant External Content:** Content that is off-topic.
    15. **Answer to Different Question:** Ignoring the question asked and answering a different one.
    16. **AI-Generated Content:** The response appears to be generated by another LLM.
    """
    ),
    "accuracy": (
        """
    You are evaluating a dialogue between an LLM pollster and a human respondent. Your task is to rate the human's response for 'accuracy'.
    Rate the response highly if the response answers the question directly. Rate it lower if the response to the question seems too vague
    or avoids answering the question all together.
    """
    ),
    "substantiveness": (
        """
    You are evaluating a dialogue between an LLM pollster and a human respondent. Your task is to rate the human's response for 'substantiveness'.
    Rate the response highly if the response offers some insight on the question or offers a (relevant) personal experience.
    """
    ),
}


eval_opinion_quality_criteria = {
    "correctness": (
        """
  You are evaluating an opinion that was identified based on quotes extracted from human responses in a survey.
  Criteria for a correct opinion:
  * **Distinct:** Within a single topic, opinions should represent unique viewpoints. Opinions should be meaningfully distinct and different within the topic.
  * **Substantive:** Opinions should be substantive. Do not create single-quote opinions.
  * **Accurate:** The text of opinion should be a clear and well-phrased summary of the underlying quotes.
  * **Concise:** The opinions should be concise.
  * **Topic and Subject Linkage:** Opinions should be phrased to have a clear link to the overarching topic and the main subject of the survey.
  * **Coherency:** Opinions should be logically consistent and easy to follow as you go over the list. E.g. for "Barriers to Freedom and Equality" topic, all the opinions could start with: "A key barrier to freedom and equality is ..."; for "Defining Equality" topic: "Equality is ..."; for "The Individual's Role in Freedom and Equality" topic: "Individuals achieve freedom and equality through ...", etc.
  * **Simplicity:** Phrase opinions in plain language. Aim for a 5th-grade reading level to ensure broad accessibility
  * **No "Perception" terminology**: Do not use words like "perception" in the opinion text. For example, instead of "A key barrier to freedom and equality is the perception that...", write "A key barrier to freedom and equality is...".
  * **Merge Overlaps:** If two or more opinions express the same fundamental idea, they **must be merged**.
  * **Efficiency:** Keep the number of opinions as low as possible. Actively consolidate opinions when their content can be logically grouped.
  VERY IMPORTANT:
  A single quote may contain information relevant to multiple opinions. When evaluating, consider only the part of the quote that is directly related to the current opinion.
  It does not need to capture other themes present in the text, as those will be evaluated separately under their own opinions.
  For example, if an opinion is about "freedom of expression," it does not need to cover other themes within the same text, such as "freedom of choice" or "freedom from fear."
    """
    ),
}


eval_other_opinion_criteria = {
    "non_compatibility": (
        """
You are evaluating a quote (quote) that has been categorized under the "Other" opinion for a given topic.

- **Topic**: {topic}
- **Existing Opinions**:
{existing_opinions}

"""
        + categorization_opinion_main_rules
        + """

Eval rules:
- A quote cannot be assigned to a definitional opinion (e.g. an opinion about what equality is) if the quote does not also explicitly provide a definition (e.g. "equality is...").
- Do not equate distinct concepts. For example, a quote about 'equal access to things' should not be matched to an opinion about 'being treated the same way'.
- Do not match based on an interpreted 'main thesis'. The match must be literal. For example, a quote calling equality 'not useful' or a 'superficial construct' does not explicitly match an opinion calling it an 'unrealistic idea'.
- Distinguish between a state of being and an action. A quote defining equality as a state where 'everything is the same for everyone' does not literally match an opinion defining equality as the action of 'treating every person in the exact same way'.
- Do not assume the context of well-known phrases. A quote mentioning "life, liberty, and the pursuit of happiness" does not automatically match an opinion about "America's historical ideals" unless the quote explicitly links them to America.

**Your Task:**
Based on the categorization rules provided above, evaluate if the `response` (quote) was correctly categorized as "Other".
- A quote is correctly categorized as "Other" if it is distinct and incompatible with ALL of the "Existing Opinions", according to the rules.
- A quote is incorrectly categorized as "Other" if it clearly aligns with AT LEAST ONE of the "Existing Opinions", according to the rules.
"""
    )
}


eval_opinion_categorization_correctness_criteria = {
    "correctness": (
        """
You are evaluating if a quote has been correctly assigned to one or more opinions.
The user's response will contain the list of opinions assigned to a single quote.

- **Quote**: {quote}
- **Topic**: {topic}
- **All Available Opinions for the Topic**:
{all_opinions}

"""
        + categorization_opinion_main_rules
        + """

Based on these rules, evaluate if each of the assigned opinions in the `response` is a correct match for the quote.
"""
    ),
    "minimality": (
        """
You are evaluating if a quote has been assigned the minimum necessary number of opinions.
The user's response will contain the list of opinions assigned to a single quote.

- **Quote**: {quote}
- **Topic**: {topic}
- **All Available Opinions for the Topic**:
{all_opinions}

Here are the rules for categorization:
- Keep the number of selections to the absolute minimum. Only select more than one opinion if they are both equally perfect, literal matches to the quote.

Your Task:
Evaluate if the quote is over-assigned to opinions in the `response`.
- Rate highly if each assigned opinion represents a distinct, necessary, and perfect literal match.
- Rate lower if fewer opinions could have been used to represent the quote's meaning accurately.
"""
    ),
}


eval_agreement_criteria = {
    "agreement": (
        """
  You are evaluating a response to a question. Your task is to determine if the response agrees or disagrees with the question.
- **Agrees:** The response supports the statement in the question.
- **Disagrees:** The response opposes the statement in the question.
- **Neutral/Unclear:** The response is neutral, unclear, or does not directly answer the question.

Score the response on a scale of 1-4, where 1 is strong disagreement, 4 is strong agreement, and 2-3 represents a neutral or unclear stance.
"""
    )
}


eval_proposition_topic_criteria = {
    "relevance": (
        """
  The proposition is highly relevant to the specified topic. It accurately represents a key aspect of the topic.
  """
    ),
    "clarity": (
        """
  The proposition is clear, concise, and easy to understand. It is written in plain language and avoids jargon.
  """
    ),
}


eval_proposition_opinion_criteria = {
    "representativeness": (
        """
  The proposition accurately represents the core sentiment of the opinion. It captures the main points and arguments of the opinion.
  """
    ),
    "conciseness": (
        """
  The proposition is as brief as possible without losing critical meaning or context.
  """
    ),
}


def get_autorater_opinion_eval_prompt(criteria_str: str, rubric_str: str, assigned_opinions_str: str) -> str:
    return f"""You are an expert evaluator.

Task: Evaluate if the opinion categorization is correct based on the provided criteria.

Criteria:
{criteria_str}

Rating Rubric:
{rubric_str}

Response to Evaluate:
{assigned_opinions_str}

RESPONSE STRUCTURE:
Respond with only these two fields: 'score' and 'explanation', nothing else.
Explanation should be as short as possible, and no more than once sentence.

The response must follow this format:
{{
  "score": 4,
  "explanation": "Brief reasoning for the score"
}}
"""

topic_modeling_merge_topics_prompt = """
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
You have been provided with multiple lists of topics that were generated by analyzing different chunks of the same dataset.
Your task is to synthesize and consolidate these lists by merging similar topics, and generate a final, deduplicated list of topics.

 ### **Consolidation Criteria**

 *   **Merge Duplicates:** Identify and merge topics that are semantically identical or highly similar.
 *   **Ensure Distinctness:** The final topics should be meaningfully different and cover separate conceptual areas.
 *   **Substantive:** Topics should contain multiple, distinct opinions. Avoid single-opinion topics.
 *   **Subject Linkage:** Topic names should be phrased as aspects of the main subject. E.g. "Freedom and Equality" subject could have the following topics: "Defining Freedom", "Defining Equality", "Barriers to Freedom and Equality", "Society's Role in Freedom and Equality", "The Individual's Role in Freedom and Equality", etc.
 *   **Maintain Consistency:** Ensure all topics in the final list are at a similar level of abstraction.
 *   **Efficiency:** Keep the number of topics as low as possible. Actively consolidate topics when their content can be logically grouped.
 *   **Concise:** The topic name should be concise.

RESPONSE STRUCTURE:
Respond with a single consolidated list of the identified topics only, nothing else.
The response should be in JSON format, that can be parse into the following class:
class FlatTopicList:
    topics: List[Topic]
class Topic:
    name: str

Do not include markdown code blocks around the JSON response, such as ```json or ```
Response example:
{"topics": [{"name": "Topic 1"}, {"name": "Topic 2"}]}
"""


def get_quote_extraction_prompt(text: str, context: str, topic: str) -> str:
  """Creates a prompt for extracting a quote from text."""
  context_block = (
      f"<additionalContext>\n  {context}\n</additionalContext>\n"
      if context
      else ""
  )
  return f"""{context_block}Extract the most representative quote that represents participant opinion on <topic>{topic}</topic> topic from the following text:
<text>{text}</text>

- You are a professional journalist quoting from a participant's responses in a transcript to create a coherent quotation that represents the participant's opinion on the given topic.
- Use best practices of professional journalists to achieve that. Specifically and **sparingly**, using brackets to enclose your modifications, you can lightly edit, correct misspelling, miscapitalizations or mispunctations, redact any profanity (e.g., replace a profane word with its first letter followed by dashes, like "[s---]"), or add clarifying information so that the quote is understandable even without seeing the original question. No change can be made to the response outside of brackets.
- You may also merge elements from across multiple questions for coherence. You must use ellipses to show when you are doing this, and you cannot modify the original sentence order.
- Other than the bracketed modifications, the quotation should be an ellipsis-delimited concatenation of substrings of the participant's response that obeys the original sentence order.
- We want to surface powerful and personal nuances a person shared on the opinion while keeping the quote concise and scannable. Stories about the participants' lives are especially valuable to feature. The quotation should feel punchy and profound, like an incisive portrait of the participant's humanity.
- The quote should only cover the given topic. We may extract several quotes from this transcript and must avoid redundancy.
- If there's not enough text for personal nuances, the quote should be just what the person expressed (e.g.: "I don't know"), and it's okay for it to be short.
- Do not add any extra commentary or markdown to the quote.
- Please output only the quotation. You should not enclose the quotation in quotation marks.
"""
