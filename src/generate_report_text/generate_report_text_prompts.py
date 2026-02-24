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
Prompts used by generate_report_text.py
"""


def get_overview_prompt(additional_context, topic_summaries):
  joined_topic_summaries = '\n  '.join(['<topic>' + ts + '</topic>' for ts in topic_summaries.values()])
  return f"""
<instructions>
  **Goal**
  Your task is to write the narrative body of an Executive Summary.

  **Context**
  This text appears immediately following a statistical header.

  **Hard Constraints**
  * **Length:** Maximum 175 words total.
  * **Format:** 2 cohesive paragraphs.

  **Stylistic Guidelines (Civil & Concise)**
  **Hew closely and sensitively to the context and goals outlined in the additionalContext element.**
  1.  **Synthesize, Don't List:** Do not try to include every single nuance. Group similar ideas into broad themes.
      * *Bad:* "Some said X, others said Y, and a few said Z."
      * *Good:* "Perspectives ranged from strict adherence to tradition to calls for systemic reform."
  2.  **Tone:** Civil and analytical. Frame differences as a "spectrum of thought," not a conflict.
  3.  **Density:** Use high-value words to save space.
      * *Instead of:* "...emphasized the critical need for systemic change and tailored support..."
      * *Use:* "...emphasized the need for systemic reform..."
  4.  **Avoid Abstract Subjects:** Do not let "perspectives" or "views" be the subject of the sentence. Use "Participants," "Critics," "Advocates," or "Groups" to make the writing feel human and active.
  5.  **Connect the Dots:** Ensure the second paragraph (definitions) logically flows from the first (history). Show how the worldview influences the definition.

  **Input Data**
  Here are the summaries of the specific topics:
  {joined_topic_summaries}
</instructions>

<additionalContext>
  {additional_context}
</additionalContext>

<output_format>
  Output **only** the 2 paragraphs of text.
</output_format>
"""


def get_opinion_summary_prompt(
    topic: str, opinion: str, additional_context: str,
    quotes: list[str], opinions_per_topic: dict[str]):
  joined_quotes = '\n  '.join(['<comment>' + q + '</comment>' for q in quotes])
  return f"""
<instructions>
  **Task**
  Please write a comprehensive summary of the quotes about the opinion "{opinion}" in the topic "{topic}".

  **Summary context**
  Your summary will be inform a subsequent summary of the entire topic "{topic}", which has the following opinions: {opinions_per_topic[topic]}
  The topic summaries will appear altogther in a report. All topics to be summarized are: {list(opinions_per_topic.keys())}.
  Your summary will appear alongside the opinion name, so you do not need to restate or rephrase the opinion in your summary.
  Please refrain from summarizing other topics and opinions to avoid redundancy across summaries.
  This summary should focus on the opinion "{opinion}". If unrelated ideas are raised they should be omitted.
  This summary will be used as context for the LLM when summarizing all opinions in the topic "{topic}". Please use as much context as needed to inform an LLM about the opinion.

  Additionally, do not output any markdown.

</instructions>

<additionalContext>
  {additional_context}
</additionalContext>

<data>
  {joined_quotes}
</data>"""


def get_topic_summary_prompt(
    topic: str, additional_context: str, opinion_summaries_for_topic, opinion_sizes_for_topic, all_topics):
  opinion_summary_xml = ''
  for opinion, summary in opinion_summaries_for_topic.items():
    opinion_summary_xml += f"""
  <opinionSummary>
    <title>{opinion}</title>
    <text>{summary}</text>
    <size>{opinion_sizes_for_topic[opinion]}</size>
  </opinionSummary>
"""

  return f"""
<instructions>
  **Task**
  Your job is to compose an executive summary about the following topic: {topic}.

  **Summary context**
  Your summary will be included in a report on the results of a survey or discussion.
  Your summary will appear alongside the topic name, so you do not need to restate the topic.
  Your summary will be based on already-composed summaries of differing participant opinions about the topic.

  The report will contain multiple summaries like this one, each corresponding to a different topic that came up in the survey.

  All topics to be summarized are: {all_topics}.
  Please refrain from summarizing other topics to avoid redundancy across summaries.

  **Tone**
  You have been given 100 words or fewer. Hew closely and sensitively to the context and goal as described in the additionalContext field.
  **Cut the Fluff:** Remove all "reporting about the reporting."
  * *Bad:* "Discussions around defining equality revealed several distinct viewpoints."
  * *Good:* "Definitions of equality varied wildly."
  **Use Plain English:** Ban multi-syllable corporate words where simple ones will do.
  * *Change* "Necessitates" $\\rightarrow$ "Requires"
  * *Change* "Instillation" $\\rightarrow$ "Teaching"
  * *Change* "Paramount" $\\rightarrow$ "Key"
  **Vary Sentence Length:** Use short, punchy sentences to break up long ideas. Don't be afraid of a simple subject-verb-object structure.
  **Active Voice:** Make the *people* or the *ideas* the subject, not the "viewpoint."
  **Ban the "See-Saw" Structure:** Strictly avoid the repetitive pattern of "Some said X, while others said Y. Another group said Z."
  **Paragraph Structure:** Lead with the synthesis and "lightbulb" insight, foreground shared themes (you have been given the opinion sizes, measured in number of component quotes, to assist with this), and highlight points of disagreement or uncertainty that might require further research.
  **Use a Neutral Point of View :** The Neutral Point of View is defined as representing fairly, proportionately (you have been given the opinion sizes, measured in number of component quotes, to assist with this), and, as far as possible, without editorial bias, all the significant views in the conversation. Your prose should reflect that you are representing what participants said, not declaring actual points of fact. Do not make overly authoritative or generalizing statements (e.g. Do not say "The opportunity is…"). Do not use the first person or claim opinions as your own.
</instructions>

<additionalContext>
  {additional_context}
</additionalContext>

<data>
{opinion_summary_xml}
</data>
"""