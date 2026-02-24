"""
Library for combining propositions from different topics to find common ground.
"""

import os
import re
from case_studies.wtp.models import genai_model


def generate_combination_prompt(
    propositions_by_topic: dict[str, list[str]],
    num_combinations: int = 30,
    additional_context: str | None = None,
) -> str:
  """Generates a prompt for combining propositions from different topics."""
  prompt = f"""You are an expert in political science and public deliberation. Your task is to synthesize new, concise, and compelling propositions that reflect points common ground between different perspectives.

Below are a series of propositions, grouped by topic, that emerged from an AI faciliated discussion amongst a large group of participants. These propositions have been identified as being particularly likely to serve as points of common ground emerging from the discussion.

**Propositions by Topic:**
"""
  for topic, propositions in propositions_by_topic.items():
    prompt += f"\n**Topic: {topic}**\n"
    for i, proposition in enumerate(propositions, 1):
      prompt += f"{i}. {proposition}\n"

  if additional_context:
    prompt += (
        f"\n<additionalContext>\n  {additional_context}\n</additionalContext>\n"
    )

  prompt += f"""\n**Your Task:**\n
Generate {num_combinations} new propositions that combine ideas from different topics in ways that are likely to result in interesting, and importantly NEW, points of common ground. The goal is to create statements that are greater than the sum of their parts and more likely to find broad agreement than any single idea from the simple topics above.

**Instructions:**

1.  **Identify Tradeoffs and Synergies:** Look for opportunities to create "win-win" scenarios or to find a middle ground between competing values. For example, how can the desire for "freedom from government" be reconciled with the need for "physical safety and security"? How can "economic opportunity" be balanced with "responsibility and respect for others"?
2.  **Synthesize, Don't Just Concatenate:** Do not simply join two propositions together. Instead, synthesize the underlying ideas into a new, coherent statement.
3.  **Maintain a Declarative Style:** The new propositions should be written as concise, declarative statements. They should sound like plausible conclusions from a large-scale deliberative process.
4.  **Frame as Fact:** Write each proposition as a factual statement, not as a belief or opinion (e.g., say 'X is a priority' rather than 'We believe X should be a priority').
5.  **Capture the Collective View:** The statements should reflect the collective perspective of the people, grounded in the provided topics.
6.  **Focus on Common Ground:** The new propositions should be framed in a way that is likely to appeal to a broad range of people, even those with different starting points.
7.  **Keep it Simple and Clear:** The propositions should be easy to understand for a general audience and expressedly plainly in simple and easy to understand terms, avoiding running on with jargon or overly complex language, while still reflecting nuanced and thoughtful perspectives.
8.  *Novel:** Make sure the proposition are NEW. Each one should be distinct from the individual simple propositions by topic, and be addative to the collection of comments as a whole.

**Provide your answer as a numbered list of {num_combinations} new propositions.**
"""
  return prompt


def parse_nuanced_propositions(resp: dict, job: dict) -> list[str]:
  """Parses the LLM response into a list of propositions."""
  response_text = resp["text"]
  if not response_text:
    return []
  # Split by newline and filter out empty strings
  propositions = [
      line.strip() for line in response_text.splitlines() if line.strip()
  ]
  # Remove the numbering from the beginning of the propositions
  propositions = [re.sub(r"^\s*\d+\.\s*", "", prop) for prop in propositions]
  return propositions


async def combine_propositions(
    propositions_by_topic: dict[str, list[str]],
    model: genai_model.GenaiModel,
    additional_context: str | None = None,
):
  """Combines propositions from different topics to find common ground."""
  prompt = generate_combination_prompt(
      propositions_by_topic, additional_context=additional_context
  )

  return await model.process_prompts_concurrently(
      [{"prompt": prompt, "topic": "na"}],
      response_parser=parse_nuanced_propositions,
      max_concurrent_calls=3,
  )
