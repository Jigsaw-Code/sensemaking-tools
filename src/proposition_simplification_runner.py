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
Runs simplification on a CSV of propositions.

Example input CSV (`input.csv`):
proposition
the cat sat on the mat
the dog chased the ball

Example output CSV (`output.csv`):
proposition,simplification
the cat sat on the mat,The cat was on the mat.
the dog chased the ball,The dog ran after the ball.


Sample command:
python -m src.proposition_simplification_runner \
    --input_csv src/proposition_simplification.csv \
    --output_csv src/proposition_simplification_output.csv \
    --model_name gemini-2.5-pro
"""

import argparse
import asyncio
import pandas as pd

from src.models import genai_model


DEFAULT_SIMPLIFICATION_PROMPT = """
You are an expert editor. Your task is to rewrite a list of propositions for maximum clarity and accessibility, without losing any of the original meaning.

An expert editor simplifies the language, not the idea. To do this, you will apply a ""lossless"" simplification method. This involves deconstructing the original sentence into its essential components and then rebuilding it with simpler words.

**Follow this professional process for each proposition:**

* **Original Proposition:** ""The principle that all people have equal worth is the moral foundation for a society where everyone is treated fairly under the law and has the same chance to succeed.""

* **Step 1: Deconstruct into Essential Components.** Break the sentence down to its core logical parts.
    1.  The core belief: all people have equal worth.
    2.  Its function: it's a moral foundation.
    3.  The outcome (part A): a society with fair legal treatment.
    4.  The outcome (part B): a society where everyone gets an equal chance.

* **Step 2: Reconstruct with Simple, Clear Language.** Write a new sentence between 15-20 words that incorporates a simple version of every component.
    * **Rewritten Proposition:** ""The belief that all people have equal worth is the foundation for fair laws and equal opportunities for everyone.""

* **Step 3: Final Check.** Verify that every component from Step 1 is present in the final version.
    1.  ""The belief that all people have equal worth"" (Covers Component 1)
    2.  ""...is the foundation for..."" (Covers Component 2)
    3.  ""...fair laws..."" (Covers Component 3)
    4.  ""...and equal opportunities for everyone."" (Covers Component 4)

Apply this exact process to the list below.
"""


async def generate_text_in_parallel(
    model: genai_model.GenaiModel, prompt_obj_list: list[dict]
) -> pd.DataFrame:
  response_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_obj_list, lambda x, _: x['text']
  )
  return response_df


def get_full_prompt(instructions, proposition):
  return (
      f"{instructions}\n\n"
      "Here is the proposition to rewrite:\n"
      f"<proposition>\n{proposition}\n</proposition>\n\n"
      "Provide only the rewritten proposition as your output. "
      "Do not include any other text or markdown."
  )


async def main(args):
  """Main function to run the proposition simplification experiment."""
  model = genai_model.GenaiModel(
      model_name=args.model_name,
      api_key=args.gemini_api_key,
  )

  df = pd.read_csv(args.input_csv)

  # For each proposition, run the simplification prompt
  prompt_objs = []
  for i, row in df.iterrows():
    prompt_objs.append({
      'prompt': get_full_prompt(args.prompt, row[args.proposition_column])
    })
  response_df = await generate_text_in_parallel(model, prompt_objs)

  # Copy simplification results from the response_df['result'] to a
  # "simplification" column, and write to disk.
  df['simplification'] = response_df['result']
  df.to_csv(args.output_csv, index=False)
  print(f"Successfully processed. Output written to {args.output_csv}")


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Simplify propositions from a CSV file."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--gemini_api_key",
      required=False,
      help=(
          "Google AI Studio API Key. If not provided, uses GOOGLE_API_KEY env"
          " var."
      ),
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default=DEFAULT_SIMPLIFICATION_PROMPT,
      help="Set the logging level (default: INFO).",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.5-pro",
      help="The name of the AI model to use. Default: gemini-2.5-pro.",
  )
  parser.add_argument(
      "--proposition_column",
      type=str,
      default="proposition",
      help="The name of the proposition column to use. Default: proposition.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
