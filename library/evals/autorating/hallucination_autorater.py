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

# Module for automated evaluation of hallucination using LLMs.

import logging
import os
import time
from typing import List
import pandas as pd
from autorating_utils import (
    EvalResults,
    EvalInput,
    generate_evaluation_report,
    format_comments,
    format_summary,
)
from vertex_model import VertexModel, run_tasks_in_parallel


class HallucinationAutorater:
    def __init__(self, model: VertexModel, output_dir: str):
        self.model = model
        self.output_dir = output_dir

    async def rate_hallucination(self, summaries: List[EvalInput], context: str = ""):
        """
        Evaluates the hallucination/fabrication tendency of generated summary statements.

        Args:
            summaries: A list of `EvalInput` objects.
            context: optional additional context to provide to the model
        """
        start_time_total = time.perf_counter()

        # Initialize DataFrame with the header row
        autorating_results_df = pd.DataFrame(
            columns=[
                "Generated Statement",
                "Source Comments",
                "Has Hallucinations?",
                "LLM Analysis",
                "LLM Explanation",
                "Runtime (seconds)",
            ]
        )

        aggregated_results: EvalResults = {
            "totalSummaries": 0,
            "metrics": {"Hallucinations": {"no": 0, "yes": 0, "maybe": 0}},
        }

        prompts = []
        for summary_data in summaries:
            statement = summary_data["summary"]
            comments = summary_data["source"]

            formatted_summary = format_summary(statement)

            # TODO: pass comments context so we don't need to ask model to ignore claims like 'broad support', 'common ground' and potentially miss some hallucinations
            formatted_comments = format_comments(comments)

            prompt = f"""
You are analyzing a statement that attempts to summarize a comment or a set of comments.

STATEMENT:
{formatted_summary}

INPUT COMMENTS:
{formatted_comments}

ADDITIONAL CONTEXT:
{context}

INSTRUCTIONS:
Step 1. Statement Breakdown and Evidence Mapping:
a. Break down the statement into individual units of information.  Each distinct topic, claim, or assertion should be considered a separate unit.
b. For each unit of information, determine if it is mentioned in any of the provided INPUT COMMENTS.
 * If the unit of information is supported by one or more comments, list the number(s) of the supporting comment(s) in square brackets after the unit. For example:  "improving educational opportunities[2]" (if supported by comment 2).  If multiple comments support the unit, list all of them: "environment[1,3]".
 * If the unit of information is supported by the additional context, mark it with "context" in square brackets. For example:  "conversation in Kentucky[context]"
 * If the unit of information is NOT mentioned in *any* of the comments, mark it with an "X" in square brackets to indicate hallucination.  For example:  "supporting local businesses[X]"
 * If a summary statement contains the following claims: 'disagreement among participants', 'broad support', 'common ground' or 'A participant stated', don't flag it as fabrication as those claims were added on purpose and are not a part of evaluation.
 * Having valid acronyms is fine, do not flag acronyms if they are present, but the full name is not mentioned in the input. And vice versa.
c. Present the complete statement with the bracketed evidence tags. Example: "High consensus was reached on topics such as preserving green spaces[1], supporting local businesses[X], and improving educational opportunities[2]."

Step 2. Answer the following question with "YES", "NO" or "MAYBE", followed by a *brief* explanation of the reasoning behind why this answer was given:
- Does the statement contain fabricated information *not* mentioned in the comments?  (YES indicates hallucination/fabrication).

RESPONSE STRUCTURE:
Respond with your analysis, followed by the "YES", "NO" or "MAYBE" answers to the questions, and a brief explanation for each answer.
The response should be in JSON format.
Do not include markdown code blocks around the JSON response, such as ```json or ```
For example:
{{"analysis": "...", "answer": "NO", "explanation": "NO because..."}}
"""
            prompts.append((prompt, statement, comments))

        # inner function to evaluate a single prompt
        async def evaluate(prompt_data):
            prompt, statement, comments = prompt_data
            start_time_statement = time.perf_counter()

            try:
                response = await self.model.generate_data(prompt)

            except Exception as e:
                logging.error(f"Error during LLM call or parsing: {e}")
                return {
                    "statement": statement,
                    "comments": comments,
                    "has_hallucinations": "NULL",
                    "analysis": "NULL",
                    "explanation": "NULL",
                    "runtime": "NULL",
                }

            if not response:
                logging.warning("Skipping due to invalid response from LLM.")
                return {
                    "statement": statement,
                    "comments": comments,
                    "has_hallucinations": "NULL",
                    "analysis": "NULL",
                    "explanation": "NULL",
                    "runtime": "NULL",
                }

            statement_runtime_sec = time.perf_counter() - start_time_statement
            logging.info(f"STATEMENT hallucination check took {statement_runtime_sec} seconds.")

            return {
                "statement": statement,
                "comments": comments,
                "has_hallucinations": response["answer"],
                "analysis": response["analysis"],
                "explanation": response["explanation"],
                "runtime": f"{statement_runtime_sec:.2f}",
            }

        results = await run_tasks_in_parallel(prompts, evaluate)

        for result in results:
            # add to dataframe
            new_row = pd.DataFrame(
                [
                    [
                        result["statement"],
                        result["comments"],
                        result["has_hallucinations"],
                        result["analysis"],
                        result["explanation"],
                        result["runtime"],
                    ]
                ],
                columns=autorating_results_df.columns,
            )
            autorating_results_df = pd.concat(
                [autorating_results_df, new_row], ignore_index=True
            )

            # Update aggregated results
            has_hallucinations = result["has_hallucinations"]
            if has_hallucinations == "YES":
                aggregated_results["metrics"]["Hallucinations"]["yes"] += 1
            elif has_hallucinations == "NO":
                aggregated_results["metrics"]["Hallucinations"]["no"] += 1
            elif has_hallucinations == "MAYBE":
                aggregated_results["metrics"]["Hallucinations"]["maybe"] += 1

            aggregated_results["totalSummaries"] += 1

        # Save autorating results to a CSV file
        output_file_path = os.path.join(
            self.output_dir, "hallucination_autoratings.csv"
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create directory if needed
        try:
            autorating_results_df.to_csv(output_file_path, index=False)  # Save DataFrame to CSV
            logging.info(f"CSV data saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Error writing CSV data to file: {e}")

        # Generate a report
        total_runtime_min = (time.perf_counter() - start_time_total) / 60
        report = generate_evaluation_report(aggregated_results, total_runtime_min)
        logging.info(report)
        report_file_path = os.path.join(
            self.output_dir, "hallucination_report.txt"
        )
        try:
            with open(report_file_path, "w", encoding="utf-8") as report_file:
                report_file.write(report)
            logging.info(f"Report saved to {report_file_path}")
        except Exception as e:
            logging.error(f"Error writing report to file: {e}")
