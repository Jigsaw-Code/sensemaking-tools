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

# Module for automated evaluation of summarization results using LLMs.

# Command to run hallucination evals:
# python evals/autorating/run_autoraters.py -p <your-gcp-project> -l <gcp-project-location> -m <vertex-model-name> -i <path-to-your-input-csv> -o <output-directory>

# Examples:

# 1. Specifying all flags:
# python evals/autorating/run_autoraters.py -p your-project-id -l europe-west4 -m gemini-2.0-flash-001 -i evals/summary.csv -o evals/hallucination_results

# 2. Using all default flag values:
# python evals/autorating/run_autoraters.py -p your-project-id -i evals/summary.csv

# Example of input data:
# +--------------------+----------------------+
# | summary            | source               | <- "summary" and "source" columns are required
# +--------------------+----------------------+
# | A summary claim... | Source statements... |
# +--------------------+----------------------+

import argparse
import asyncio
import logging

from autorating_utils import read_csv
from hallucination_autorater import HallucinationAutorater
from models.vertex_model import VertexModel


async def main():

  logging.basicConfig(
      level=logging.INFO,
      format="%(message)s",  # print log messages only without any extra info
  )

  parser = argparse.ArgumentParser(
      description="Run automated evaluation of summarization results."
  )
  parser.add_argument(
      "-p", "--gcpProject", required=True, help="GCP project name for Vertex AI"
  )
  parser.add_argument(
      "-i",
      "--inputFile",
      required=True,
      help="CSV file with summary claims and source statements",
  )
  parser.add_argument(
      "-o",
      "--outputDir",
      default="evals/autorating/results",
      help="Where to save evaluation results to",
  )
  parser.add_argument(
      "-l",
      "--location",
      default="global",
      help="In which location to run the model",
  )
  parser.add_argument(
      "-m",
      "--model",
      default="gemini-2.5-pro",
      help="Vertex AI model name",
  )
  parser.add_argument(
      "-c",
      "--additionalContext",
      default="",
      help="Additional context to provide to the model",
  )
  args = parser.parse_args()

  model = VertexModel(args.gcpProject, args.location, args.model)
  autorater = HallucinationAutorater(model, args.outputDir)
  summaries = read_csv(args.inputFile)

  await autorater.rate_hallucination(summaries, args.additionalContext)


if __name__ == "__main__":
  asyncio.run(main())
