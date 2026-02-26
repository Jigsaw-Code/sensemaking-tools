#!/usr/bin/env bash

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

DEFAULT_VERTEX_PARALLELISM=100
NUM_RUNS=50


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
# Set a default output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="eval_output_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inputFile=*) INPUT_FILE="${1#*=}" ;;
    --outputDir=*) OUTPUT_DIR="${1#*=}" ;;
    --vertexProject=*) VERTEX_PROJECT="${1#*=}" ;;
    --vertexParallelism=*) DEFAULT_VERTEX_PARALLELISM="${1#*=}" ;;
    --additionalContext=*) ADDITIONAL_CONTEXT="${1#*=}" ;;
    --numRuns=*) NUM_RUNS="${1#*=}" ;;
    --help)
      echo "Usage: $0 --inputFile=<input_file> --outputDir=<output_dir> --vertexProject=<vertex_project> --vertexParallelism=<vertex_parallelism> --additionalContext=<additional_context> [--numRuns=<num_runs>] [--help]"
      exit 0
      ;;
    *)
      echo "Error: Invalid option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

# Check if required options are provided
if [[ -z "${INPUT_FILE}" || -z "${OUTPUT_DIR}" || -z "${VERTEX_PROJECT}" || -z "${DEFAULT_VERTEX_PARALLELISM}" || -z "${ADDITIONAL_CONTEXT}" ]]; then
  echo "Error: Missing required options." >&2
  echo "Usage: $0 --inputFile=<input_file> --outputDir=<output_dir> --vertexProject=<vertex_project> --vertexParallelism=<vertex_parallelism> --additionalContext=<additional_context> [--numRuns=<num_runs>] [--help]" >&2
  exit 1
fi

if [ "$(ls -A ${OUTPUT_DIR} | wc -l)" -ne 0 ]; then
  echo "Expected output directory to be empty, exiting..."
  exit 1
fi

# Determine the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Construct absolute paths to the other files
RUNNER_DIR="${SCRIPT_DIR}/../../runner-cli"

run_categorization() {
  local run_number=$1
  local output_file="${OUTPUT_DIR}/run-${run_number}.csv"

  echo "Running categorization for run: ${run_number}"
  npx ts-node "${RUNNER_DIR}/categorization_runner.ts" \
    --topicDepth 1 \
    --outputFile "${output_file}" \
    --vertexProject "${VERTEX_PROJECT}" \
    --inputFile "${INPUT_FILE}" \
    --additionalContext "${ADDITIONAL_CONTEXT}" \
    --forceRerun
}

export DEFAULT_VERTEX_PARALLELISM=${DEFAULT_VERTEX_PARALLELISM};
for ((i = 1; i <= NUM_RUNS; i++)); do
  run_categorization "$i"
done

python3 "${SCRIPT_DIR}/run_evals.py" \
  --input-data  ${OUTPUT_DIR}/run-*csv \
  --output-csv-path "${OUTPUT_DIR}/evals.csv"
