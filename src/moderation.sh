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

# Post processing script to create eval and moderation CSV files.
#
# Sample command:
# bash src/moderation.sh \
#      --processed_csv ~/Downloads/processed.csv \
#      --output_dir src/output \
#      --api_key 1234abc
#
# What this outputs:
# 1. evals/ directory: this contains the summary_metrics.csv for the overall
#   input data quality as well as a metrics.csv file which is every survey
#   response and it's quality rating with an explanation.
# 2. moderated.csv: the survey responses with the Gemini/Perspective scores and data
#   quality score.

# Fail on any error
set -e

PROCESSED_CSV=""
OUTPUT_DIR=""
API_KEY=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --processed_csv) PROCESSED_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROCESSED_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$API_KEY" ]; then
    echo "Usage: $0 --processed_csv <path_to_input.csv> --output_dir <path_to_output_dir> --api_key <api_key> [optional overrides]"
    exit 1
fi

MODERATED_CSV="$OUTPUT_DIR/moderated.csv"
EVALS_DIR="$OUTPUT_DIR/evals"

mkdir -p "$EVALS_DIR"

# Then use autoraters to score how good the data is
python3 -m src.evals.evals \
    --baseline_csv "$PROCESSED_CSV" \
    --output_dir "$EVALS_DIR" \
    --model_name "gemini-2.5-pro" \
    --metric_name input_evals \
    --api_key "$API_KEY"

# Then add Gemini/Perspective scores for moderation. Use the input data
# evaluation output as the input data so the output has both the autorater
# results and the Gemini/Perspective scores.
python3 -m src.moderation.prepare_for_moderation \
    --input_csv "$PROCESSED_CSV" \
    --input_evals_csv "$EVALS_DIR/metrics.csv" \
    --output_csv "$MODERATED_CSV" \
    --text_column "survey_text" \
    --data_type "ROUND_1" \
    --api_key "$API_KEY"

