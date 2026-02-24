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

# Pre processing of data for Round 2 survey. These tasks include
# categorization and quote extraction. This script also runs evals on these
# outputs.

INPUT_CSV=""
OUTPUT_DIR=""
PERSPECTIVE_API_KEY=""


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_csv) INPUT_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --perspective_api_key) PERSPECTIVE_API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --input_csv <path_to_input.csv> --output_dir <path_to_output_dir>"
    exit 1
fi

EVALS_DIR="$OUTPUT_DIR/evals"
mkdir -p "$EVALS_DIR"

# TODO: Add categorization when ready.

QUOTE_EVALS_DIR="$EVALS_DIR/quote_extraction"
mkdir -p "$QUOTE_EVALS_DIR"
python3 -m case_studies.wtp.evals \
    --baseline_csv "$INPUT_CSV" \
    --output_dir "$QUOTE_EVALS_DIR" \
    --project conversation-ai-experiments \
    --location us-central1 \
    --model_name gemini-2.5-pro \
    --metric_name quote_extraction

# Reformat the data for clustering evaluations.
PROCESSED_CSV_PATH="${EVALS_DIR}/processed_for_evals.csv"
# Renames 'topic' to 'topics' and 'survey_text' to 'comment_text'
awk 'BEGIN{FS=OFS=","} NR==1 {for(i=1;i<=NF;i++) {if($i=="topic") $i="topics"; if($i=="survey_text") $i="comment_text"}} 1' \
  "$INPUT_CSV" > "$PROCESSED_CSV_PATH"
python3 "library/evals/clustering_evals/run_evals.py" \
  --input-data "$PROCESSED_CSV_PATH" \
  --output-csv-path "${EVALS_DIR}/topic_clustering_evals.csv"

# Reformat the data for opinion quality evaluations.
OPINION_EVALS_DIR="$EVALS_DIR/opinion_quality"
mkdir -p "$OPINION_EVALS_DIR"
python3 -m case_studies.wtp.evals \
    --baseline_csv "$INPUT_CSV" \
    --output_dir "$OPINION_EVALS_DIR" \
    --project conversation-ai-experiments \
    --location us-central1 \
    --model_name gemini-2.5-pro \
    --metric_name opinion_quality

mkdir -p "$OUTPUT_DIR/quotes"
export PERSPECTIVE_API_KEY
python3 case_studies.wtp.select_quotes \
    --input_csv "$INPUT_CSV" \
    --api_key "$PERSPECTIVE_API_KEY"
