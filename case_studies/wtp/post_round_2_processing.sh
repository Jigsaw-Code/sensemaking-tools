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

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Usage: $0 --input_csv <path_to_input.csv> --output_dir <path_to_output_dir> --perspective_api_key <api_key>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RENAMED_COLS_CSV="$OUTPUT_DIR/renamed_cols.csv"
PROCESSED_CSV="$OUTPUT_DIR/processed.csv"
MODERATED_CSV="$OUTPUT_DIR/moderated.csv"

# Depending on the data this might not be needed.
python3 -m case_studies.wtp.qualtrics.rename_round2_cols \
  --input_csv "$INPUT_CSV" \
  --output_csv "$RENAMED_COLS_CSV"

# First process the data from Qualtrics
python3 -m case_studies.wtp.qualtrics.process_qualtrics_output \
    --input_csv "$RENAMED_COLS_CSV" \
    --output_csv "$PROCESSED_CSV" \
    --data_type "ROUND_2"

# TODO: Add evals on human inputs here and use that for moderation

# Then add Perspective scores for moderation.
python3 -m case_studies.wtp.moderation.prepare_for_moderation \
    --input_csv "$PROCESSED_CSV" \
    --output_csv "$MODERATED_CSV" \
    --text_column "response_text" \
    --data_type "ROUND_2" \
    --api_key "$PERSPECTIVE_API_KEY"

# Validate the processed csv.
python3 -m case_studies.wtp.propositions.input_csv_validation \
    --r2_input_file "$PROCESSED_CSV"
