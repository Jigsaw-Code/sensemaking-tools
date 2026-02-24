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

R1_INPUT_CSV=""
R2_INPUT_CSV=""
OUTPUT_DIR="case_studies/wtp/testdata"
OUTPUT_FILE_NAME="wtp_s1"
PROPOSITION_COUNT="5"
API_KEY=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --r1_input_file) R1_INPUT_CSV="$2"; shift ;;
        --r2_input_file) R2_INPUT_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        --proposition_count) PROPOSITION_COUNT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$R1_INPUT_CSV" ]; then
    echo "Missing required argument: $0 --r1_input_file <path_to_input.csv>"
    exit 1
fi
if [ -z "$R2_INPUT_CSV" ]; then
    echo "Missing required argument: $0 --r2_input_file <path_to_input.csv>"
    exit 1
fi
if [ -z "$API_KEY" ]; then
    echo "Missing required argument: $0 --api_key <api_key>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -m case_studies.wtp.propositions.proposition_generator \
--prop_count="$PROPOSITION_COUNT" \
--r1_input_file="$R1_INPUT_CSV" \
--r2_input_file="$R2_INPUT_CSV" \
--output_dir="$OUTPUT_DIR" \
--gemini_api_key="$API_KEY" \
--output_file_name="$OUTPUT_FILE_NAME"
