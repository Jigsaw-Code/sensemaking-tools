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

# Post processing script for Qualtrics data. This is expected to be in the format
# of 8 questions per survey respondent, with half being LLM generated follow up
# questions.
#
# Sample command:
# bash src/post_round_1_processing.sh \
#      --input_csv ~/Downloads/immigration.csv \
#      --output_dir src/output \
#      --one_line_question_text \
#      --round_1_question_response_text "Q3.1,Q5.1,Q7.1,Q9.3" \
#      --round_1_follow_up_questions "Q1FU,Q2FU,Q3FU" \
#      --round_1_follow_up_question_response_text "Q4.1,Q6.1,Q8.1"
#
# This outputs processed.csv: the qualtrics survey with incomplete rows removed,
# and two new columns: "survey_text" column with all the questions and answers
# combined, and a "response_text" column with only the human responses.

# Fail on any error
set -e

INPUT_CSV=""
OUTPUT_DIR=""
GEMINI_API_KEY=""
TRANSLATION_MODEL_NAME="gemini-3.1-flash-lite-preview"

# Optional overrides
ROUND_1_QUESTION_RESPONSE_TEXT=""
ROUND_1_FOLLOW_UP_QUESTIONS=""
ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS=""
ONE_LINE_QUESTION_TEXT_ARG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_csv) INPUT_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --gemini_api_key) GEMINI_API_KEY="$2"; shift ;;
        --translation_model_name) TRANSLATION_MODEL_NAME="$2"; shift ;;
        --round_1_question_response_text) ROUND_1_QUESTION_RESPONSE_TEXT="$2"; shift ;;
        --round_1_follow_up_questions) ROUND_1_FOLLOW_UP_QUESTIONS="$2"; shift ;;
        --round_1_follow_up_question_response_text) ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS="$2"; shift ;;
        --one_line_question_text) ONE_LINE_QUESTION_TEXT_ARG="--one_line_question_text"; ;;
        --no-one_line_question_text) ONE_LINE_QUESTION_TEXT_ARG="--no-one_line_question_text"; ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --input_csv <path_to_input.csv> --output_dir <path_to_output_dir> [optional overrides]"
    exit 1
fi

# Construct optional arguments
OPTIONAL_ARGS=""
if [ -n "$ROUND_1_QUESTION_RESPONSE_TEXT" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --round_1_question_response_text $ROUND_1_QUESTION_RESPONSE_TEXT"
fi
if [ -n "$ROUND_1_FOLLOW_UP_QUESTIONS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --round_1_follow_up_questions $ROUND_1_FOLLOW_UP_QUESTIONS"
fi
if [ -n "$ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --round_1_follow_up_question_response_text $ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS"
fi
if [ -n "$ONE_LINE_QUESTION_TEXT_ARG" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS $ONE_LINE_QUESTION_TEXT_ARG"
fi

# First process the data from Qualtrics
# shellcheck disable=SC2086
python3 -m src.qualtrics.process_qualtrics_output \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_DIR/processed.csv" \
    --data_type "ROUND_1" $OPTIONAL_ARGS

# Then translate the non-English responses in-place if Gemini API key is provided
if [ -n "$GEMINI_API_KEY" ]; then
    echo "Translating responses using $TRANSLATION_MODEL_NAME..."

    # Determine columns to translate
    COLUMNS_TO_TRANSLATE="Q1,Q2,Q3,Q1FU_Text,Q2FU_Text,Q3FU_Text"
    if [ -n "$ROUND_1_QUESTION_RESPONSE_TEXT" ] || [ -n "$ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS" ]; then
        COLUMNS_TO_TRANSLATE="$ROUND_1_QUESTION_RESPONSE_TEXT,$ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS"
    fi

    # We read and write to the same file (processed.csv)
    python3 -m src.translate_csv \
        --input_csv "$OUTPUT_DIR/processed.csv" \
        --output_csv "$OUTPUT_DIR/processed.csv" \
        --columns "$COLUMNS_TO_TRANSLATE" \
        --gemini_api_key "$GEMINI_API_KEY" \
        --model_name "$TRANSLATION_MODEL_NAME"
fi