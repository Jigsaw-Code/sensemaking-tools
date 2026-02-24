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

# Post processing script for Round 1 data. This is expected to be in the format
# of 8 questions per survey respondent, with half being LLM generated follow up
# questions.
#
# Sample command:
# bash case_studies/wtp/post_round_1_processing.sh \
#      --input_csv ~/Downloads/immigration.csv \
#      --output_dir case_studies/wtp/output \
#      --api_key 1234abc
#      --one_line_question_text \
#      --round_1_question_response_text "Q3.1,Q5.1,Q7.1,Q9.3" \
#      --round_1_follow_up_questions "Q1FU,Q2FU,Q3FU" \
#      --round_1_follow_up_question_response_text "Q4.1,Q6.1,Q8.1"
#
# What this outputs:
# 1. processed.csv: the qualtrics survey with incomplete rows removed, and two
#   new columns: "survey_text" column with all the questions and answers
#   combined, and a "response_text" column with only the human responses.
# 2. evals/ directory: this contains the summary_metrics.csv for the overall
#   input data quality as well as a metrics.csv file which is every survey
#   response and it's quality rating with an explanation.
# 3. moderated.csv: the survey responses with the Perspective scores and data
#   quality score.

# Fail on any error
set -e

INPUT_CSV=""
OUTPUT_DIR=""
API_KEY=""

# Optional overrides
ROUND_1_QUESTION_RESPONSE_TEXT=""
ROUND_1_FOLLOW_UP_QUESTIONS=""
ROUND_1_FOLLOW_UP_QUESTIONS=""
ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS=""
ONE_LINE_QUESTION_TEXT_ARG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_csv) INPUT_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        --round_1_question_response_text) ROUND_1_QUESTION_RESPONSE_TEXT="$2"; shift ;;
        --round_1_follow_up_questions) ROUND_1_FOLLOW_UP_QUESTIONS="$2"; shift ;;
        --round_1_follow_up_question_response_text) ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS="$2"; shift ;;
        --one_line_question_text) ONE_LINE_QUESTION_TEXT_ARG="--one_line_question_text"; ;;
        --no-one_line_question_text) ONE_LINE_QUESTION_TEXT_ARG="--no-one_line_question_text"; ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$API_KEY" ]; then
    echo "Usage: $0 --input_csv <path_to_input.csv> --output_dir <path_to_output_dir> --perspective_api_key <api_key> [optional overrides]"
    exit 1
fi

PROCESSED_CSV="$OUTPUT_DIR/processed.csv"
MODERATED_CSV="$OUTPUT_DIR/moderated.csv"
EVALS_DIR="$OUTPUT_DIR/evals"

mkdir -p "$EVALS_DIR"

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
python3 -m case_studies.wtp.qualtrics.process_qualtrics_output \
    --input_csv "$INPUT_CSV" \
    --output_csv "$PROCESSED_CSV" \
    --data_type "ROUND_1" $OPTIONAL_ARGS

# Then use autoraters to score how good the data is
python3 -m case_studies.wtp.evals.evals \
    --baseline_csv "$PROCESSED_CSV" \
    --output_dir "$EVALS_DIR" \
    --project conversation-ai-experiments \
    --location us-central1 \
    --model_name "gemini-2.5-pro" \
    --metric_name input_evals

# Then add Perspective scores for moderation. Use the input data evaluation
# output as the input data so the output has both the autorater results and the
# Perspective scores.
python3 -m case_studies.wtp.moderation.prepare_for_moderation \
    --input_csv "$PROCESSED_CSV" \
    --input_evals_csv "$EVALS_DIR/metrics.csv" \
    --output_csv "$MODERATED_CSV" \
    --text_column "survey_text" \
    --data_type "ROUND_1" \
    --api_key "$API_KEY"

