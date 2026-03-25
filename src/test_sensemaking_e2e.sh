# Copyright 2026 Google LLC
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

# Runs each step of theSensemaking pipeline, intended as an end-to-end test
# Example command:
#   bash src/test_sensemaking_e2e.sh <API_KEY> <API_KEY> <QUALTRICS_CSV> <WORKING_DIR> gemini-2.5-flash

set -e  # Exit on any error

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <API_KEY> <QUALTRICS_CSV> <WORKING_DIR>"
    echo "Optional: [MODEL_NAME] [FIXED_RESPONSE_COLS] [FOLLOW_UP_QUESTION_COLS] [FOLLOW_UP_RESPONSE_COLS]"
    exit 1
fi

# Assign command line args
API_KEY=$1
QUALTRICS_CSV=$2
WORKING_DIR=$3
MODEL_NAME=${4:-"gemini-2.5-flash"}

# Args for Qualtrics survey processing
FIXED_RESPONSE_COLS=${5:-"Q1,Q35,Q36"}
FOLLOW_UP_QUESTION_COLS=${6:-"Q1FU,Q2FU,Q3FU"}
FOLLOW_UP_RESPONSE_COLS=${7:-"Q23,Q37,Q39"}

# Create working directory if it doesn't exist.
mkdir -p "$WORKING_DIR"

# export Gemini key used by some tasks.
export GOOGLE_API_KEY="$API_KEY"

# Utility function to print each step
print_step() {
  local prefix="Starting Step: "
  printf "\n%s%s\n" "${prefix}" "${1}"
}

print_step "Processing the Survey Results"
bash src/survey_processing.sh \
  --input_csv "$QUALTRICS_CSV" \
  --output_dir "$WORKING_DIR/survey_processing_ouput" \
  --round_1_question_response_text "$FIXED_RESPONSE_COLS" \
  --round_1_follow_up_questions "$FOLLOW_UP_QUESTION_COLS" \
  --round_1_follow_up_question_response_text "$FOLLOW_UP_RESPONSE_COLS"

print_step "Categorization and quote extraction"
# Set --skip_autoraters to run faster
python3 -m src.categorization_runner \
  --additional_context_file src/default-additional-context.md \
  --input_file "$WORKING_DIR/survey_processing_ouput/processed.csv" \
  --output_dir "$WORKING_DIR/categorization_outputs" \
  --skip_autoraters \
  --model_name "$MODEL_NAME" \
  --log_level DEBUG

print_step "Score Quotes with Bridging Classifiers"
python3 -m src.get_bridging_scores \
  --input_csv "$WORKING_DIR/categorization_outputs/categorized_without_other_filtered.csv" \
  --output_csv "$WORKING_DIR/bridging_scores.csv" \
  --api_key "$API_KEY" \
  --model_name "$MODEL_NAME"

print_step "Generating the Report Text"
python3 -m src.generate_report_text.generate_report_text \
  --input_csv "$WORKING_DIR/bridging_scores.csv" \
  --additional_context ./default-additional-context.md \
  --output_dir "$WORKING_DIR/report_outputs" \
  --model_name "$MODEL_NAME"

# TODO: consider generating report HTML at this point

print_step "Creating Propositions"
python3 -m src.propositions.proposition_generator \
  --r1_input_file "$WORKING_DIR/categorization_outputs/categorized_without_other_filtered.csv" \
  --output_dir "$WORKING_DIR/proposition_outputs" \
  --reasoning \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$API_KEY" \
  --model_name "$MODEL_NAME"

print_step "Proposition Refinement"
# Use --jury_size 0.02 to run as fast as possible.
python3 -m src.proposition_refinement.main \
  --input_pkl "$WORKING_DIR/proposition_outputs/world_model.pkl" \
  --output_pkl "$WORKING_DIR/proposition_outputs/refined_world_model.pkl" \
  --final_propositions_per_topic 4 \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$API_KEY" \
  --run_pav_selection \
  --jury_size 0.02

print_step "Extract final propositions"
python3 -m src.world_model.main --query=all_by_topic --output_format=csv \
  "$WORKING_DIR/proposition_outputs/refined_world_model.pkl" > "$WORKING_DIR/final_propositions_by_topic.csv"

python3 -m src.world_model.main --query=all_nuanced --output_format=csv \
  "$WORKING_DIR/proposition_outputs/refined_world_model.pkl" > "$WORKING_DIR/final_nuanced_propositions.csv"

print_step "Simplifying nuanced propositions"
python3 -m src.proposition_simplification_runner \
  --input_csv "$WORKING_DIR/final_nuanced_propositions.csv" \
  --output_csv "$WORKING_DIR/final_nuanced_propositions_simplified.csv" \
  --gemini_api_key "$API_KEY" \
  --model_name "$MODEL_NAME"

printf "\n\nSensemaking pipeline completed successfully!\n"
echo "Please review the following files:"
echo "  $WORKING_DIR/categorization_outputs/categorized_with_other_filtered.csv"
echo "  $WORKING_DIR/categorization_outputs/categorized_with_other_topic_tree.txt"
echo "  $WORKING_DIR/final_propositions_by_topic.csv"
echo "  $WORKING_DIR/final_nuanced_propositions_simplified.csv"
