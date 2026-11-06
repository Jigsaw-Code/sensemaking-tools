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

# Runs each step of the Sensemaking pipeline, intended as an end-to-end test
# Example command:
#   bash src/test_sensemaking_e2e.sh "" "" src/smoke_test_data.csv /tmp/sensemaking_e2e_output [gemma3:270m]

set -e  # Exit on any error

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <GEMINI_API_KEY> <GCLOUD_API_KEY> <QUALTRICS_CSV> <WORKING_DIR>"
    echo "Optional: [MODEL_NAME (default: gemma3:270m)] [FIXED_RESPONSE_COLS] [FOLLOW_UP_QUESTION_COLS] [FOLLOW_UP_RESPONSE_COLS]"
    exit 1
fi

# Assign command line args
GEMINI_API_KEY=$1
GCLOUD_API_KEY=$2
QUALTRICS_CSV=$3
WORKING_DIR=$4
MODEL_NAME=${5:-"gemma3:270m"}

# Args for Qualtrics survey processing
FIXED_RESPONSE_COLS=${6:-"Q1,Q35,Q36"}
FOLLOW_UP_QUESTION_COLS=${7:-"Q1FU,Q2FU,Q3FU"}
FOLLOW_UP_RESPONSE_COLS=${8:-"Q23,Q37,Q39"}

# Create working directory if it doesn't exist.
mkdir -p "$WORKING_DIR"

# Utility function to print each step
print_step() {
  local prefix="Starting Step: "
  printf "\n%s%s\n" "${prefix}" "${1}"
}

OLLAMA_SERVER_PID=""
cleanup() {
  if [[ "$MODEL_NAME" == gemma* ]]; then
    printf "\nTearing down local Gemma server resources...\n"
    if command -v ollama >/dev/null 2>&1; then
      ollama stop "$MODEL_NAME" >/dev/null 2>&1 || true
    fi
    if [[ -n "$OLLAMA_SERVER_PID" ]] && kill -0 "$OLLAMA_SERVER_PID" 2>/dev/null; then
      kill "$OLLAMA_SERVER_PID" 2>/dev/null || true
      wait "$OLLAMA_SERVER_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

if [[ "$MODEL_NAME" == gemma* ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Error: 'ollama' is required to run local Gemma models ($MODEL_NAME), but was not found in PATH." >&2
    exit 1
  fi

  # Force CPU execution for the local Ollama server.
  export OLLAMA_NUM_GPU=0
  export OLLAMA_LLM_LIBRARY="cpu"
  export CUDA_VISIBLE_DEVICES=""
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

  if ollama list >/dev/null 2>&1; then
    print_step "Reusing running local Ollama server at $OLLAMA_HOST for $MODEL_NAME"
  else
    print_step "Starting local Ollama server on CPU ($OLLAMA_HOST) for $MODEL_NAME"
    ollama serve > "$WORKING_DIR/ollama_server.log" 2>&1 &
    OLLAMA_SERVER_PID=$!

    # Wait for the Ollama server to become ready.
    SERVER_READY=0
    for _ in {1..30}; do
      if ollama list >/dev/null 2>&1; then
        SERVER_READY=1
        break
      fi
      sleep 1
    done

    if [[ "$SERVER_READY" -ne 1 ]]; then
      echo "Error: Local Ollama server failed to start within 30 seconds. Check $WORKING_DIR/ollama_server.log" >&2
      exit 1
    fi
  fi

  # Ensure the requested Gemma model is available locally.
  if ! ollama list | awk -v model="$MODEL_NAME" 'NR>1 && ($1 == model || $1 == model ":latest") {found=1} END {exit !found}'; then
    print_step "Pulling model $MODEL_NAME"
    ollama pull "$MODEL_NAME"
  fi

  export MODEL_ENDPOINT_TYPE="openai_api_compatible"
  export OPENAI_API_ENDPOINT_URL="http://${OLLAMA_HOST}/v1"
  export OPENAI_API_ENDPOINT_KEY="${GEMINI_API_KEY:-ollama}"
  export OPENAI_RESPONSE_FORMAT_STYLE="openai_json_schema"
  export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-1024}"
  export OPENAI_FREQUENCY_PENALTY="${OPENAI_FREQUENCY_PENALTY:-0.5}"
  export MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-1500}"
  export MAX_ITEMS_PER_BATCH="${MAX_ITEMS_PER_BATCH:-2}"
fi

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
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME" \
  --log_level DEBUG

print_step "Score Quotes with Bridging Classifiers"
python3 -m src.get_bridging_scores \
  --input_csv "$WORKING_DIR/categorization_outputs/categorized_without_other_filtered.csv" \
  --output_csv "$WORKING_DIR/bridging_scores.csv" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

print_step "Generating the Report Text"
python3 -m src.generate_report_text.generate_report_text \
  --input_csv "$WORKING_DIR/bridging_scores.csv" \
  --additional_context_file src/default-additional-context.md \
  --output_dir "$WORKING_DIR/report_outputs" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

# TODO: consider generating report HTML at this point

print_step "Creating Propositions"
python3 -m src.propositions.proposition_generator \
  --r1_input_file "$WORKING_DIR/categorization_outputs/categorized_without_other_filtered.csv" \
  --output_dir "$WORKING_DIR/proposition_outputs" \
  --reasoning \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

print_step "Proposition Refinement"
# Use --jury_size 0.02 to run as fast as possible.
python3 -m src.proposition_refinement.main \
  --input_pkl "$WORKING_DIR/proposition_outputs/world_model.pkl" \
  --output_pkl "$WORKING_DIR/proposition_outputs/refined_world_model.pkl" \
  --final_propositions_per_topic 4 \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$GEMINI_API_KEY" \
  --run_pav_selection \
  --simulated_jury_model_name "$MODEL_NAME" \
  --nuanced_propositions_model_name "$MODEL_NAME" \
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
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

printf "\n\nSensemaking pipeline completed successfully!\n"
echo "Please review the following files:"
echo "  $WORKING_DIR/categorization_outputs/categorized_with_other_filtered.csv"
echo "  $WORKING_DIR/categorization_outputs/categorized_with_other_topic_tree.txt"
echo "  $WORKING_DIR/final_propositions_by_topic.csv"
echo "  $WORKING_DIR/final_nuanced_propositions_simplified.csv"
