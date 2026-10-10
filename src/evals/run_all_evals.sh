#!/bin/bash
# ==============================================================================
# SENSEMAKING EVALUATION SCRIPT
# ==============================================================================
# This script automates the categorization pipeline and its subsequent evaluations.
# It takes a raw processed survey CSV, runs it through the topic categorization
# runner (using Gemini), and then sequentially feeds the outputs into the various
# LLM-as-a-judge evaluation metrics defined in src/evals/evals.py.
#
# USAGE:
#   ./src/evals/run_all_evals.sh \
#     --gemini_api_key <YOUR_GEMINI_API_KEY> \
#     [--processed_csv <path/to/processed.csv>] \
#     [--categorization_model <model_name>] \
#     [--output_base_dir <optional/path/to/output_dir>] \
#     [--categorization_output_dir <optional/path/to/cat_dir>]
#
# EXAMPLE:
#   ./src/evals/run_all_evals.sh \
#     --processed_csv my_data.csv \
#     --gemini_api_key $GEMINI_API_KEY \
#     --categorization_model gemini-3.8-flash
# ==============================================================================

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --processed_csv) PROCESSED_CSV="$2"; shift ;;
    --gemini_api_key) GEMINI_API_KEY="$2"; shift ;;
    --categorization_model) CATEGORIZATION_MODEL="$2"; shift ;;
    --output_base_dir) OUTPUT_BASE_DIR="$2"; shift ;;
    --categorization_output_dir) CATEGORIZATION_OUTPUT_DIR="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# User must specify either --processed_csv (to run categorization_runner with)
# or --categorization_output_dir (to reuse existing categorization)
if { [ -z "$PROCESSED_CSV" ] && [ -z "$CATEGORIZATION_OUTPUT_DIR" ]; } || \
  [ -z "$GEMINI_API_KEY" ]; then
  echo "Usage: $0 [--processed_csv <path.csv>] --gemini_api_key <key>" \
    "[--categorization_model <model>] [--output_base_dir <dir>]" \
    "[--categorization_output_dir <dir>]"
  echo "Example: $0 --processed_csv my_processed_data.csv" \
    "--gemini_api_key YOUR_API_KEY --categorization_model gemini-3.8-flash"
  exit 1
fi

CATEGORIZATION_MODEL="${CATEGORIZATION_MODEL:-gemini-3.8-flash}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./all_evals_output}"

mkdir -p "$OUTPUT_BASE_DIR"

# Create metadata.txt file to track relevant commands.
METADATA_FILE="${OUTPUT_BASE_DIR}/metadata.txt"
cat <<EOF > "$METADATA_FILE"
processed_csv=${PROCESSED_CSV}
categorization_model=${CATEGORIZATION_MODEL}
categorization_output_dir=${CATEGORIZATION_OUTPUT_DIR}
EOF

# Categorize if needed
if [ -n "$CATEGORIZATION_OUTPUT_DIR" ]; then
  echo "========================================================"
  echo "Step 1: Skipping Categorization Runner" \
    "(using $CATEGORIZATION_OUTPUT_DIR)"
  echo "========================================================"
  CAT_DIR="$CATEGORIZATION_OUTPUT_DIR"
else
  echo "========================================================"
  echo "Step 1: Running Categorization Runner (Model: $CATEGORIZATION_MODEL)"
  echo "========================================================"
  CAT_DIR="${OUTPUT_BASE_DIR}/categorization"
  mkdir -p "$CAT_DIR"

  python3 -m src.categorization_runner \
    --input_file "$PROCESSED_CSV" \
    --output_dir "$CAT_DIR" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --model_name "$CATEGORIZATION_MODEL" \
    --additional_context_file src/default-additional-context.md \
    --skip_autoraters
fi

BASELINE_CSV="${CAT_DIR}/categorized_with_other_filtered.csv"

if [ ! -f "$BASELINE_CSV" ]; then
  echo "Error: Categorization runner failed to produce $BASELINE_CSV." \
    "Check the logs."
  exit 1
fi

echo ""
echo "========================================================"
echo "Step 2: Running Evaluations using $BASELINE_CSV"
echo "========================================================"

# These metrics both work with categorized_with_other_filtered.csv output
# from categorization_runner.
# TODO: add all metrics
METRICS=(
  "opinion_quality"
  "opinion_categorization"
)

# Run each metric using src.evals.evals
for metric in "${METRICS[@]}"; do
  echo "--------------------------------------------------------"
  echo "Running eval for metric: $metric"
  echo "--------------------------------------------------------"

  OUTPUT_DIR="${OUTPUT_BASE_DIR}/evals/${metric}"
  mkdir -p "$OUTPUT_DIR"
  rm -f "${OUTPUT_DIR}/summary_metrics.csv"

  if python3 -m src.evals.evals \
    --baseline_csv "$BASELINE_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --metric_name "$metric" \
    --gemini_api_key "$GEMINI_API_KEY"; then
    echo "Finished $metric."
  else
    echo "========================================================"
    echo "ERROR: Eval run FAILED for metric: $metric"
    echo "========================================================"
  fi
  echo ""
done

echo ""
echo "========================================================"
echo "Step 3: Consolidating metrics"
echo "========================================================"

# Create a CSV with columns metric_name,mean_score,count
# This is created using the data from each summary_metrics.csv file.
# Any failed eval task will be logged.
CONSOLIDATED_CSV="${OUTPUT_BASE_DIR}/consolidated_summary_metrics.csv"
echo "metric_name,mean_score,count" > "$CONSOLIDATED_CSV"
FAILED_METRICS=()
for metric in "${METRICS[@]}"; do
  summary_file="${OUTPUT_BASE_DIR}/evals/${metric}/summary_metrics.csv"
  if [ -s "$summary_file" ] && [ "$(wc -l < "$summary_file")" -ge 2 ]; then
    tail -n +2 "$summary_file" | while IFS= read -r line; do
      if [ -n "$line" ]; then
        echo "${metric},${line}" >> "$CONSOLIDATED_CSV"
      fi
    done
  else
    echo "WARNING: No valid results for ${metric} - marking as FAILED."
    echo "${metric},FAILED,FAILED" >> "$CONSOLIDATED_CSV"
    FAILED_METRICS+=("$metric")
  fi
done

# Print the consolidated metrics csv as a table.
CSV_DATA=$(cat "$CONSOLIDATED_CSV")
echo "$CSV_DATA" | column -s, -t
echo ""

# Log any failed metrics
if [ "${#FAILED_METRICS[@]}" -gt 0 ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "WARNING: ${#FAILED_METRICS[@]} eval metric(s) FAILED:"
  for failed_metric in "${FAILED_METRICS[@]}"; do
    echo "  - ${failed_metric}"
  done
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo ""
fi

echo "Pipeline complete! Consolidated metrics saved to $CONSOLIDATED_CSV"
