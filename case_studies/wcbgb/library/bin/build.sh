#!/usr/bin/env bash

# Error handling settings
set -eo pipefail

# This script orchestrates application of the sensemaking tools to the data in
# the given directory. By default, sensemaking outputs are placed in the same
# directory as the the input data. An optional second argument will allow for
# output to go to another directory.
#
# Usage: ./bin/build.sh -i <input-dir> [-o <output-dir> -t <topics-list> -c <additional-context> -p <vertex-project>]

# compute absolute path to this script's directory
SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
ROOT_DIR="$(readlink -f "${SCRIPT_DIR}/..")"
RUNNER_DIR="$(readlink -f "${ROOT_DIR}/runner-cli")"

# Add defaults for when these flags are not set
TOPICS_FLAG=() # Initialize as an empty array
ADDITIONAL_CONTEXT_FLAG=() # Initialize as an empty array
VERTEX_PROJECT="conversation-ai-experiments"

# Parse command line arguments using getopts
while getopts ":i:o:t:c:p:" opt; do
  case "${opt}" in
    i)
      INPUT_DIR="${OPTARG}"
      ;;
    o)
      OUTPUT_DIR="${OPTARG}"
      ;;
    t)
      TOPICS_FLAG+=("--topics" "${OPTARG}") # Add to the array
      ;;
    c)
      ADDITIONAL_CONTEXT_FLAG+=("--additionalContext" "${OPTARG}") # Add to the array
      ;;
    p)
      VERTEX_PROJECT="${OPTARG}"
      ;;
    \?)
      echo "Invalid option: -${OPTARG}" >&2
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      exit 1
      ;;
  esac
done

# Raise an error if no input directory is specified
if [[ -z "${INPUT_DIR}" ]]; then
  echo "Error: Please provide an input directory with -i."
  exit 1
fi

# Default the output directory to the input directory if not specified
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${INPUT_DIR}"
else
  # Make sure the output dir actually exists
  mkdir -p "${OUTPUT_DIR}"
fi
echo "Outputing data to: ${OUTPUT_DIR}"


# Run preprocessing script:
commentsData="${OUTPUT_DIR}/comments.processed.csv"
echo "Building ${commentsData}"
"${SCRIPT_DIR}/process_polis_data.py" "${INPUT_DIR}" -o "${commentsData}"
echo ""

# Run categorization and summarization scripts at the subtopic level

subtopicCategorizedData="${OUTPUT_DIR}/comments.subtopic-categorized.csv"
echo "Building ${subtopicCategorizedData}"
# Build the command as an array
declare -a subtopic_categorization_command=(
  npx
  ts-node
  "${RUNNER_DIR}/categorization_runner.ts"
  --outputFile "${subtopicCategorizedData}"
  --vertexProject "${VERTEX_PROJECT}"
  --inputFile "${commentsData}"
  "${TOPICS_FLAG[@]}" # Expand the array here
  "${ADDITIONAL_CONTEXT_FLAG[@]}" # Expand the array here
)
echo "${subtopic_categorization_command[@]}"
"${subtopic_categorization_command[@]}" # Execute the command
echo ""

subtopicSummaryOutput="${OUTPUT_DIR}/subtopic-level"
echo "Building ${subtopicSummaryOutput}"
declare -a subtopic_summary_command=(
  npx
  ts-node
  "${RUNNER_DIR}/runner.ts"
  --outputBasename "${subtopicSummaryOutput}"
  --vertexProject "${VERTEX_PROJECT}"
  --inputFile "${subtopicCategorizedData}"
  "${ADDITIONAL_CONTEXT_FLAG[@]}"
)
echo "${subtopic_summary_command[@]}"
DEBUG_MODE=true "${subtopic_summary_command[@]}"
echo ""

# Run categorization and summarization scripts at the topic level

topicCategorizedData="${OUTPUT_DIR}/comments.topic-categorized.csv"
echo "Building ${topicCategorizedData}"
declare -a topic_categorization_command=(
  npx
  ts-node
  "${RUNNER_DIR}/categorization_runner.ts"
  --outputFile "${topicCategorizedData}"
  --vertexProject "${VERTEX_PROJECT}"
  --inputFile "${commentsData}"
  --skip-subtopics
  "${TOPICS_FLAG[@]}"
  "${ADDITIONAL_CONTEXT_FLAG[@]}"
)
echo "${topic_categorization_command[@]}"
"${topic_categorization_command[@]}"
echo ""

topicSummaryOutput="${OUTPUT_DIR}/topic-level"
echo "Building ${topicSummaryOutput}"
declare -a topic_summary_command=(
  npx
  ts-node
  "${RUNNER_DIR}/runner.ts"
  --outputBasename "${topicSummaryOutput}"
  --vertexProject "${VERTEX_PROJECT}"
  --inputFile "${topicCategorizedData}"
  "${ADDITIONAL_CONTEXT_FLAG[@]}"
)
echo "${topic_summary_command[@]}"
DEBUG_MODE=true "${topic_summary_command[@]}"
echo ""
