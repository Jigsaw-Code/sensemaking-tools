#!/usr/bin/env bash

# Error handling settings
set -euo pipefail

# This script downloads all of the export files associated with the given
# polis report id (i.e. https://pol.is/report/<report-id>).
#
# Usage: `./bin/download_polis_data.sh <report-id> <output-dir>`

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
ROOT_DIR="$(readlink -f "${SCRIPT_DIR}/..")"

if [[ -z "$1" ]]; then
  echo "Error: Please provide a report ID argument"
  exit 1
fi
REPORT_ID="$1"
EXPORT_URL_BASE="https://pol.is/api/v3/reportExport/${REPORT_ID}"

if [[ -z "$2" ]]; then
  echo "Error: Please provide an output directory argument"
  exit 1
fi
OUTPUT_DIR="$2"
# Make sure the directory exists:
mkdir -p "${OUTPUT_DIR}"

echo "Downloading data for report: $REPORT_ID"
echo "Outputing data to: ${OUTPUT_DIR}"

# Download the data files
curl "${EXPORT_URL_BASE}/comments.csv" > "${OUTPUT_DIR}/comments.csv"
curl "${EXPORT_URL_BASE}/participant-votes.csv" > "${OUTPUT_DIR}/participants-votes.csv"
curl "${EXPORT_URL_BASE}/votes.csv" > "${OUTPUT_DIR}/votes.csv"
curl "${EXPORT_URL_BASE}/summary.csv" > "${OUTPUT_DIR}/summary.csv"
curl "${EXPORT_URL_BASE}/comment-groups.csv" > "${OUTPUT_DIR}/comment-groups.csv"
