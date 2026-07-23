# Step 2: Categorization and Quote Extraction

Uses Gemini to identify discussion topics and opinions, extracting representative quotes from survey responses.

## E2E Script Reference
This step corresponds to the command block under `# Step 2: Categorization and Quote Extraction` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
python3 -m src.categorization_runner \
  --input_file <WORK_DIR>/processed.csv \
  --output_dir <WORK_DIR>/categorization_outputs \
  --gemini_api_key "$GEMINI_API_KEY" \
  --additional_context_file src/default-additional-context.md \
  --model_name <MODEL_NAME>
```

## Key Flags and Arguments
* `--input_file`: Standardized CSV output from Step 1 (containing `participant_id` and `survey_text`).
* `--output_dir`: Directory for outputs and log files.
* `--topics`: (Optional) Comma-separated list of predefined topics instead of letting the model discover them.
* `--skip_autoraters`: (Optional) Skips the self-evaluation check to save API costs/time.
* `--skip_quote_extraction`: (Optional) Skips quote extraction and uses the full survey response instead.
* `--model_name`: Gemini model to use (defaults to `gemini-2.5-pro`).

## Expected Outputs
* `categorized_with_other.csv`: Complete output with topic and opinion assignments.
* `categorized_without_other_filtered.csv`: Cleaned output excluding fallback "Other" category and containing only fields required for downstream steps.
* `categorized_with_other_topic_tree.txt`: Readable text tree showing topic and opinion structure.
