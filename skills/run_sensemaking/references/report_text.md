# Step 4: Generating Report Text (Discussion Summarization)

Generates hierarchical topic summaries and high-level conversation overviews based on the scored quotes.

## E2E Script Reference
This step corresponds to the command block under `# Step 4: Generating the Report Text` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
python3 -m src.generate_report_text.generate_report_text \
  --input_csv <WORK_DIR>/bridging_scores.csv \
  --output_dir <WORK_DIR>/report_outputs \
  --additional_context src/default-additional-context.md \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name <MODEL_NAME>
```

## Key Flags and Arguments
* `--input_csv`: Scored quotes CSV from Step 3.
* `--output_dir`: Target folder for the report text outputs.
* `--additional_context`: Additional instructions or markdown file describing guidelines for formatting.

## Expected Outputs
* `report_data.json`: Main file containing overview summaries and topic summaries.
* `report_data_with_opinions.json`: Expanded file with opinion-level summaries for debugging or detailed views.
