# Step 3: Bridging Scores

Scores the extracted quotes on constructiveness attributes such as curiosity, personal story, and reasoning.

## E2E Script Reference
This step corresponds to the command block under `# Step 3: Score Quotes with Bridging Classifiers` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
python3 -m src.get_bridging_scores \
  --input_csv <WORK_DIR>/categorization_outputs/categorized_without_other_filtered.csv \
  --output_csv <WORK_DIR>/bridging_scores.csv \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name <MODEL_NAME>
```

## Key Flags and Arguments
* `--input_csv`: Path to the filtered categorized CSV from Step 2.
* `--output_csv`: Target output file path.
* `--scorer_type`: Either `GEMINI` (default) or `PERSPECTIVE`.
* `--model_name`: Model name to use when `scorer_type` is `GEMINI`.

## Expected Outputs
* `<WORK_DIR>/bridging_scores.csv`: The input CSV with additional score columns (`CURIOSITY_EXPERIMENTAL`, `PERSONAL_STORY_EXPERIMENTAL`, `REASONING_EXPERIMENTAL`, and `AVERAGE_OF_3_BRIDGING`).
