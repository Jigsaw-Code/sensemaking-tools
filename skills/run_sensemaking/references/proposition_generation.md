# Step 6: Proposition Generation

Generates a broad set of statements (propositions) reflecting the viewpoints in the community feedback.

## E2E Script Reference
This step corresponds to the command block under `# Step 6: Creating Propositions` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
python3 -m src.propositions.proposition_generator \
  --r1_input_file <WORK_DIR>/categorization_outputs/categorized_without_other_filtered.csv \
  --output_dir <WORK_DIR>/proposition_outputs \
  --reasoning \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name <MODEL_NAME>
```

## Key Flags and Arguments
* `--r1_input_file`: Filtered categorized CSV from Step 2.
* `--output_dir`: Output path for proposition structures.
* `--reasoning`: Include model reasoning in outputs.
* `--prop_count`: (Optional) How many propositions to generate per opinion.

## Expected Outputs
* `<WORK_DIR>/proposition_outputs/world_model.pkl`: A serialized pandas DataFrame containing the generated propositions along with supporting metadata.
