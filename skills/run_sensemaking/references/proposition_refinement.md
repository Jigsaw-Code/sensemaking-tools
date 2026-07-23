# Step 7: Proposition Refinement (Simulated Jury)

Ranks and refines the generated propositions using a simulated jury of participant profiles, selecting a high-consensus set.

## E2E Script Reference
This step corresponds to the command block under `# Step 7: Proposition Refinement (Simulated Jury)` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
python3 -m src.proposition_refinement.main \
  --input_pkl <WORK_DIR>/proposition_outputs/world_model.pkl \
  --output_pkl <WORK_DIR>/proposition_outputs/refined_world_model.pkl \
  --final_propositions_per_topic 4 \
  --additional_context_file src/default-additional-context.md \
  --gemini_api_key "$GEMINI_API_KEY" \
  --run_pav_selection \
  --simulated_jury_model_name <MODEL_NAME> \
  --nuanced_propositions_model_name <MODEL_NAME> \
  --jury_size 0.02
```

## Key Flags and Arguments
* `--input_pkl`: Pickled world model from Step 6.
* `--output_pkl`: Target path to save the refined world model.
* `--run_pav_selection`: Enables Proportional Approval Voting (PAV) selection to ensure fair representation.
* `--jury_size`: Fraction of the participant pool to use (e.g. `0.02` for fast testing, `1.0` for full run).

## Expected Outputs
* `<WORK_DIR>/proposition_outputs/refined_world_model.pkl`: Updated world model containing refined rankings and nuanced statements.
