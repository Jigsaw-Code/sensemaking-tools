# Step 8: Exporting Final Results

Extracts the ranked simple and nuanced propositions from the refined world model.

## E2E Script Reference
This step corresponds to the command block under `# Step 8: Extract final propositions` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
# Get all ranked propositions by topic
python3 -m src.world_model.main --query=all_by_topic --output_format=csv \
  <WORK_DIR>/proposition_outputs/refined_world_model.pkl > <WORK_DIR>/final_propositions_by_topic.csv

# Get all ranked nuanced propositions
python3 -m src.world_model.main --query=all_nuanced --output_format=csv \
  <WORK_DIR>/proposition_outputs/refined_world_model.pkl > <WORK_DIR>/final_nuanced_propositions.csv
```

## Key Flags and Arguments
* `--query`: Either `all_by_topic` or `all_nuanced`.
* `--output_format`: Format type (typically `csv`).

## Expected Outputs
* `<WORK_DIR>/final_propositions_by_topic.csv`: CSV of ranked propositions grouped by topic.
* `<WORK_DIR>/final_nuanced_propositions.csv`: CSV of ranked nuanced propositions.
