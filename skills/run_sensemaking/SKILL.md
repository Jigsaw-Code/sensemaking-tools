---
name: run-sensemaking
description: Orchestrates the complete Sensemaking pipeline to process survey responses, discover topics, extract/score quotes, and generate propositions with simulated juries.
---

# Run Sensemaking Skill

This skill automates and guides the orchestration of the Sensemaking pipeline. The pipeline transforms raw participant responses into structured, categorized insights, scoring quote quality, and refining propositions using simulated juries.

## Behavior Guidelines for Interactive Run

When this skill is activated:

### 1. Pre-Run Checks & Verification
Do it every time before the whole pipeline.
* **Check Input CSV & API Keys:** Validate input file presence, schema, and API key environment variables (`GEMINI_API_KEY`, `GCLOUD_API_KEY`).
* **Decide Output Directory:** If user instructs you to use any ouptut directory, use it for output. Otherwise, propose a timestamped output directory (e.g. `outputs/run_YYYYMMDD_HHMMSS`) and ask the user for confirmation.
* **Determine Target Model:** Ask the user what model to use while giving default option `gemini-3.5-flash`.
* **Input Validation Script:** The validation tool is kept for input verification:
  ```bash
  python3 skills/run_sensemaking/scripts/validate_input.py --input_csv <PATH_TO_INPUT_CSV>
  ```

### 2. Step-by-Step Execution
For every step of the pipeline:
1. **Pre-Step Briefing:** Explain to the user what the step does and what inputs/outputs are involved.
2. **Execute Step:** Execute the command.
3. **Monitoring & Progress Updates:** For long-running steps, use the `schedule` tool to check logs and report progress without exposing agent internal states.
4. **Post-Step Inspection:** Validate output files exist and are not empty.
5. **Approval:** Send a summary brief and explicitly ask the user for confirmation before initiating the next step.

## Pipeline Overview & Step Index

The Sensemaking pipeline consists of 9 core steps. When instructed to run one single step, read the corresponding md file in references/ to know about detailed documentation, parameters, command script block, and how to run this step, and follow it to run the step. When running the pipeline end to end, start from step 1, do the above for each step until the last step,

1. **[Step 1: Survey Processing](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/survey_processing.md)**
   Formats raw Qualtrics CSVs into standard `participant_id` and `survey_text` columns.
2. **[Step 2: Topic Discovery & Quote Extraction](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/categorization.md)**
   Uses Gemini to identify discussion topics and extract specific, meaningful quotes.
3. **[Step 3: Bridging Scores](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/bridging.md)**
   Scores quotes on constructiveness metrics (reasoning, personal story, curiosity).
4. **[Step 4: Generating Report Text](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/report_text.md)**
   Generates recursive summaries for opinions, topics, and overviews.
5. **[Step 5: Interactive Report UI](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/report_ui.md)**
   Compiles and builds the web-based visual report dashboard.
6. **[Step 6: Proposition Generation](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/proposition_generation.md)**
   Extracts distinct statements/propositions from categorization outputs.
7. **[Step 7: Proposition Refinement](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/proposition_refinement.md)**
   Simulates a panel of jurists to vote and rank propositions.
8. **[Step 8: Exporting Final Results](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/export_results.md)**
   Queries refined world model to output CSV files.
9. **[Step 9: Simplifying Nuanced Propositions](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/skills/run_sensemaking/references/proposition_simplification.md)**
   Rewrites nuanced propositions for maximum clarity.