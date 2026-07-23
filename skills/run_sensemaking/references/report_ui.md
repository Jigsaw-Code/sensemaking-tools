# Step 5: Interactive Report UI

Deploys and compiles the web UI visualization of the categorized and summarized conversation data.

## E2E Script Reference
This step corresponds to the command block under `# Step 5: Interactive Report UI` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh) where output files are automatically copied to UI input folders.

## Setup Instructions
1. Run the E2E script or execute the copy commands:
   ```bash
   mkdir -p src/report_ui/input
   cp <WORK_DIR>/bridging_scores.csv src/report_ui/input/opinions.csv
   cp <WORK_DIR>/report_outputs/report_data.json src/report_ui/input/summary.json
   ```
2. Navigate to the UI directory and install dependencies:
   ```bash
   cd src/report_ui
   npm install
   ```

## Serving & Building
* **Local development server:** `npm run dev`
* **Static web server deployment:** `npm run build` (Outputs to `src/report_ui/output/static`)
* **Single inline HTML build:** `npm run inline` (Outputs self-contained page to `src/report_ui/output/inline`)
