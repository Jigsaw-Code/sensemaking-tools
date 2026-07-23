# Step 1: Survey Processing (Qualtrics Parsing)

Formats raw Qualtrics exports into a standardized format containing `participant_id` and `survey_text`.

## E2E Script Reference
This step corresponds to the command block under `# Step 1: Survey Processing (Qualtrics parsing)` in [src/test_sensemaking_e2e.sh](file:///google/cog/cloud/weiduan/sensemaking_skills/participation-project-internal/src/test_sensemaking_e2e.sh).

## CLI Command
```bash
bash src/survey_processing.sh \
  --input_csv <RAW_SURVEY_CSV> \
  --output_dir <OUTPUT_DIR> \
  --round_1_question_response_text <FIXED_COLS> \
  --round_1_follow_up_questions <FU_COLS> \
  --round_1_follow_up_question_response_text <FU_RESP_COLS>
```

## Arguments
* `--input_csv`: Path to the raw export from Qualtrics.
* `--output_dir`: Directory where the parsed CSV should be stored.
* `--round_1_question_response_text`: Comma-separated names of columns containing fixed questions.
* `--round_1_follow_up_questions`: Comma-separated names of columns containing the follow-up question text prompts.
* `--round_1_follow_up_question_response_text`: Comma-separated names of columns containing answers to the follow-up questions.

## Expected Outputs
* `<OUTPUT_DIR>/processed.csv`: Contains cleaned, parsed responses with `participant_id` and `survey_text`.
