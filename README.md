# Sensemaking by Jigsaw \- A Google AI Proof of Concept

This repository shares tools developed by [Jigsaw](http://jigsaw.google) as a proof of concept to help make sense of large-scale online conversations. It demonstrates how Large Language Models (LLMs) like Gemini can be leveraged for such tasks. The code provided here offers a transparent look into Jigsaw's methods for categorization, summarization, and identifying points of agreement and disagreement in free response public opinion research. Our goal in sharing this is to inspire others by providing a potential starting point and useful elements for those tackling similar challenges.

## Overview

Effectively understanding large-scale public input is a significant challenge. Traditional methods require choosing between the breadth of polls or the depth of focus groups. This initiative showcases how Google's Gemini models can allow those seeking to understand public opinion to get the best of both approaches, transforming massive volumes of raw community feedback into clear, digestible insights.

## Using Sensemaking AI

**The tools in the `src` directory are provided as a functional pipeline you can run on your own data.** You can leverage these components to transform raw community feedback into structured insights. Specifically, tools are provided for:

* **Adaptive Interviewing**: Launch surveys with Adaptive Interviewing to gather dynamic, follow-up driven responses.
* **Data Preparation**: Reformat and moderate raw exports from Qualtrics into a consistent structure for analysis.
* **Topic Modeling and Quote Extraction**: Use Gemini to identify discussion topics and extract representative quotes from participant dialogue.
* **Quote Ranking**: Apply classifiers to score quotes on attributes like reasoning and curiosity, helping you identify high-quality contributions.
* **Discussion Summarization**: Automatically generate report text, including overviews and topic summaries.
* **Interactive Visualization**: Build a user-friendly web interface for exploring the synthesized data.
* **Proposition Generation & Simulated Juries**: Generate and rank propositions using a Simulated Jury technique to identify consensus and create nuanced summaries.

### Running the Sensemaking Pipeline

This guide provides step-by-step instructions for running the Sensemaking tools. This pipeline transforms raw survey data into structured insights, including summaries, and visualizations.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3** and **pip3**
* **Node.js** and **NPM** (required for the interactive visualization step)
* A **Gemini API Key**

Install the Python dependencies by running:

```shell
pip3 install -r requirements.txt
```

Confirm all tests pass by running:

```shell
python3 -m pytest src/
```

### Step-by-Step Pipeline

#### 1\. Data Preparation

If you are using Qualtrics for data collection, you can use the following script to transform your raw survey export into a standard format. You will need to specify which columns map to your fixed questions and AI follow-up questions.

```shell
bash src/survey_processing.sh \
  --input_csv <PATH_TO_RAW_EXPORT.csv> \
  --output_dir <OUTPUT_DIR> \
  --round_1_question_response_text "Q1,Q35,Q36" \
  --round_1_follow_up_questions "Q1FU,Q2FU,Q3FU" \
  --round_1_follow_up_question_response_text "Q23,Q37,Q39"
```

*This generates a `processed.csv` file in your output directory.*

**The pipeline can be run on data from sources other than Qualtrics. Later steps only require a CSV with columns for participant\_id (a unique identifier for each participant) and survey\_text (the content to be analyzed).**

**(Optional) Run Moderation and Quality Checks:** Utilities are provided to score the responses for quality and flag rows that may need moderation by a human reviewer. You can get these scores by running:

```shell
bash src/moderation.sh \
  --processed_csv <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR> \
  --api_key "$GOOGLE_API_KEY"
```

#### 2\. Topic Discovery and Quote Extraction

Use Gemini to discover discussion topics and extract representative quotes from the dialogue. This script can be run on data from sources other than qualtrics. It only requires an input csv with columns for participant\_id (a unique identifier for each participant) and survey\_text (the text to be analyzed).

```shell
export GOOGLE_API_KEY="your_api_key_here"
python3 -m src.categorization_runner \
  --input_file <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR>/categorization \
  --additional_context_file src/default-additional-context.md
```

##### Optional Flags

* #### `--skip_autoraters`: Skips the self-evaluation step where the model checks its own work for accuracy.

  * #### **Considerations**: Use this to speed up testing or reduce API usage. Keep it enabled for production runs to ensure higher quality categorizations.

* #### `--skip_quote_extraction`: Skips extracting specific quotes and uses the full participant response instead.

  * #### **Considerations**: Use this if your responses are already short and concise, or if you want to preserve the full context of every response.

* #### `--topics "Topic A,Topic B"`: Provide a list of predefined topics instead of letting the model discover them.

  * #### **Considerations**: Use this when you have a specific taxonomy you want to enforce.

##### Understanding the Outputs:

#### Running this script generates several files in your output directory. Here is how they differ:

* #### `categorized_with_other.csv`: The complete output containing all original survey columns plus the new topic and opinion assignments.

* #### `categorized_without_other.csv`: The same as above, but filtering out any responses assigned to the fallback "Other" category (often excluded from final reports).

* #### `categorized_..._filtered.csv`: Streamlined versions containing only the essential columns needed for the next steps: `participant_id`, `survey_text`, `quote`, `topic`, and `opinion`. Use these for downstream steps to save processing time.

* #### `categorized_with_other_topic_tree.txt`: A readable text file showing the hierarchy of topics and opinions discovered.

#### 3\. Constructive Quality Scoring

Apply classifiers to score the extracted quotes on attributes like reasoning, personal stories, and curiosity. This helps surface the most constructive contributions. By default these are the first quotes shown in the interactive report.

```shell
python3 -m src.get_bridging_scores \
  --input_csv <OUTPUT_DIR>/categorization/categorized_without_other_filtered.csv \
  --output_csv <OUTPUT_DIR>/bridging_scores.csv \
  --api_key "$GOOGLE_API_KEY"
```

#### 4\. Discussion Summarization

Automatically generate topic-level summaries along with a summary of top-level take aways from the conversation.

```shell
python3 -m src.generate_report_text.generate_report_text \
  --input_csv <OUTPUT_DIR>/bridging_scores.csv \
  --output_dir <OUTPUT_DIR>/report_text \
  --additional_context src/default-additional-context.md
```

##### Understanding the Outputs:

This script generates two JSON files in your output directory:

* `report_data.json`: The primary file used by the interactive visualization. It contains the high-level overview and the summaries for each discovered topic.
* `report_data_with_opinions.json`: A more detailed version that also includes summaries for every individual opinion within the topics. This is useful for debugging or if you want to build a more granular custom interface.

##### How it works:

The script follows a recursive summarization process to ensure accuracy:

1. It summarizes the quotes for each **opinion**.
2. It synthesizes those opinion summaries into a summary for the overall **topic**.
3. It creates the high-level **overview** based on the topic summaries.

#### 5\. Interactive Report

In order to build the web interface to explore the summarized data, begin by **copying  the processed data** into the UI input folder.

```shell
cp <OUTPUT_DIR>/bridging_scores.csv src/report_ui/input/opinions.csv
cp <OUTPUT_DIR>/report_text/report_data.json src/report_ui/input/summary.json
```

##### **Customize the Report (Optional)**:

You can edit `src/report_ui/input/config.json` to customize the report. Key options include:

* `title`: Set the display title of the report.
* `logo`: Set the filename of your header image (placed in the `input/` folder).
* `overview_chart`: Set the display mode for the main chart (`"toggle"`, `"topics"`, or `"opinions"`).
* `number_of_sample_quotes`: Control how many quote previews to display for each opinion.
* `chart_colors`: Provide an array of hex color codes to customize the chart palette.
* `excludedTopics`: Add topic names to this array to hide them from the report.
* `excludedOpinions`: Add opinion names to this array to hide them from the report.
* *For a full list of configuration options, check the `README.md` file in `src/report_ui/`.*
3. **Install dependencies**:

```shell
cd src/report_ui
npm install
```

##### **View and Build the Report**:

4. **Local Viewing**: Run `npm run dev` to start a local web server to view the report.
5. **Web Server Deployment**: Run `npm run build` to output a version optimized for delivery via a web server to `src/report_ui/output/static`.
6. **Offline Viewing**: Run `npm run inline` to output a self-contained version of the report to `src/report_ui/output/inline`.
   * *Note: This may not be suitable for larger conversations.*

#### 6\. Proposition Generation & Simulated Juries

Generate distinct propositions and use a simulated jury technique to rank them and identify statements likely to receive broad agreement. These statements can be used in future validation polls.

**Generate Propositions:**

```shell
python3 -m src.propositions.proposition_generator \
  --r1_input_file <OUTPUT_DIR>/categorization/categorized_without_other_filtered.csv \
  --output_dir <OUTPUT_DIR>/propositions \
  --gemini_api_key "$GOOGLE_API_KEY"
```

**Rank and Refine with Simulated Jury:**

```shell
python3 -m src.proposition_refinement.main \
  --input_pkl <OUTPUT_DIR>/propositions/world_model.pkl \
  --output_pkl <OUTPUT_DIR>/propositions/refined_world_model.pkl \
  --gemini_api_key "$GOOGLE_API_KEY" \
  --run_pav_selection
```

**Extract Final CSVs:**

```shell
# Get all ranked propositions
python3 -m src.world_model.main --query=all_by_topic --output_format=csv \
  <OUTPUT_DIR>/propositions/refined_world_model.pkl > final_propositions.csv

# Get all ranked nuanced propositions:
python -m src.world_model.main --query=all_nuanced --output_format=csv \     <OUTPUT_DIR>/propositions/refined_world_model.pkl > final_nuanced_propositions.csv

```

**Optional: to simplify the language used in propositions (e.g. for nuanced propositions):**

```shell
python -m src.proposition_simplification_runner \
--input_csv <INPUT_CSV> --output_csv <OUTPUT_CSV> \
--gemini_api_key <API_KEY> --model_name gemini-2.5-pro
```

INPUT\_CSV should contain a single column called "original" with the original proposition text.  The OUTPUT\_CSV contains an additional column called "simplification" with the rewritten proposition.

---

### Appendix: Standalone Simulated Juries

If you have a simple CSV of statements and participants and want to predict agreement *without* running the full pipeline, you can run the jury standalone:

```shell
python3 -m src.simulated_jury.main \
  --participants_csv <CSV_WITH_PARTICIPANT_RESPONSES> \
  --statements_csv <CSV_WITH_STATEMENTS_TO_TEST> \
  --output_csv <OUTPUT_DIR>/jury_results.csv \
  --statement_column "proposition"
```

### Case Studies

Jigsaw explored the application of AI to public opinion research through two proofs of concept, [What Could BG Be?](https://jigsaw.google/our-work/reimagining-the-town-hall-meeting/) and [We the People](https://jigsaw.google/our-work/scaling-traditional-focus-groups/).

What Could BG Be? was a month-long public consultation crowdsourcing ideas to inform the 25-year plan for Warren County, Kentucky. [Visit the case study section of this repository to learn more about the project and explore the code base](case_studies/wcbgb/).

We the People was a nationally representative public opinion research initiative, combining the depth of focus groups with the breadth of polls with an aim to identify areas of broad public agreement on the topic of freedom and equality in America. Replication code for this project will be released soon.

## Our Approach

The tools here show how Jigsaw is approaching the application of AI to the emerging field of “sensemaking.” It is offered as an insight into our experimental methods. While parts of this library may be adaptable for other projects, developers should anticipate their own work for implementation, customization, and ongoing support for their specific use case.

## Contribution and Improvements

This repository offers a transparent view of Jigsaw's approach to large-scale conversation sensemaking with AI. Developers can:

* **Review the code** to understand Jigsaw's techniques with LLMs.
* **Leverage components** for their own projects (some customization may be needed).
* **Use the command and prompt examples, and overall approach,** as inspiration for their own sensemaking tools.

This codebase is not actively maintained, but we hope it can serve as inspiration for other projects and encourage experimentation and building upon the ideas shared here\!