# Sensemaking by Jigsaw \- A Google AI Proof of Concept

This repository shares tools developed by [Jigsaw](http://jigsaw.google) as a proof of concept to help make sense of large-scale online conversations. It demonstrates how Large Language Models (LLMs) like Gemini can be leveraged for such tasks. The code provided here offers a transparent look into Jigsaw's methods for categorization, summarization, and identifying points of agreement and disagreement in free response public opinion research. Our goal in sharing this is to inspire others by providing a potential starting point and useful elements for those tackling similar challenges.

To learn more about Jigsaw’s previous work in Sensemaking, please also [visit our case studies repository](https://github.com/Jigsaw-Code/sensemaking-case-studies/).

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
* A **Gemini API Key** (and optionally a Google Cloud API Key for DLP)

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


**Working with non-English or multilingual responses?** See [Working with Non-English and Multilingual Data](#working-with-non-english-and-multilingual-data) below for guidance and considerations.


**(Optional) Run Moderation and Quality Checks:** Utilities are provided to score the responses for quality and flag rows that may need moderation by a human reviewer. You can get these scores by running:

```shell
bash src/moderation.sh \
  --processed_csv <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR> \
  --gemini_api_key "$GEMINI_API_KEY" \
  --gcloud_api_key "$GCLOUD_API_KEY"
```

#### 2\. Topic Discovery and Quote Extraction

Use Gemini to discover discussion topics and extract representative quotes from the dialogue. This script can be run on data from sources other than qualtrics. It only requires an input csv with columns for participant\_id (a unique identifier for each participant) and survey\_text (the text to be analyzed).

```shell
python3 -m src.categorization_runner \
  --input_file <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR>/categorization \
  --gemini_api_key "$GEMINI_API_KEY" \
  --additional_context_file src/default-additional-context.md
```

##### Optional Flags

* #### `--skip_autoraters`: Skips the self-evaluation step where the model checks its own work for accuracy.

  * #### **Considerations**: Use this to speed up testing or reduce API usage. Keep it enabled for production runs to ensure higher quality categorizations.

* #### `--skip_quote_extraction`: Skips extracting specific quotes and uses the full participant response instead.

  * #### **Considerations**: Use this if your responses are already short and concise, or if you want to preserve the full context of every response.

* #### `--topics "Topic A,Topic B"`: Provide a list of predefined topics instead of letting the model discover them.

  * #### **Considerations**: Use this when you have a specific taxonomy you want to enforce.

* #### `--max_concurrent_calls N` (or `-c N`): Maximum number of concurrent LLM API calls, applied to every stage of categorization. Defaults to 100 for Gemini and 20 for open models. Must be at least 1.

  * #### **Considerations**: Lower this (e.g. `-c 5`) if you hit rate limit or quota errors, or are using an API key with low quota. Higher values finish faster but consume quota more quickly.

##### Running with Gemma (and other open models)

In addition to Google's Gemini API, the pipeline supports running on **open models (such as Gemma)** or any other open-weights model served via an **OpenAI API compatible endpoint** (e.g., using vLLM, LiteLLM, Ollama, or LM Studio).

To route requests to an open model instead of Gemini, configure the following environment variables before running the pipeline:

```shell
# 1. Instruct the factory to use the OpenAI API compatible client
export MODEL_ENDPOINT_TYPE="openai_api_compatible"

# 2. Set the URL of your inference server (e.g., vLLM or LM Studio)
export OPENAI_API_ENDPOINT_URL="http://localhost:8000/v1"

# 3. Set the API key (use a dummy value if your local server doesn't require one)
export OPENAI_API_ENDPOINT_KEY="your-api-key-or-dummy"

# (Optional) Choose the response format style
# - "openai_json_schema": Uses the modern structured JSON schema format (default, recommended for vLLM/LM Studio)
# - "llama_cpp_old": Fallback for older llama-cpp servers
export OPENAI_RESPONSE_FORMAT_STYLE="openai_json_schema"
```

Then, run the runner passing the path or identifier of the model to the `--model_name` flag:

```shell
python3 -m src.categorization_runner \
  --input_file <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR>/categorization \
  --additional_context_file src/default-additional-context.md \
  --model_name "google/gemma-4-26b-it" # Or the absolute path to your local model weights
```

##### Running judgments on a System One decision model

System One decision models answer typed questions (yes/no, a choice among options, or a position on an ordered scale) with probabilities, in one pass, instead of generating text. They do not implement chat, so `MODEL_ENDPOINT_TYPE=openai_api_compatible` cannot point at them. When the variables below are set, judgment stages send their questions to a decision model. Generative stages stay on Gemini or the OpenAI-compatible model: topic and opinion names, quote extraction, report text, proposition writing, translation, and jury ranking.

Judgment stages that move: topic assignment, opinion assignment, moderation and bridging scores, opinion autorater verdicts, other eval verdicts, jury approval votes, proposition equivalence clustering, and the dedup winner / topic-home pick.

```shell
export DECISION_ENDPOINT_TYPE="systemone"
export DECISION_MODEL="clef"
```

With only these two set, requests go to `POST /v1/systemone` on a local Ollama 0.35.1+ server running Clef. Optional settings, which only need to be set to change the default:

* `DECISION_MODEL_HOST`: where the model is served. A bare host such as `gpu-box` or `gpu-box:11434` is read the way the Ollama CLI reads `OLLAMA_HOST`. A URL without a path gets `/v1/systemone`. A URL with a path is used as given.
* `DECISION_MODEL_API_KEY`: bearer key for a hosted endpoint.
* `DECISION_MODEL_KEEP_ALIVE`: how long Ollama keeps the model loaded between stages (default `30m` for the default local server). Once `DECISION_MODEL_HOST` is set it is sent only when given, because other servers reject the field.
* `DECISION_MAX_QUESTIONS`: most questions per request (default `64`, Ollama's limit). Lower it for hosts with a smaller limit.
* `DECISION_MAX_BODY_BYTES`: largest request body in bytes (default `65536`, Ollama's limit). Every request carries the whole state, so this bounds how long a state can be. Raise it for hosts that read long states, such as SubQ Snap.
* `DECISION_THRESHOLD`: probability cutoff for labels and equivalence, between 0 and 1 (default `0.5`).
* `DECISION_MAX_CONCURRENT`: requests in flight (default `1`, because a local runner has one slot).

Locally, Clef and `pplx-decider:27b-q4_K_M` have been tested on Ollama 0.40.11. Hosted endpoints that have been tested with the same request shape:

| Model | `DECISION_MODEL_HOST` | `DECISION_MODEL` |
| --- | --- | --- |
| Clef on Cloudflare Workers AI | `https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/run/@cf/cloudflare/clef` | `clef` |
| TypeSafe Jev | `https://api.typesafe.ai` | `jev-latest` |
| Inception Mercury Decide | `https://api.inceptionlabs.ai/v1/decisions` | `mercury-decide` |
| SubQ Snap | `https://api.subq.ai` | `subq-snap-1.1` |
| Liquid d1-ultra | `https://api.liquid.ai/decisions/v1/systemone` | `d1-ultra` |

Assignment asks one yes/no question per label, and Other is not asked: it is assigned only when no other label clears the threshold. Labels beyond the question limit are split across requests. Equivalence is pairwise, then joined in code. Approval votes use the existing scale. Autorater verdicts are a score question on the 0-4 rubric, read at its most likely level, with no written explanation; a level below `autorater_min_score` is finalized as Other without retry, because the decision is deterministic. Ranking still needs a generative model.

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
  --gemini_api_key "$GEMINI_API_KEY"
```

To save quota, the script only scores what it needs to:

* Rows that already have all three bridging scores are reused (for example, when re-running on a previous output). Rows with any score missing are scored in full. Pass `--force_rerun` to rescore every row.
* Duplicate quotes are scored once.
* Empty quotes are never scored. Optionally, pass `--min_text_length <N>` to also skip quotes shorter than `N` characters (default `0`, off).

Skipped rows have empty scores and a reason in the `BRIDGING_SKIP_REASON` column (`empty_text`, `too_short`, or `scoring_failed`).

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

#### 5\. Proposition Generation & Simulated Juries

Generate distinct propositions and use a simulated jury technique to rank them and identify statements likely to receive broad agreement. These statements can be used in future validation polls.

**Generate Propositions:**

```shell
python3 -m src.propositions.proposition_generator \
  --r1_input_file <OUTPUT_DIR>/categorization/categorized_without_other_filtered.csv \
  --output_dir <OUTPUT_DIR>/propositions \
  --gemini_api_key "$GEMINI_API_KEY"
```

**Rank and Refine with Simulated Jury:**

```shell
python3 -m src.proposition_refinement.main \
  --input_pkl <OUTPUT_DIR>/propositions/world_model.pkl \
  --output_pkl <OUTPUT_DIR>/propositions/refined_world_model.pkl \
  --gemini_api_key "$GEMINI_API_KEY" \
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
--gemini_api_key "$GEMINI_API_KEY" --model_name gemini-3.5-flash
```

INPUT\_CSV should contain a single column called "original" with the original proposition text.  The OUTPUT\_CSV contains an additional column called "simplification" with the rewritten proposition.

**Note: Standalone Simulated Juries**

If you have a simple CSV of statements and participants and want to predict agreement *without* running the full pipeline, you can run the jury standalone:

```shell
python3 -m src.simulated_jury.main \
  --participants_csv <CSV_WITH_PARTICIPANT_RESPONSES> \
  --statements_csv <CSV_WITH_STATEMENTS_TO_TEST> \
  --output_csv <OUTPUT_DIR>/jury_results.csv \
  --statement_column "proposition"
```

#### 6\. Interactive Report

In order to build the web interface to explore the summarized data, begin by **copying the processed data** into the UI input folder.

```shell
cp <OUTPUT_DIR>/bridging_scores.csv src/report_ui/input/opinions.csv
cp <OUTPUT_DIR>/report_text/report_data.json src/report_ui/input/summary.json
```

Alternatively, skip the copy step and pass paths explicitly:

```shell
cd src/report_ui
node build.js inline \
  --bridging_scores <OUTPUT_DIR>/bridging_scores.csv \
  --summary <OUTPUT_DIR>/report_text/report_data.json \
  --output <OUTPUT_DIR>/report.html
```

##### **Customize the Report (Optional)**:

You can edit `src/report_ui/input/config.json` to customize the report. Key options include:

* `title`: Set the display title of the report.
* `logo`: Set the filename of your header image (placed in the `input/` folder).
* `overview_chart`: Set the display mode for the main chart (`"toggle"`, `"topics"`, or `"opinions"`).
* `number_of_sample_quotes`: Control how many quote previews to display for each opinion.
* `chart_colors`: Provide an array of hex color codes to customize the chart palette.
* `demographic_colors`: Provide a single hex color code (to generate a six-step light-to-dark palette) or an array of color codes to customize the participant chart palette.
* `excluded_topics`: Add topic names to this array to hide them from the report.
* `excluded_opinions`: Add opinion names to this array to hide them from the report.
* *For a full list of configuration options, check the `README.md` file in `src/report_ui/`.*

##### **Add Advanced Features (Optional)**:

You can enhance your report by configuring two additional features before building:

1. **Demographic Support:** You can add demographic dimensions to your data by ensuring the relevant columns in your `opinions.csv` are prefixed with `demo:`. For example, `demo:Age` or `demo:Location`. The tool will automatically use these to build a demographic breakdown chart and quote filters.
2. **Predicted Agreement (Propositions):** Because you generated propositions in step 5, you can feature them in a dedicated "Predicted Agreement" tab. Copy your `selected_propositions.csv` file to `src/report_ui/input/propositions.csv` (it requires columns for `topic`, `proposition`, and `approval_rate`).

##### **View and Build the Report**:

3. **Web Server Deployment (Static)**: Run `npm run build` from the `src/report_ui` directory to output a version optimized for delivery via a web server to `src/report_ui/output/static`.
4. **Offline Viewing (Inline)**: Run `npm run inline` to output a self-contained version of the report to `src/report_ui/output/inline`.
   * *Note: This may not be suitable for larger conversations.*

*(Optional) Local Development:*
If you want to view the report using a live local server during development, you will need to install the dev dependencies first:
```shell
cd src/report_ui
npm install
npm run dev
```

### Working with Non-English and Multilingual Data

All system prompts and few-shot examples in the Sensemaking pipeline are written in English. While the pipeline includes utilities for analyzing non-English or mixed-language conversations and requesting outputs in a specific target language, **multilingual performance can vary significantly depending on both the underlying data, including both the size of the dataset and the languages involved, and the model used**:

* **Model Choice:** Frontier multilingual models (such as Gemini) can often categorize mixed-language inputs and follow target-language output instructions while still adhering to structured JSON schemas. Smaller models or open-weights models connected via the OpenAI-compatible endpoint may struggle with cross-lingual clustering, complex multi-step reasoning (such as Simulated Juries), or maintaining non-English output alongside English instructions.
* **Language Resource Level:** Even with frontier models, lower-resource languages typically have higher token density, slower processing times, and less reliable instruction-following than high-resource languages.
* **Validation Recommended:** If you are working with non-English data or non-English output targets, we recommend testing your chosen model on a small sample first to decide whether to run natively on the original text or pre-translate your dataset.

#### 1\. Specifying the Output Language (`--output_language` / `OUTPUT_LANGUAGE`)

By default, when no output language is specified:
* **Mixed-language datasets** and **lower-resource languages** typically produce **English** outputs (topics, opinions, summaries, and propositions).
* **Monolingual datasets in high-resource languages** (such as Spanish) often infer the input language for topic and opinion discovery, but can still unpredictably drift back into English.

As a result, explicitly setting a target output language is generally recommended when working with multilingual and non-English data. To explicitly instruct the model to generate all synthesized outputs (topics, opinions, summaries, propositions, and simplified propositions) in a specific language, set `OUTPUT_LANGUAGE` in your environment (or `.env` file) or pass `--output_language` to each runner step:

```shell
# Option 1: Set once for all pipeline steps
export OUTPUT_LANGUAGE="Spanish"

# Option 2: Pass --output_language to individual LLM runner commands
python3 -m src.categorization_runner \
  --input_file <OUTPUT_DIR>/processed.csv \
  --output_dir <OUTPUT_DIR>/categorization \
  --gemini_api_key "$GEMINI_API_KEY" \
  --additional_context_file src/default-additional-context.md \
  --output_language "Spanish"
```

*Note: `--output_language` only affects model-generated labels, summaries, and propositions. Extracted participant quotes remain verbatim in the participant's original language unless translated separately.*

#### 2\. Working with Mixed-Language Inputs & Translation (`src.translate_csv`)

Sensemaking can create reports from participant responses in multiple languages. There are 2 options for working with multilingual input:

1. **Default:** Frontier multilingual models like Gemini can natively analyze mixed-language datasets (for example, English and Spanish responses in the same survey) without pre-translation—discovering shared topics, categorizing opinions, and generating summaries while preserving participants' original phrasing and tone.
2. **Pre-translation:** We also provide an optional utility (`src.translate_csv`) that translates participant input into a single language before analysis so downstream tasks only process one language.

Using `src.translate_csv` can be helpful in a number of scenarios, including:

* **Low-Resource Languages or Smaller Open Models (Pre-Analysis):** If your dataset includes low-resource languages, or if you are running the pipeline with smaller open-weights models that struggle with cross-lingual clustering and multi-step reasoning (such as Simulated Juries), translating `survey_text` into a single target language before Step 2 can improve consistency.
* **High Token-per-Word Languages:** Because participant responses are passed to the model across many stages of the pipeline (topic discovery, categorization, autoraters, quote extraction, bridging scoring, summarization, and simulated juries), languages or scripts with significantly higher token counts per word can multiply token usage and costs. Translating once upfront into a more token-compact language can reduce overall token consumption.
* **Translating Extracted Quotes for a Monolingual Audience (Post-Analysis):** If you would like to translate all quotes in the final report to a single language, you can use the `translate_csv.py` tool to translate the `quote` column before generating the Interactive Report in Step 6.

##### How to Use `src.translate_csv`

The script detects the language of each entry and translates only rows that are not already in `--target_language` (default `en_us`). Whenever a column is translated, the original untranslated text is automatically preserved in a new `original_text_<column>` column (for example, `original_text_survey_text` or `original_text_quote`).

**To pre-translate raw survey responses before analysis:**

When run on `survey_text` (the default for `--columns`), the script automatically translates the inner `<response>...</response>` text from Qualtrics exports without altering `<question>` tags or XML structure, as well as plain-text `survey_text` columns from custom CSVs:

```shell
python3 -m src.translate_csv \
  --input_csv <OUTPUT_DIR>/processed.csv \
  --output_csv <OUTPUT_DIR>/processed_translated.csv \
  --columns "survey_text" \
  --target_language "en_us" \
  --gemini_api_key "$GEMINI_API_KEY"
```

**To translate only extracted quotes at the end of the pipeline (before Step 6):**

```shell
python3 -m src.translate_csv \
  --input_csv <OUTPUT_DIR>/bridging_scores.csv \
  --output_csv <OUTPUT_DIR>/bridging_scores_translated.csv \
  --columns "quote" \
  --target_language "en_us" \
  --gemini_api_key "$GEMINI_API_KEY"
```

##### Optional Flags

* `--columns`: Comma-separated list of columns to translate (defaults to `survey_text`).
* `--target_language`: Target language code or name to translate into (defaults to `en_us`).
* `--model_name`: Model used for language detection and translation (defaults to `gemini-3.8-flash`).
* `--gemini_api_key`: Gemini API key (if omitted, falls back to the `GEMINI_API_KEY` environment variable).

## Our Approach

The tools here show how Jigsaw is approaching the application of AI to the emerging field of “sensemaking.” It is offered as an insight into our experimental methods. While parts of this library may be adaptable for other projects, developers should anticipate their own work for implementation, customization, and ongoing support for their specific use case.

## Contribution and Improvements

This repository offers a transparent view of Jigsaw's approach to large-scale conversation sensemaking with AI. Developers can:

* **Review the code** to understand Jigsaw's techniques with LLMs.
* **Leverage components** for their own projects (some customization may be needed).
* **Use the command and prompt examples, and overall approach,** as inspiration for their own sensemaking tools.

This codebase is not actively maintained, but we hope it can serve as inspiration for other projects and encourage experimentation and building upon the ideas shared here\!
