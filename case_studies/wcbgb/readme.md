# **What Could BG Be?**

What Could BG Be? was a month-long public consultation held in Warren County, Kentucky. The product of a partnership between [InnoEngine](https://www.innoengine.co/), a local innovation consultancy, and [Jigsaw](https://jigsaw.google/), What Could BG Be? aimed to assist local leaders in developing a 25-year plan for the county’s future development by crowd-sourcing ideas from constituents via [Pol.is](http://Pol.is) and leveraging Google‘s most advanced AI models to translate that public input into actionable insights.

The tools here illustrate methods for:

* **Topic Identification**: identifies the main points of discussion. The level of detail is configurable, allowing the tool to discover: just the top level topics; topics and subtopics; or the deepest level, topics, subtopics, and themes (sub-subtopics).  
* **Statement Categorization**: sorts statements into topics defined by a user or from the Topic Identification function. Statements can belong to more than one topic.  
* **Summarization**: analyzes statements and vote data to output a summary of the conversation, including an overview, themes discussed, and areas of agreement and disagreement.

Please see these [docs](https://jigsaw-code.github.io/sensemaking-tools/docs/) for a full breakdown of available methods and types. You can [learn more about What Could BG Be? on Jigsaw’s website](https://jigsaw.google/our-work/reimagining-the-town-hall-meeting/).

## **Running the Tools**

This walkthrough has been written specifically for working with data from Polis. You can, however, use these tools with data derived from other sources. To learn more, [see working with data from non-Polis sources](#working-with-data-from-non-polis-sources) at the bottom of this walkthrough.

### **Setup**

First make sure you have `npm` installed (`apt-get install npm` on Ubuntu-esque systems).

Install the project modules by running: `npm install`

### **GCloud Authentication**

A Google Cloud project is required to control quota and access. Installation instructions for all machines are [here](https://cloud.google.com/sdk/docs/install-sdk#deb).

First, set your GCloud project by running: 

```shell
gcloud config set project <your project name here>
```

Then log in by running the following command. This will open a new browser window where you will provide your Google account credentials.

```shell
gcloud auth application-default login
```

### **Retrieve your Data**

You can download data from your conversation by visiting the reports tab in the Polis administrative interface for your conversation. You will need to download both the comments.csv and participant-votes.csv files.

Alternatively, data can be accessed with wget by running the following command from the terminal. You’ll need need your report ID in order to do so. You can find it at the end of the URL when you access the administrative interface for your report: https://pol.is/m/YOUR-REPORT-ID

```shell
REPORT_ID=ADD_YOUR_REPORT_ID_HERE
REPORT_DIR="https://pol.is/api/v3/reportExport/${REPORT_ID}"
DATA_DIR="${REPORT_ID}-$(date +'%Y%m%d-%H%M%S')"
wget -P ${DATA_DIR} ${REPORT_DIR}/comments.csv
wget -P ${DATA_DIR} ${REPORT_DIR}/participant-votes.csv
mv ${DATA_DIR}/participant-votes.csv ${DATA_DIR}/participants-votes.csv 
```

### **Preprocess your Data**

Run the following code to transform the data, combining comments.csv and participant-votes.csv to prepare it to be run through the topic modeling and categorization routines.

```shell
python3 library/bin/process_polis_data.py $PWD/${DATA_DIR} --output_file=${DATA_DIR}/processed_polis_data.csv
```

### **Generate Topics and Categorize Statements**

From the root of the directory, run the following command to generate topics and categorize the statements from your conversation.

```shell
# topicDepth: defaults to 2, maximum of 3
# additionalContext: Optional string added to the topic modeling prompt, this can help provide more specific or relevant topics for categorization by giving the model greater detail on the focus of the conversation
# topics: Optional comma separated array of top-level topics. If no top-level topics are provided, the categorization routine will identify them itself
npx ts-node library/runner-cli/categorization_runner.ts \
  --vertexProject "$(gcloud config get-value project)" \
  --topicDepth 2 \
  --inputFile "${DATA_DIR}/processed_polis_data.csv" \
  --outputFile "${DATA_DIR}/categorized_polis_data.csv" \
  --additionalContext "" \
  --topics ""

```

### **Generate Summary**

To generate the narrative summary, run the following command:

```shell
npx ts-node library/runner-cli/advanced_runner.ts \
  --vertexProject "$(gcloud config get-value project)" \
  --outputBasename "final" \
  --inputFile ${DATA_DIR}/categorized_polis_data.csv
```

This process generates `final-comments-with-scores.json`, `final-summary.json`, and `final-topic-stats.json` for use in generating the final HTML report.

### **Create the Interactive Report**

The final steps can be used to generate an interactive report that can be deployed to a web server or, in a final step, converted into a single HTML file that can be shared via email.

To begin, run the following command to move the newly created data files into the report data directory.

```shell
mv -f final-comments-with-scores.json web-ui/data/comments-with-scores.json
mv -f final-summary.json web-ui/data/summary.json
mv -f final-topic-stats.json web-ui/data/topic-stats.json
```

Then install the necessary dependencies for the report data visualization by running the following:

```shell
cd visualization-library && npm install
npm run build
```

Next you’ll need to change to the web-ui directory and install the necessary dependencies. This can be done by running the following commands from the terminal from the visualization-library directory.

```shell
cd ../web-ui && npm install
npm run build
```

Build the report by running the following code.

```shell
npx ts-node site-build.ts --topics data/topic-stats.json --summary data/summary.json --comments data/comments.json --reportTitle "Title of Report"
```

Finally, the report files can optionally be bundled into a single HTML file allowing for easier distribution via email if so desired. To do so run:

```shell
npx ts-node single-html-build.js
```

### **Working with Data from Non-Polis Sources**

While the code in this repository was originally built to work with Polis, it can also be used to analyze and provide a summary of data from other sources, including other deliberative technology platforms.

The categorization runner requires only that the csv of your data contains columns with the headers `comment-id`, a unique string for each piece of text to be categorized, and `comment_body` containing the text to be categorized.

The data can also be summarized by running `runner.ts`, which likewise requires both the `comment-id` and `comment_body` columns along with vote data organized into either `agrees`, `disagrees`, and `passes` columns or, if respondents are grouped, `{GroupName}-agree-count`, `{GroupName}-disagree-count`, and `{GroupName}-pass-count` columns.

In order to generate the JSON files used to create the interactive report with `advanced_runner.ts` your data must also contain a `topics` column, either generated by this library or in another manner. Topics should be formatted as a semi-colon separated list of topic and sub-topic separated by a colon, for example: Infrastructure and Transportation:Pedestrian and Bicycle Infrastructure;Infrastructure and Transportation:Public Transportation.

In cases where no vote data is available, reports can still be generated by adding the agrees, disagrees and passes columns and filling them with 0s. This will however reduce the utility of the visualizations, which rely on this data.

## **How it Works**

### **Documentation**

The documentation [here](https://jigsaw-code.github.io/sensemaking-tools/docs/) is the hosted version of the html from the docs/ subdirectory. This documentation is automatically generated using typedoc, and to update the documentation run:  
`npx typedoc`

### **Topic Identification**

The library provides an option to identify the topics present in the comments. The tool offers flexibility to learn:

* Top-level topics  
* Both top-level and subtopics  
* Sub-topics only, given a set of pre-specified top-level topics

Topic identification code can be found in [library/src/tasks/topic\_modeling.ts](https://github.com/Jigsaw-Code/sensemaking-tools/blob/main/library/src/tasks/topic_modeling.ts).

### **Statement Categorization**

Categorization assigns statements to one or more of the topics and subtopics. These topics can either be provided by the user, or can be the result of the "topic identification" method described above.

Topics are assigned to statements in batches, asking the model to return the appropriate categories for each statement, and leveraging the Vertex API constrained decoding feature to structure this output according to a pre-specified JSON schema, to avoid issues with output formatting. Additionally, error handling has been added to retry in case an assignment fails.

Statement categorization code can be found in [library/src/tasks/categorization.ts](https://github.com/Jigsaw-Code/sensemaking-tools/blob/main/library/src/tasks/categorization.ts).

### **Summarization**

The summarization is output as a narrative report, but users are encouraged to pick and choose which elements are right for their data (see example from the runner [here](https://github.com/Jigsaw-Code/sensemaking-tools/blob/521dd0c4c2039f0ceb7c728653a9ea495eb2c8e9/runner-cli/runner.ts#L54)) and consider showing the summarizations alongside visualizations.

Summarization code can be found in [library/rc/tasks/summarization.ts](https://github.com/Jigsaw-Code/sensemaking-tools/blob/main/library/src/tasks/summarization.ts).

### **Testing**

Unit tests can be run with the following command: `npm test`  
To run tests continuously as you make changes run: `npm run test-watch`

### **LLMs Used and Custom Models**

This library is implemented using Google Cloud’s [Vertex AI](https://cloud.google.com/vertex-ai), and works with the latest Gemini models. The access and quota requirements are controlled by a user’s Google Cloud account.

In addition to Gemini models available through Vertex AI, users can integrate custom models using the library’s `Model` abstraction. This can be done by implementing a class with only two methods, one for generating plain text and one for generating structured data ([docs](https://jigsaw-code.github.io/sensemaking-tools/docs/classes/models_model.Model.html) for methods). This allows for the library to be used with models other than Gemini, with other cloud providers, and even with on-premise infrastructure for complete data sovereignty.

Please note that performance results for existing functionality may vary depending on the model selected.

### **Costs of Running**

LLM pricing is based on token count and constantly changing. Here we list the token counts for a conversation with \~1000 statements. Please see [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) for an up-to-date cost per input token. As of April 10, 2025 the cost for running topic identification, statement categorization, and summarization was in total under $1 on Gemini 1.5 Pro.  
Token Counts for a 1000 statement conversation

|  | Topic Identification | Statement Categorization | Summarization |
| :---- | :---- | :---- | :---- |
| Input Tokens | 130,000 | 130,000 | 80,000 |
| Output Tokens | 50,000 | 50,000 | 7,500 |

### **Evaluations**

Our text summary consists of outputs from multiple LLM calls, each focused on summarizing a subset of comments. We have evaluated these LLM outputs for hallucinations both manually and using autoraters. Autorating code can be found in [library/evals/autorating](https://github.com/Jigsaw-Code/sensemaking-tools/tree/main/library/evals/autorating).

We have evaluated topic identification and categorization using methods based on the silhouette coefficient. This evaluation code will be published in the near future. We have also considered how stable the outputs are run to run and comments are categorized into the same topic(s) \~90% of the time, and the identified topics also show high stability.

## **Cloud Vertex Terms of Use**

This library is designed to leverage Vertex AI, and usage is subject to the [Cloud Vertex Terms of Service](https://cloud.google.com/terms/service-terms) and the [Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy).