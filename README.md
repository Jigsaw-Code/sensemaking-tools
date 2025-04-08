# **Sensemaking tools**

# **Overview**

Jigsaw’s [Sensemaking tools](https://medium.com/jigsaw/making-sense-of-large-scale-online-conversations-b153340bda55) help make sense of large-scale online conversations, leveraging LLMs to categorize statements, and summarize statements and voting patterns to surface actionable insights. There are currently three main functions:

* Topic Identification \- identifies topics and optionally subtopics from a set of statements.  
* Statement Categorization \- sorts statements into topics defined by a user or from the Topic Identification function. Statements can belong to more than one topic.  
* Summarization \- analyzes statements and vote data to output a summary of the conversation, including areas of agreement and areas of disagreement.  
  * Voting patterns are passed in, aggregated by group.  
  * Summaries are run through grounding routines, ensuring that claims are backed up by what was actually said in the conversation, and adding citations or references to surface representative statements.

Please see these [docs](https://jigsaw-code.github.io/sensemaking-tools) for a full breakdown of available methods and types. These tools are still in their beta stage.

# How It Works

## Topic Identification

Jigsaw’s Sensemaking tools provide an option to identify the topics present in the comments. The tool offers flexibility to learn:

* Top-level topics  
* Both top-level and subtopics  
* Sub-topics only, given a set of pre-specified top-level topics

## Statement Categorization

Categorization assigns statements to one or more of the topics and subtopics.  These topics can either be provided by the user, or can be the result of the "topic identification" method described above. Topics are assigned to statements in batches, asking the model to return the appropriate categories for each statement, and leveraging the Vertex API constrained decoding feature to structure this output according to a pre-specified JSON schema, to avoid issues with output formatting.  Additionally, error handling has been added to retry in case an assignment fails.

## Summarization

The summary structure is centered on topics and subtopics. For each subtopic the tool summarizes the primary areas of agreement and disagreement between groups for that subtopic.

### Intro Section

Includes a background on the report, and the number of statements and votes within it. Next there is a list of all the topics and subtopics discussed in the deliberation and how many statements fit under each category.

### Identifying “common ground” and “differences of opinion”

[Computational metrics](https://github.com/Jigsaw-Code/sensemaking-tools/blob/main/src/stats/stats_util.ts) are used to select statements corresponding to points of “common ground” and “differences of opinion”. The metrics used rely on the participant body being partitioned into *opinion groups* (for example, the outputs of a clustering algorithm in the Polis software). These clusters represent groups of participants who tend to vote more similarly to each other than to those from other groups. 

Based on these opinion groups, “common ground” statements are defined as those having broad support across groups. To qualify as a point of common ground, each group has to be in agreement with a statement by at least 60%. Statements are then ranked by *group informed consensus*, defined as the product of each group’s agreement rate. This is highest when all groups agree strongly on a statement, thereby respecting minority dissent.

“Differences of opinion” are identified based on the difference between the agreement rate for each opinion group, as compared with the rest of the participant body. Those statements with the highest difference for a particular group help us understand what distinguishes that group. To qualify as a difference of opinion, the agree rate difference for a group must be at least 30%. To avoid an edge-case where a statement could appear in both the "common ground" and "differences of opinion" sections, statements in the "differences of opinion" section must also have a minimum agree rate below 60%.

Because small sample sizes (low vote counts) can create misleading impressions, statements with fewer than 20 votes total are not included. This avoids, for example, a total of 2 votes in favor of a particular statement being taken as evidence of broad support, and included as a point of common ground, when more voting might reveal relatively low support (or significant differences of opinion between groups).

### Opinion Groups Section

Each group is described based on what makes them unique, using the differences of opinion criteria described above, while also ensuring each group in question mostly agrees with the statements selected.

This section also describes what makes groups similar and different, and uses the common ground logic above to identify similarities. Differences of opinion are selected according where the agree rate differences are highest between any given group and the rest of the participant body, regardless of whether in the direction of agreement or disagreement.

### Per Topic and Subtopic sections

Using the topics and subtopics from our "Topic Identification" and "Statement Categorization" features, short summaries are produced for each subtopic (or topic, if no subtopics are present). Following similar criteria as above, only filtered to the given (sub)topic, points of “Common ground between groups” and “Differences of opinion” are identified and then the top set of statements for each are summarized using an LLM. The statements included in the summary are shown as citations within the summary text. 

### **LLMs Used and Custom Models**

This library is implemented using Google Cloud’s [VertexAI](https://cloud.google.com/vertex-ai). This means the library can be set up to use any model available on VertexAI’s Google Cloud’s [Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models), including the latest Gemini models, the open source model Gemma, and other models like Llama and Claude (full list [here](https://cloud.google.com/model-garden)). The access and quota requirements are controlled by a user’s Google Cloud account.

In addition to models available through VertexAI’s Model Garden, users can integrate custom models using the library’s `Model` abstraction. This can be done by implementing a class with only two methods, one for generating plain text and one for generating structured data ([docs](https://jigsaw-code.github.io/sensemaking-tools/classes/models_model.Model.html) for methods). This allows for the library to be used with models not available in the Model Garden, with other cloud providers, and even with on-premise infrastructure for complete data sovereignty.

### **Costs of Running**

LLM pricing is based on token count and constantly changing. Here we list the token counts for a conversation with \~1000 statements. Please see [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) for an up-to-date cost per input token. As of January 23, 2025 the cost for running topic identification, statement categorization, and summarization was in total under $1 on Gemini 1.5 Pro.  
Token Counts for a 1000 statement conversation

|  | Topic Identification | Statement Categorization | Summarization |
| ----- | ----- | ----- | ----- |
| Input Tokens | 41,000 | 41,000 | 19,000 |
| Output Tokens | 1,000 | 26,000 | 5,000 |

## **Running the tools \- Setup**

First make sure you have `npm` installed (`apt-get install npm` on Ubuntu-esque systems).

Next install the project modules by running:  
`npm install`

### **Using the Default Models \- GCloud Authentication**

A Google Cloud project is required to control quota and access when using the default models that connect to Model Garden. Installation instructions for all machines are [here](https://cloud.google.com/sdk/docs/install-sdk#deb). 

For Linux the GCloud CLI can be installed like:  
`sudo apt install -y google-cloud-cli`

Then to log in locally run:

`gcloud config set project <your project name here>`

`gcloud auth application-default login`

## Example Usage - Javascript

Summarize Seattle’s $15 Minimum Wage Conversation.  

```js
// Set up the tools to use the default Vertex model (Gemini Pro 1.5) and related authentication info.
const mySensemaker = new Sensemaker({
  defaultModel: new VertexModel(
    "myGoogleCloudProject123,
    "us-central1",
  ),
});

// Note: this function does not exist.
// Get data from a discussion in Seattle over a $15 minimum wage.
// CSV containing comment text, vote counts, and group information from:
// https://github.com/compdemocracy/openData/tree/master/15-per-hour-seattle
const comments: Comments[] = getCommentsFromCsv("./comments.csv");

// Learn what topics were discussed and print them out.
const topics = mySensemaker.learnTopics(
  comments,
  // Should include subtopics:
  true,
  // There are no existing topics:
  undefined,
  // Additional context:
  "This is from a conversation on a $15 minimum wage in Seattle"
);
console.log(topics);

// Summarize the conversation and print the result as Markdown.
const summary = mySensemaker.summarize(
  comments,
  SummarizationType.GROUP_INFORMED_CONSENSUS,
  topics,
  // Additional context:
  "This is from a conversation on a $15 minimum wage in Seattle"
);
console.log(summary.getText("MARKDOWN"));
```

## **CLI Usage**

There is also a simple CLI set up for testing. There are two tools:

* ./runner-cli/runner.ts: takes in a CSV representing a conversation and outputs an HTML file containing the summary. The summary is best viewed as an HTML file so that the included citations can be hovered over to see the original comment and votes.  
* ./runner-cli/rerunner.ts: takes in a CSV representing a conversation and reruns summarization a number of times and outputs each of the summaries in one CSV. This is useful for testing consistency.


## **Making Changes to the tools \- Development**

### **Testing**

Unit tests can be run with the following command: `npm test`  
To run tests continuously as you make changes run: `npm run test-watch`

## **Documentation**

The documentation [here](https://jigsaw-code.github.io/sensemaking-tools) is the hosted version of the html from the docs/ subdirectory. This documentation is automatically generated using typedoc, and to update the documentation run:  
`npx typedoc`

## **Feedback**

If you have questions or issues with this library, please leave feedback [here](https://docs.google.com/forms/d/e/1FAIpQLSd6kScXaf0d8XR7X9mgHBgG11DJYXV1hEzYLmqpxMcDFJxOhQ/viewform?resourcekey=0-GTVtn872epNsEHtI2ClBEA) and we will reach out to you. Our team is actively evaluating Sensemaking performance and is aiming to share our results on this page in the future. Please note that performance results may vary depending on the model selected.

## **Cloud Vertex Terms of Use**

This library is designed to leverage Cloud Vertex, and usage is subject to the [Cloud Vertex Terms of Service](https://cloud.google.com/terms/service-terms) and the [Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy).  
