# Summarization Output

The output of the summarization process is a narrative report that can be generated as JSON, Markdown, CSV or HTML. The output consists of several sections outlined below. To learn how to generate a summary using this code, check the \[readme\](./readme.md)..

## **Introduction Section**

Includes a short bullet list of the number of statements, votes, topics and subtopics within the summary.

## **Overview Section**

The overview section summarizes the “Themes” sections for all subtopics, along with summaries generated for each top-level topic (these summaries are generated as an intermediate step, but not shown to users, and can be thought of as intermediate “chain of thought” steps in the overall recursive summarization approach).

Currently the Overview does not reference the “Common Ground” and “Differences of Opinion” sections described below.

Percentages in the overview (e.g. “Arts and Culture (17%)”) are the percentage of statements that are about this topic. Since statements can be categorized into multiple topics these percentages may add up to more than 100\.

## **Top 5 Subtopics**

Sensemaking selects the top 5 subtopics by statement count, and concisely summarizes key themes found in statements within these subtopics. These themes are more concise than what appears later in the summary, to act as a quick overview.

## **Topic and Subtopic Sections**

Using the topics and subtopics from our “Topic Identification” and “Statement Categorization” features, short summaries are produced for each subtopic (or topic, if no subtopics are present).

For each subtopic, Sensemaking surfaces:

* The number of statements assigned to this subtopic.  
* Prominent themes.  
* A summary of the top statements where we find “common ground” and “differences of opinion", based on agree and disagree rates.  
* The relative level of agreement within the subtopic, as compared to the average subtopic, based on how many comments end up in “common ground” vs “differences of opinion” buckets.

### **Themes**

For each subtopic, Sensemaking identifies up to 5 themes found across statements assigned to that subtopic, and writes a short description of each theme. This section considers all statements assigned to that subtopic.

When identifying themes, Sensemaking leverages statement text and not vote information. Sensemaking attempts to account for differing viewpoints in how it presents themes.

### **Common Ground and Differences of Opinion**

When summarizing “common ground” and “differences of opinion” within a subtopic, Sensemaking summarizes a sample of statements selected based on statistics calculated using the agree, disagree, and pass vote counts for those statements. For each section, Sensemaking selects statements with the clearest signals for common ground and disagreement, respectively. It does not use any form of text analysis (beyond categorization) when selecting the statements, and only considers vote information.

Because small sample sizes (low vote counts) can create misleading impressions, statements with fewer than 20 votes total are not included. This avoids, for example, a total of 2 votes in favor of a particular statement being taken as evidence of broad support, and included as a point of common ground, when more voting might reveal relatively low support (or significant differences of opinion).

For this section, Sensemaking provides grounding citations to show which statements the LLM referenced, and to allow readers to check the underlying text and vote counts.

### **Relative Agreement**

Each subtopic is labeled as “high,” “moderately high,” “moderately low,” or “low” agreement. This is determined by, for each subtopic, getting *all* the comments that qualify as common ground comments and normalizing it based on how many comments were in that subtopic. Then these numbers are compared subtopic to subtopic.