// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This code processes data from the `bin/` directory ingest scripts. In general, the shape
// takes the form of the `CoreCommentCsvRow` structure below, together with the vote tally
// columns of the form <Group Name>-agree-count, <Group Name>-disagree-count, and
// <Group Name>-pass-count.

import { Sensemaker } from "../src/sensemaker";
import { VertexModel } from "../src/models/vertex_model";
import {
  Summary,
  VoteTally,
  Comment,
  SummarizationType,
  Topic,
  SummaryContent,
  VoteInfo,
  SupportedLanguage,
} from "../src/types";
import * as path from "path";
import * as fs from "fs";
import { parse } from "csv-parse";
import { marked } from "marked";
import { createObjectCsvWriter } from "csv-writer";

/**
 * Core comment columns, sans any vote tally rows
 */
type CoreCommentCsvRow = {
  index: number;
  timestamp: number;
  datetime: string;
  "comment-id": number;
  "author-id": number;
  agrees: number;
  disagrees: number;
  moderated: number;
  comment_text: string;
  passes: number;
  topics: string; // can contain both topics and subtopics
  topic: string;
  subtopic: string;
};

// Make this interface require that key names look like `group-N-VOTE-count`
type VoteTallyGroupKey =
  | `${string}-agree-count`
  | `${string}-disagree-count`
  | `${string}-pass-count`;

export interface VoteTallyCsvRow {
  [key: VoteTallyGroupKey]: number;
}

//This is a type that combines VoteTallyCsvRow and CoreCommentCsvRow
export type CommentCsvRow = VoteTallyCsvRow & CoreCommentCsvRow;

/**
 * Add the text and supporting comments to statementsWithComments. Also adds nested content.
 * @param summaryContent the content and subcontent to add
 * @param allComments all the comments from the deliberation
 * @param statementsWithComments where to add new summary text and supporting source comments
 * @returns none
 */
function addStatement(
  summaryContent: SummaryContent,
  allComments: Comment[],
  statementsWithComments: { summary: string; source: string }[]
) {
  if (summaryContent.subContents) {
    summaryContent.subContents.forEach((subContent) => {
      addStatement(subContent, allComments, statementsWithComments);
    });
  }

  if (summaryContent.text.length === 0 && !summaryContent.title) {
    return;
  }
  let comments: Comment[] = [];
  if (summaryContent.citations) {
    comments = summaryContent.citations
      .map((commentId: string) => allComments.find((comment: Comment) => comment.id === commentId))
      .filter((comment) => comment !== undefined);
  }
  statementsWithComments.push({
    summary: (summaryContent.title || "") + summaryContent.text,
    source: comments.map((comment) => `*        [${comment.id}] ${comment.text}`).join("\n"),
  });
}

/**
 * Outputs a CSV where each row represents a statement and its associated comments.
 *
 * @param summary the summary to split.
 * @param outputFilePath Path to the output CSV file that will have columns "summary" for the statement, and "comments" for the comment texts associated with that statement.
 */
export function writeSummaryToGroundedCSV(summary: Summary, outputFilePath: string) {
  const statementsWithComments: { summary: string; source: string }[] = [];

  for (const summaryContent of summary.contents) {
    addStatement(summaryContent, summary.comments, statementsWithComments);
  }

  const csvWriter = createObjectCsvWriter({
    path: outputFilePath,
    header: [
      { id: "summary", title: "summary" },
      { id: "source", title: "source" },
    ],
  });
  csvWriter.writeRecords(statementsWithComments);
  console.log(`Summary statements saved to ${outputFilePath}`);
}
/**
 * Identify topics and subtopics when input data has not already been categorized.
 * @param project The Vertex GCloud project name
 * @param comments The comments from which topics need to be identified
 * @returns Promise resolving to a Topic collection containing the newly discovered topics and subtopics for the given comments
 */
export async function getTopicsAndSubtopics(
  project: string,
  comments: Comment[]
): Promise<Topic[]> {
  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(project, "global"),
  });
  return await sensemaker.learnTopics(comments, true);
}

/**
 * Runs the summarization routines for the data set.
 * @param project The Vertex GCloud project name
 * @param comments The comments to summarize
 * @param topics The input topics to categorize against
 * @param additionalContext Additional context about the conversation to pass through
 * @returns Promise resolving to a Summary object containing the summary of the comments
 */
export async function getSummary(
  project: string,
  comments: Comment[],
  topics?: Topic[],
  additionalContext?: string,
  output_lang: SupportedLanguage = "en"
): Promise<Summary> {
  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(project, "global"),
  });
  // TODO: Make the summariation type an argument and add it as a flag in runner.ts. The data
  // requirements (like requiring votes) would also need updated.
  const summary = await sensemaker.summarize(
    comments,
    SummarizationType.AGGREGATE_VOTE,
    topics,
    additionalContext,
    output_lang
  );
  // For now, remove all Common Ground, Difference of Opinion, or TopicSummary sections
  return summary.withoutContents((sc) => sc.type === "TopicSummary");
}

export function writeSummaryToHtml(summary: Summary, outputFile: string) {
  const markdownContent = summary.getText("MARKDOWN");
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Summary</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
    </style>
    ${
      // When in DEBUG_MODE, we need to add the DataTables and jQuery libraries, and hook
      // into our table elements to add support for features like sorting and search.
      process.env.DEBUG_MODE === "true"
        ? `
    <script src="https://code.jquery.com/jquery-3.7.1.js"></script>
    <script src="https://cdn.datatables.net/2.2.1/js/dataTables.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/2.2.1/css/dataTables.dataTables.css" />
    <script>$(document).ready( function () {$('table').DataTable();} )</script>
    `
        : ""
    }
</head>
<body>
    ${marked(markdownContent)}
</body>
</html>`;

  fs.writeFileSync(outputFile, htmlContent);
  console.log(`Written summary to ${outputFile}`);
}

// Returns topics and subtopics concatenated together like
// "Transportation:PublicTransit;Transportation:Parking;Technology:Internet"
export function concatTopics(comment: Comment): string {
  const pairsArray = [];
  for (const topic of comment.topics || []) {
    if ("subtopics" in topic) {
      for (const subtopic of topic.subtopics || []) {
        if ("subtopics" in subtopic && (subtopic.subtopics as Topic[]).length) {
          if ("subtopics" in (subtopic as Topic)) {
            for (const subsubtopic of subtopic.subtopics as Topic[]) {
              pairsArray.push(`${topic.name}:${subtopic.name}:${subsubtopic.name}`);
            }
          }
        } else {
          pairsArray.push(`${topic.name}:${subtopic.name}`);
        }
      }
    } else {
      // handle case where no subtopics available
      pairsArray.push(`${topic.name}`);
    }
  }
  return pairsArray.join(";");
}

/**
 * Parse a topics string from the categorization_runner.ts into a (possibly) nested topics
 * array, omitting subtopics and subsubtopics if not present in the labels.
 * @param topicsString A string in the format Topic1:Subtopic1:A;Topic2:Subtopic2.A
 * @returns Nested Topic structure
 */
export function parseTopicsString(topicsString: string): Topic[] {
  // use the new multiple topic output notation to parse multiple topics/subtopics
  const subtopicMappings = topicsString
    .split(";")
    .reduce(
      (
        topicMapping: { [key: string]: Topic[] },
        topicString: string
      ): { [key: string]: Topic[] } => {
        const [topicName, subtopicName, subsubtopicName] = topicString.split(":");
        // if we already have a mapping for this topic, add, otherwise create a new one
        topicMapping[topicName] = topicMapping[topicName] || [];
        if (subtopicName) {
          let subsubtopic: Topic[] = [];
          let subtopicUpdated = false;
          // Check for an existing subtopic and add subsubtopics there if possible.
          for (const subtopic of topicMapping[topicName]) {
            if (subtopic.name === subtopicName) {
              subsubtopic = "subtopics" in subtopic ? subtopic.subtopics : [];
              if (subsubtopicName) {
                subsubtopic.push({ name: subsubtopicName });
                subtopicUpdated = true;
                break;
              }
            }
          }

          if (subsubtopicName) {
            subsubtopic = [{ name: subsubtopicName }];
          }
          if (!subtopicUpdated) {
            topicMapping[topicName].push({ name: subtopicName, subtopics: subsubtopic });
          }
        }

        return topicMapping;
      },
      {}
    );

  // map key/value pairs from subtopicMappings to Topic objects
  return Object.entries(subtopicMappings).map(([topicName, subtopics]) => {
    if (subtopics.length === 0) {
      return { name: topicName };
    } else {
      return { name: topicName, subtopics: subtopics };
    }
  });
}

/**
 * Gets comments from a CSV file, in the style of the output from the input processing files
 * in the project's `bin/` directory. Core CSV rows are as for `CoreCommentCsvRow`, plus any
 * vote tallies in `VoteTallyCsvRow`.
 * @param inputFilePath
 * @returns
 */
export async function getCommentsFromCsv(inputFilePath: string): Promise<Comment[]> {
  // Determine the groups names from the header row
  const header = fs.readFileSync(inputFilePath, { encoding: "utf-8" }).split("\n")[0];
  const groupNames = header
    .split(",")
    .filter((name: string) => name.includes("-agree-count"))
    .map((name: string) => name.replace("-agree-count", ""))
    .sort();

  const usesGroups = groupNames.length > 0;

  if (!inputFilePath) {
    throw new Error("Input file path is missing!");
  }
  const filePath = path.resolve(inputFilePath);
  const fileContent = fs.readFileSync(filePath, { encoding: "utf-8" });

  const parser = parse(fileContent, {
    delimiter: ",",
    columns: true,
  });

  return new Promise((resolve, reject) => {
    const data: Comment[] = [];
    fs.createReadStream(filePath)
      .pipe(parser)
      .on("error", reject)
      .on("data", (row: CommentCsvRow) => {
        const newComment: Comment = {
          text: row.comment_text,
          id: row["comment-id"].toString(),
          voteInfo: getVoteInfoFromCsvRow(row, usesGroups, groupNames),
        };
        if (row.topics) {
          // In this case, use the topics output format from the categorization_runner.ts routines
          newComment.topics = parseTopicsString(row.topics);
        } else if (row.topic) {
          // Add topic and subtopic from single value columns if available
          newComment.topics = [];
          newComment.topics.push({
            name: row.topic.toString(),
            subtopics: row.subtopic ? [{ name: row.subtopic.toString() }] : [],
          });
        }

        data.push(newComment);
      })
      .on("end", () => resolve(data));
  });
}

function getVoteInfoFromCsvRow(
  row: CommentCsvRow,
  usesGroups: boolean,
  groupNames: string[]
): VoteInfo {
  if (usesGroups) {
    const voteInfo: { [key: string]: VoteTally } = {};
    for (const groupName of groupNames) {
      voteInfo[groupName] = new VoteTally(
        Number(row[`${groupName}-agree-count`]),
        Number(row[`${groupName}-disagree-count`]),
        Number(row[`${groupName}-pass-count`])
      );
    }
    return voteInfo;
  } else {
    return new VoteTally(Number(row["agrees"]), Number(row["disagrees"]), Number(row["passes"]));
  }
}

export function getTopicsFromComments(comments: Comment[]): Topic[] {
  // Create a map from the topic name to a set of subtopic names.
  const mapTopicToSubtopicSet: { [topicName: string]: Set<string> } = {};
  for (const comment of comments) {
    for (const topic of comment.topics || []) {
      if (mapTopicToSubtopicSet[topic.name] == undefined) {
        mapTopicToSubtopicSet[topic.name] = new Set();
      }
      if ("subtopics" in topic) {
        for (const subtopic of topic.subtopics || []) {
          mapTopicToSubtopicSet[topic.name].add(subtopic.name);
        }
      }
    }
  }

  // Convert that map to a Topic array and return
  const returnTopics: Topic[] = [];
  for (const topicName in mapTopicToSubtopicSet) {
    const topic: Topic = { name: topicName, subtopics: [] };
    for (const subtopicName of mapTopicToSubtopicSet[topicName]!.keys()) {
      topic.subtopics.push({ name: subtopicName });
    }
    returnTopics.push(topic);
  }
  return returnTopics;
}
