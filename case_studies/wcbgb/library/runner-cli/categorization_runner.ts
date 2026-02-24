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

// Learns and assigns topics and subtopics to a CSV of comments.
//
// The input CSV must contain "comment_text" and "comment-id" fields. The output CSV will contain
// all input fields plus a new "topics" field which concatenates all topics and subtopics, e.g.
// "Transportation:PublicTransit;Transportation:Parking;Technology:Internet"
//
// Sample Usage:
// npx ts-node library/runner-cli/categorization_runner.ts \
//    --topicDepth 2 \
//    --outputFile ~/outputs/test.csv  \
//    --vertexProject "{CLOUD_PROJECT_ID}" \
//    --inputFile ~/input.csv \

import { VertexModel } from "../src/models/vertex_model";
import { Sensemaker } from "../src/sensemaker";
import { Comment, Topic } from "../src/types";
import { Command } from "commander";
import { parse } from "csv-parse";
import { createObjectCsvWriter } from "csv-writer";
import * as fs from "fs";
import * as path from "path";
import { concatTopics } from "./runner_utils";

type CommentCsvRow = {
  "comment-id": string;
  comment_text: string;
  topics: string;
};

async function main(): Promise<void> {
  // Parse command line arguments.
  const program = new Command();
  program
    .option("-o, --outputFile <file>", "The output file name.")
    .option("-i, --inputFile <file>", "The input file name.")
    .option("-t, --topics <comma separated list>", "Optional list of top-level topics.")
    .option(
      "-d, --topicDepth [number]",
      "If set, will learn only topics (1), topics and subtopics (2), or topics, subtopics, and subsubtopics (3). The default is 2.",
      "2"
    )
    .option(
      "-a, --additionalContext <instructions>",
      "A short description of the conversation to add context."
    )
    .option(
      "--additionalContextFile <file>",
      "A file containing the additional context."
    )
    .option("-v, --vertexProject <project>", "The Vertex Project name.")
    .option(
      "-f, --forceRerun",
      "Force rerun of categorization, ignoring existing topics in the input file."
    );
  program.parse(process.argv);
  const options = program.opts();

  if (options.additionalContext && options.additionalContextFile) {
    console.error("Error: Cannot specify both --additionalContext and --additionalContextFile");
    process.exit(1);
  }

  let additionalContext = options.additionalContext;
  if (options.additionalContextFile) {
    additionalContext = fs.readFileSync(options.additionalContextFile, "utf-8").trim();
  }

  options.topicDepth = parseInt(options.topicDepth);
  if (![1, 2, 3].includes(options.topicDepth)) {
    throw Error("topicDepth must be one of 1, 2, or 3");
  }

  const csvRows = await readCsv(options.inputFile);
  let comments = convertCsvRowsToComments(csvRows);
  if (options.forceRerun) {
    comments = comments.map((comment) => {
      delete comment.topics;
      return comment;
    });
  }

  // Learn topics and categorize comments.
  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(options.vertexProject, "global"),
  });
  const topics = options.topics ? getTopics(options.topics) : undefined;
  const categorizedComments = await sensemaker.categorizeComments(
    comments,
    options.topicDepth >= 2 ? true : false,
    topics,
    additionalContext,
    options.topicDepth
  );

  const csvRowsWithTopics = setTopics(csvRows, categorizedComments);

  await writeCsv(csvRowsWithTopics, options.outputFile);
}

async function readCsv(inputFilePath: string): Promise<CommentCsvRow[]> {
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
    const allRows: CommentCsvRow[] = [];
    fs.createReadStream(filePath)
      .pipe(parser)
      .on("error", (error) => reject(error))
      .on("data", (row: CommentCsvRow) => {
        allRows.push(row);
      })
      .on("end", () => resolve(allRows));
  });
}

function convertCsvRowsToComments(csvRows: CommentCsvRow[]): Comment[] {
  const comments: Comment[] = [];
  for (const row of csvRows) {
    comments.push({
      text: row["comment_text"],
      id: row["comment-id"],
    });
  }
  return comments;
}

function setTopics(csvRows: CommentCsvRow[], categorizedComments: Comment[]): CommentCsvRow[] {
  // Create a map from comment-id to csvRow
  const mapIdToCsvRow: { [commentId: string]: CommentCsvRow } = {};
  for (const csvRow of csvRows) {
    const commentId = csvRow["comment-id"];
    mapIdToCsvRow[commentId] = csvRow;
  }

  // For each comment in categorizedComments
  //   lookup corresponding original csv row
  //   add a "topics" field that concatenates all topics/subtopics
  const csvRowsWithTopics: CommentCsvRow[] = [];
  for (const comment of categorizedComments) {
    const csvRow = mapIdToCsvRow[comment.id];
    csvRow["topics"] = concatTopics(comment);
    csvRowsWithTopics.push(csvRow);
  }
  return csvRowsWithTopics;
}

async function writeCsv(csvRows: CommentCsvRow[], outputFile: string) {
  // Expect that all objects have the same keys, and make id match header title
  const header: { id: string; title: string }[] = [];
  for (const column of Object.keys(csvRows[0])) {
    header.push({ id: column, title: column });
  }
  const csvWriter = createObjectCsvWriter({
    path: outputFile,
    header: header,
  });
  csvWriter
    .writeRecords(csvRows)
    .then(() => console.log(`CSV file written successfully to ${outputFile}.`));
}

function getTopics(commaSeparatedTopics: string): Topic[] {
  const topics: Topic[] = [];
  for (const topic of commaSeparatedTopics.split(",")) {
    topics.push({ name: topic });
  }
  return topics;
}

main();
