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

// Summarizes the input file and outputs a HTML report.
//
// There are 3 outputs:
//  summary.md: the summary as a Markdown file
//  summary.html: the summary as a HTML file with hover over citations
//  summaryAndSource.csv: a CSV of each paragraph of the summary and the comments
//     associated with them.
//
// The input CSV is expected to have the following columns: comment-id, comment_text, and votes.
// Vote data should be included in one of two forms - for data without group information the
// columns should be: agrees, disagrees, and optionally passes. For data with group information
// the columns should be: {group name}-agree-count, {group name}-disagree-count, and optionally
// {group name}-pass-count for each group.
//
// Sample Usage:
//  npx ts-node ./library/runner-cli/runner.ts --outputBasename out \
// --vertexProject "{CLOUD_PROJECT_ID}" \
// --inputFile "data.csv"

import { Command } from "commander";
import { readFileSync, writeFileSync } from "fs";
import {
  getCommentsFromCsv,
  getSummary,
  writeSummaryToGroundedCSV,
  writeSummaryToHtml,
} from "./runner_utils";

async function main(): Promise<void> {
  // Parse command line arguments.
  const program = new Command();
  program
    .option(
      "-o, --outputBasename <file>",
      "The output basename, this will be prepended to the output file names."
    )
    .option("-i, --inputFile <file>", "The input file name.")
    .option(
      "-a, --additionalContext <context>",
      "A short description of the conversation to add context."
    )
    .option(
      "--additionalContextFile <file>",
      "A file containing the additional context."
    )
    .option("-v, --vertexProject <project>", "The Vertex Project name.");
  program.parse(process.argv);
  const options = program.opts();

  if (options.additionalContext && options.additionalContextFile) {
    console.error("Error: Cannot specify both --additionalContext and --additionalContextFile");
    process.exit(1);
  }

  let additionalContext = options.additionalContext;
  if (options.additionalContextFile) {
    additionalContext = readFileSync(options.additionalContextFile, "utf-8").trim();
  }

  const comments = await getCommentsFromCsv(options.inputFile);

  const summary = await getSummary(
    options.vertexProject,
    comments,
    undefined,
    additionalContext
  );

  const markdownContent = summary.getText("MARKDOWN");
  writeFileSync(options.outputBasename + "-summary.md", markdownContent);
  writeSummaryToHtml(summary, options.outputBasename + "-summary.html");
  writeSummaryToGroundedCSV(summary, options.outputBasename + "-summaryAndSource.csv");

  const jsonContent = JSON.stringify(summary, null, 2);
  writeFileSync(options.outputBasename + "-summary.json", jsonContent);
}

main();
