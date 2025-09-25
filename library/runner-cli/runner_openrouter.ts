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

// Summarizes the input file and outputs a HTML report using OpenRouter models.
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
//  npx ts-node ./library/runner-cli/runner_openrouter.ts --outputBasename out \
// --inputFile "data.csv" \
// --additionalContext "Description of the conversation"

import { Command } from "commander";
import { writeFileSync } from "fs";
import {
  getCommentsFromCsv,
  getSummary,
  writeSummaryToGroundedCSV,
  writeSummaryToHtml,
} from "./runner_openrouter_utils";

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
      "--output_lang <language>",
      "Output language for the report (default: en, supported: en, zh-TW)",
      "en"
    );
  program.parse(process.argv);
  const options = program.opts();

  const comments = await getCommentsFromCsv(options.inputFile);

  const summary = await getSummary(
    comments,
    undefined,
    options.additionalContext,
    options.output_lang
  );

  const markdownContent = summary.getText("MARKDOWN");
  writeFileSync(options.outputBasename + "-summary.md", markdownContent);
  writeSummaryToHtml(summary, options.outputBasename + "-summary.html");
  writeSummaryToGroundedCSV(summary, options.outputBasename + "-summaryAndSource.csv");

  const jsonContent = JSON.stringify(summary, null, 2);
  writeFileSync(options.outputBasename + "-summary.json", jsonContent);
}

main();
