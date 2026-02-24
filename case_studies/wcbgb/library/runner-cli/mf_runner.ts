// Copyright 2025 Google LLC
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

// Runs the matrix factorization code in src/stats_utils/matrix_factorization.ts for an input dataset,
// and appends the helpfulness scores to the output data.
// Run like:
// npx ts-node ./evaluations/mf_runner.ts --outputFile "data1.csv" \
// --vertexProject "<your project name here>" \
// --inputFile "comments-with-vote-tallies.csv"

import { Command } from "commander";
import { createObjectCsvWriter } from "csv-writer";
import { communityNotesMatrixFactorization, Rating } from "../src/stats/matrix_factorization";

import * as fs from "fs";
import * as csv from "csv-parse";

interface VoteCsvRow {
  "comment-id": string;
  "voter-id": string;
  vote: number;
}

interface OutputCsvRow {
  "comment-id": string;
  "helpfulness-verbatim": number;
  "helpfulness-agree": number;
  "helpfulness-pass": number;
  "helpfulness-disagree": number;
}

async function getVotesFromCsv(filename: string): Promise<VoteCsvRow[]> {
  return new Promise((resolve, reject) => {
    const votes: VoteCsvRow[] = [];
    fs.createReadStream(filename)
      .pipe(csv.parse({ columns: true }))
      .on("data", (row) => {
        votes.push({ ...row, vote: parseInt(row.vote) });
      })
      .on("end", () => {
        resolve(votes);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

async function main(): Promise<void> {
  // Parse command line arguments.
  const program = new Command();
  program
    .option("-o, --outputFile <file>", "The output file name.")
    .option("-g, --votesFile <file>", "The votes file name.")
    .option(
      "-e, --epochs <epochs>",
      "The number of epochs to train for.",
      (value) => parseInt(value, 10),
      400
    )
    .option(
      "-l, --learningRate <learningRate>",
      "The learning rate for training routine.",
      parseFloats,
      [0.05, 0.01, 0.002, 0.0004]
    );
  program.parse(process.argv);
  const options = program.opts();

  const votes = await getVotesFromCsv(options.votesFile);

  // Sort and map user IDs to sequential indices
  const sortedUserIds = Array.from(new Set(votes.map((vote) => vote["voter-id"]))).sort(
    (a, b) => parseInt(a) - parseInt(b)
  );
  const userIdMap: { [key: string]: number } = {};
  sortedUserIds.forEach((userId, index) => {
    userIdMap[userId] = index;
  });
  console.log("sortedUserIds:", sortedUserIds);

  // Sort and map comment IDs to sequential indices
  const sortedCommentIds = Array.from(new Set(votes.map((vote) => vote["comment-id"]))).sort(
    (a, b) => parseInt(a) - parseInt(b)
  );
  const commentIdMap: { [key: string]: number } = {};
  sortedCommentIds.forEach((commentId, index) => {
    commentIdMap[commentId] = index;
  });
  console.log("sortedCommentIds:", sortedCommentIds);

  // First we do a straightforward ("verbatim") application of the community notes
  // matrix factorization algorithgm, using the votes exactly as we get them from
  // polis (-1, 0, 1).
  const verbatimRatings: Rating[] = votes.map(
    (vote): Rating => ({
      userId: userIdMap[vote["voter-id"]],
      noteId: commentIdMap[vote["comment-id"]],
      rating: vote["vote"],
    })
  );
  console.time("verbatim communityNotesMatrixFactorization:");
  const verbatimHelpfulnessScores = await communityNotesMatrixFactorization(
    verbatimRatings,
    1,
    options.epochs,
    options.learningRate
  );
  console.timeEnd("verbatim communityNotesMatrixFactorization:");

  // Next apply the method by sticking closer to the original implementation, where
  // scores are on a scale of 0 to 1, so that the values will be easier to interpret
  // in terms of the thresholds listed in the original paper. Here, disagree and pass
  // votes are collapsed to 0, since we don't quite want to treat pass as 0.5.
  const agreeRatings: Rating[] = votes.map(
    (vote): Rating => ({
      userId: userIdMap[vote["voter-id"]],
      noteId: commentIdMap[vote["comment-id"]],
      rating: vote["vote"] == 1 ? 1 : 0,
    })
  );
  console.time("agree communityNotesMatrixFactorization:");
  const agreeHelpfulnessScores = await communityNotesMatrixFactorization(
    agreeRatings,
    1,
    options.epochs,
    options.learningRate
  );
  console.timeEnd("agree communityNotesMatrixFactorization:");

  // Similarly as above, but for pass votes, which we can potentially use as a signal for
  // "areas of uncertainty"
  const passRatings: Rating[] = votes.map(
    (vote): Rating => ({
      userId: userIdMap[vote["voter-id"]],
      noteId: commentIdMap[vote["comment-id"]],
      rating: vote["vote"] == 0 ? 1 : 0,
    })
  );
  console.time("pass communityNotesMatrixFactorization:");
  const passHelpfulnessScores = await communityNotesMatrixFactorization(
    passRatings,
    1,
    options.epochs,
    options.learningRate
  );
  console.timeEnd("pass communityNotesMatrixFactorization:");

  // And finally, for disagree consensus, or "common ground against", treated separately
  // from agree and pass votes
  const disagreeRatings: Rating[] = votes.map(
    (vote): Rating => ({
      userId: userIdMap[vote["voter-id"]],
      noteId: commentIdMap[vote["comment-id"]],
      rating: vote["vote"] == -1 ? 1 : 0,
    })
  );
  console.time("disagree communityNotesMatrixFactorization:");
  const disagreeHelpfulnessScores = await communityNotesMatrixFactorization(
    disagreeRatings,
    1,
    options.epochs,
    options.learningRate
  );
  console.timeEnd("disagree communityNotesMatrixFactorization:");

  const outputData: OutputCsvRow[] = sortedCommentIds.map((commentId) => ({
    "comment-id": commentId,
    "helpfulness-verbatim": verbatimHelpfulnessScores[commentIdMap[commentId]],
    "helpfulness-agree": agreeHelpfulnessScores[commentIdMap[commentId]],
    "helpfulness-pass": passHelpfulnessScores[commentIdMap[commentId]],
    "helpfulness-disagree": disagreeHelpfulnessScores[commentIdMap[commentId]],
  }));

  // Write the updated rows to the output file
  const csvWriter = createObjectCsvWriter({
    path: options.outputFile,
    header: [
      { id: "comment-id", title: "comment-id" },
      { id: "helpfulness-verbatim", title: "helpfulness-verbatim" },
      { id: "helpfulness-agree", title: "helpfulness-agree" },
      { id: "helpfulness-disagree", title: "helpfulness-disagree" },
      { id: "helpfulness-pass", title: "helpfulness-pass" },
    ],
  });
  await csvWriter.writeRecords(outputData).then(() => {
    console.log("CSV file written successfully.");
  });
}

function parseFloats(value: string): number[] {
  return value.split(",").map(parseFloat);
}

main();
