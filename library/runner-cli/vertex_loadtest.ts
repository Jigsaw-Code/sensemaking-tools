// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0
//
// CLI that simulates N concurrent users making real Vertex requests via Sensemaker.
// Produces JSONL logs of each request and an HTML dashboard with latency and error charts.
//
// Sample usage (30 users, 1 iteration each):
// npx ts-node ./library/runner-cli/vertex_loadtest.ts \
//   --users 30 --iterations 1 \
//   --vertexProject "YOUR_PROJECT" \
//   --inputFile ./sample_input.csv \
//   --outDir ./.out \
//   --rpm 120 --maxConcurrency 8 \
//   --additionalContext "Short description"

import { Command } from "commander";
import * as path from "path";
import * as fs from "fs";
import { Sensemaker } from "../src/sensemaker";
import { MetricsVertexModel } from "../src/models/metrics_vertex_model";
import { SummarizationType, Comment, Topic } from "../src/types";
import { VertexMetrics } from "../src/metrics/vertex_metrics";
import { executeConcurrently } from "../src/sensemaker_utils";
import { getCommentsFromCsv, getTopicsFromComments } from "./runner_utils";

async function buildSensemaker(project: string): Promise<Sensemaker> {
  // Use MetricsVertexModel for all tasks to ensure every call is logged.
  const model = new MetricsVertexModel(project, "us-central1");
  return new Sensemaker({
    defaultModel: model,
    categorizationModel: model,
    summarizationModel: model,
    // topicModel: model,
  });
}

async function runUser(
  userId: number,
  sensemaker: Sensemaker,
  comments: Comment[],
  topics: Topic[] | undefined,
  additionalContext?: string
): Promise<void> {
  // Tag all nested Vertex calls with this virtual user
  VertexMetrics.setCurrentUser(userId);
  try {
    // We run summarize; Sensemaker will categorize (no-op if topics provided) then summarize.
    await sensemaker.summarize(
      comments,
      SummarizationType.AGGREGATE_VOTE,
      topics,
      additionalContext
    );
  } finally {
    VertexMetrics.setCurrentUser(null);
  }
}

async function main() {
  const program = new Command();
  program
    .requiredOption("-v, --vertexProject <project>", "Vertex Project ID")
    .requiredOption("-i, --inputFile <file>", "Input CSV file with comments and votes")
    .option("-o, --outDir <dir>", "Output directory", ".out")
    .option("-u, --users <n>", "Number of concurrent users", (v) => parseInt(v, 10), 30)
    .option("-n, --iterations <n>", "Iterations per user", (v) => parseInt(v, 10), 1)
    .option("--rpm <n>", "Requests per minute budget", (v) => parseInt(v, 10))
    .option("--maxConcurrency <n>", "Max parallel requests", (v) => parseInt(v, 10))
    .option(
      "-a, --additionalContext <context>",
      "Short description of the conversation for the LLM"
    );

  program.parse(process.argv);
  const opts = program.opts();

  const outDir = path.resolve(opts.outDir);
  fs.mkdirSync(outDir, { recursive: true });
  VertexMetrics.init(outDir);

  const comments = await getCommentsFromCsv(opts.inputFile);
  const topics = getTopicsFromComments(comments);

  const sensemaker = await buildSensemaker(opts.vertexProject);

  const tasks: Array<() => Promise<void>> = [];
  for (let u = 0; u < opts.users; u++) {
    for (let it = 0; it < opts.iterations; it++) {
      tasks.push(() => runUser(u, sensemaker, comments, topics, opts.additionalContext));
    }
  }

  const execOpts =
    opts.rpm && opts.maxConcurrency
      ? { rpm: opts.rpm as number, maxConcurrency: opts.maxConcurrency as number }
      : { enableRetry: true };

  const start = Date.now();
  await executeConcurrently(tasks, execOpts as never);
  const elapsed = (Date.now() - start) / 1000;
  console.log(
    `Load test finished in ${elapsed.toFixed(1)}s. Logs at ${outDir}/vertex_requests.jsonl`
  );

  VertexMetrics.close();

  // Generate dashboard HTML
  const { generateDashboard } = await import("./report_vertex_metrics");
  await generateDashboard(
    path.join(outDir, "vertex_requests.jsonl"),
    path.join(outDir, "dashboard.html")
  );
  console.log(`Dashboard written to ${path.join(outDir, "dashboard.html")}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
