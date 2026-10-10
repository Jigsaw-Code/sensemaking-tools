/**
 * @fileoverview Tests for bridging-score handling in data.js.
 * Run with `npm test` (uses the built-in node:test runner).
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import {
  BRIDGING_COLUMN,
  getBridgingScore,
  hasBridgingColumn,
  processReportData,
} from "./data.js";

const packageRoot = path.dirname(fileURLToPath(import.meta.url));

describe("getBridgingScore", () => {
  it("reads AVERAGE_OF_3_BRIDGING, the column get_bridging_scores.py writes", () => {
    assert.equal(BRIDGING_COLUMN, "AVERAGE_OF_3_BRIDGING");
    assert.equal(getBridgingScore({ AVERAGE_OF_3_BRIDGING: "0.75" }), 0.75);
  });

  it("returns 0 for missing, empty, or non-numeric values", () => {
    assert.equal(getBridgingScore({}), 0);
    assert.equal(getBridgingScore({ AVERAGE_OF_3_BRIDGING: "" }), 0);
    assert.equal(getBridgingScore({ AVERAGE_OF_3_BRIDGING: "nan" }), 0);
    assert.equal(getBridgingScore({ AVERAGE_OF_3_BRIDGING: null }), 0);
  });

  it("accepts numeric values", () => {
    assert.equal(getBridgingScore({ AVERAGE_OF_3_BRIDGING: 0.5 }), 0.5);
  });
});

describe("hasBridgingColumn", () => {
  it("is true when any row has the column, even if empty", () => {
    assert.equal(hasBridgingColumn([{ AVERAGE_OF_3_BRIDGING: "" }]), true);
  });

  it("is false when no row has the column", () => {
    assert.equal(hasBridgingColumn([{ quote: "hi" }]), false);
    assert.equal(hasBridgingColumn([]), false);
  });
});

/**
 * Runs processReportData on `opinionsRaw` in a temporary directory.
 * @param {Object[]} opinionsRaw - Rows as csvtojson would produce them.
 * @returns {{ result: Object, warnings: string[] }} The output and any
 *     console.warn messages.
 */
function runReport(opinionsRaw) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "report-ui-test-"));
  const summaryPath = path.join(workDir, "summary.json");
  fs.writeFileSync(
    summaryPath,
    JSON.stringify({
      title: "Test",
      text: "Summary.",
      sub_contents: [{ title: "Topic", text: "Topic summary." }],
    }),
  );
  const warnings = [];
  const originalWarn = console.warn;
  const originalLog = console.log;
  console.warn = (msg) => warnings.push(String(msg));
  console.log = () => {};
  try {
    const result = processReportData({
      opinionsRaw,
      summaryPath,
      inputDir: workDir,
      packageRoot,
      workDir,
    });
    return { result, warnings };
  } finally {
    console.warn = originalWarn;
    console.log = originalLog;
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

/**
 * Builds a csvtojson-style row for a single quote.
 * @param {string} quote
 * @param {string} participantId
 * @param {Object} [extra] - Extra columns, e.g. bridging scores.
 * @returns {Object}
 */
function row(quote, participantId, extra = {}) {
  return {
    topic: "Topic",
    opinion: "Opinion",
    quote,
    participant_id: participantId,
    ...extra,
  };
}

describe("processReportData bridging order", () => {
  it("orders quotes by AVERAGE_OF_3_BRIDGING, highest first", () => {
    const { result, warnings } = runReport([
      row("low", "p1", { AVERAGE_OF_3_BRIDGING: "0.2" }),
      row("unscored", "p2", { AVERAGE_OF_3_BRIDGING: "" }),
      row("high", "p3", { AVERAGE_OF_3_BRIDGING: "0.9" }),
    ]);

    assert.deepEqual(
      result.quotes.map((q) => q.quote),
      ["high", "low", "unscored"],
    );
    assert.deepEqual(warnings, []);
  });

  it("warns when no bridging column is present", () => {
    const { result, warnings } = runReport([
      row("first", "p1"),
      row("second", "p2"),
    ]);

    assert.deepEqual(
      result.quotes.map((q) => q.quote),
      ["first", "second"],
    );
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /AVERAGE_OF_3_BRIDGING/);
  });
});
