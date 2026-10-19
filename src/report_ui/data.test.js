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
 * @param {Object} [opts] - Optional overrides for summary, config, and predicted JSON.
 * @returns {{ result: Object, warnings: string[] }} The output and any
 *     console.warn messages.
 */
function runReport(opinionsRaw, opts = {}) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "report-ui-test-"));
  const summaryPath = path.join(workDir, "summary.json");
  fs.writeFileSync(
    summaryPath,
    JSON.stringify(
      opts.summary || {
        title: "Test",
        text: "Summary.",
        sub_contents: [{ title: "Topic", text: "Topic summary." }],
      },
    ),
  );
  let configPath = null;
  if (opts.config) {
    configPath = path.join(workDir, "config.json");
    fs.writeFileSync(configPath, JSON.stringify(opts.config));
  }
  if (opts.translations) {
    fs.writeFileSync(
      path.join(workDir, "translations.json"),
      JSON.stringify(opts.translations),
    );
  }
  const propositionsRaw = opts.propositions || [];
  const warnings = [];
  const originalWarn = console.warn;
  const originalLog = console.log;
  console.warn = (msg) => warnings.push(String(msg));
  console.log = () => {};
  try {
    const result = processReportData({
      opinionsRaw,
      propositionsRaw,
      summaryPath,
      configPath,
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

describe("processReportData excluded_topics and excluded_opinions", () => {
  it("filters excluded topics and opinions from both opinions and predicted agreement", () => {
    const { result } = runReport(
      [
        row("q1", "p1", {
          topic: "Keep Topic",
          opinion: "Keep Opinion",
          AVERAGE_OF_3_BRIDGING: "0.5",
        }),
        row("q2", "p2", {
          topic: "Keep Topic",
          opinion: "Drop Opinion",
          AVERAGE_OF_3_BRIDGING: "0.5",
        }),
        row("q3", "p3", {
          topic: "Drop Topic",
          opinion: "Some Opinion",
          AVERAGE_OF_3_BRIDGING: "0.5",
        }),
      ],
      {
        summary: {
          title: "Test",
          text: "Summary.",
          sub_contents: [
            { title: "## Keep Topic", text: "Keep summary." },
            { title: "## Drop Topic", text: "Drop summary." },
          ],
        },
        config: {
          excluded_topics: ["Drop Topic"],
          excluded_opinions: ["Drop Opinion", "Excluded By Opinion Field"],
        },
        propositions: [
          {
            topic: "Keep Topic",
            proposition: "Keep Opinion",
            opinion: "Keep Opinion",
            approval_rate: "0.80",
          },
          {
            topic: "Keep Topic",
            proposition: "Drop Opinion",
            opinion: "Drop Opinion",
            approval_rate: "0.50",
          },
          {
            topic: "Keep Topic",
            proposition: "Statement with excluded opinion",
            opinion: "Excluded By Opinion Field",
            approval_rate: "0.60",
          },
          {
            topic: "Drop Topic",
            proposition: "Keep Opinion",
            opinion: "Keep Opinion",
            approval_rate: "0.90",
          },
        ],
      },
    );

    assert.deepEqual(
      result.dataStatic.topics.map((t) => t.text),
      ["Keep Topic"],
    );
    assert.deepEqual(
      result.dataStatic.topics[0].opinions.map((o) => o.text),
      ["Keep Opinion"],
    );
    assert.equal(result.dataStatic.topics[0].topicCount, 1);
    assert.equal(result.dataStatic.topicsIdentified, 1);
    assert.equal(result.dataStatic.opinionsIdentified, 1);
    assert.equal(result.dataStatic.totalParticipants, "1");

    assert.equal(result.dataStatic.hasPredicted, true);
    assert.deepEqual(
      result.dataStatic.predicted.topics.map((t) => t.title),
      ["Keep Topic"],
    );
    assert.deepEqual(
      result.dataStatic.predicted.topics[0].statements.map((s) => s.text),
      ["Keep Opinion"],
    );
    assert.deepEqual(
      result.dataStatic.predicted.topics[0].statements.map(
        (s) => s.predictedAgreement,
      ),
      ["80%"],
    );
  });

  it("drops predicted topics whose statements are all excluded and updates hasPredicted", () => {
    const { result } = runReport(
      [row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" })],
      {
        config: {
          excluded_opinions: ["Only Statement"],
        },
        propositions: [
          {
            topic: "Topic",
            proposition: "Only Statement",
            opinion: "Only Statement",
            approval_rate: "0.75",
          },
        ],
      },
    );

    assert.deepEqual(result.dataStatic.predicted.topics, []);
    assert.equal(result.dataStatic.hasPredicted, false);
  });
});

describe("processReportData title", () => {
  it("uses config.title when provided, overriding summary.title", () => {
    const { result } = runReport(
      [row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" })],
      {
        summary: {
          title: "# Summary Title",
          text: "Summary.",
          sub_contents: [{ title: "Topic", text: "Topic summary." }],
        },
        config: { title: "# Config Title" },
      },
    );

    assert.equal(result.dataStatic.title, "Config Title");
  });

  it("falls back to summary.title when config.title is not set", () => {
    const { result } = runReport(
      [row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" })],
      {
        summary: {
          title: "# Summary Title",
          text: "Summary.",
          sub_contents: [{ title: "Topic", text: "Topic summary." }],
        },
      },
    );

    assert.equal(result.dataStatic.title, "Summary Title");
  });

  it("defaults to empty string when neither config.title nor summary.title is set", () => {
    const { result } = runReport(
      [row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" })],
      {
        summary: {
          text: "Summary.",
          sub_contents: [{ title: "Topic", text: "Topic summary." }],
        },
      },
    );

    assert.equal(result.dataStatic.title, "");
  });
});

describe("processReportData numeric config options", () => {
  it("preserves low_sample_warning_threshold when set to 0", () => {
    const { result } = runReport(
      [row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" })],
      {
        config: { low_sample_warning_threshold: 0 },
      },
    );

    assert.equal(result.dataStatic.lowSampleThreshold, 0);
    assert.equal(JSON.parse(result.dataStatic.payload).options.lowSampleThreshold, 0);
  });

  it("defaults low_sample_warning_threshold to 30 when unset", () => {
    const { result } = runReport([
      row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.5" }),
    ]);

    assert.equal(result.dataStatic.lowSampleThreshold, 30);
  });
});

describe("index.mustache conditional demographic controls", () => {
  const template = fs.readFileSync(
    path.join(packageRoot, "src", "index.mustache"),
    "utf-8",
  );

  it("omits Filter & Compare, filter modal, drawer filters, and demographics label when hasDemographics is false", async () => {
    const Mustache = (await import("mustache")).default;
    const { result } = runReport(
      [
        row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.9" }),
        row("q2", "p2", { AVERAGE_OF_3_BRIDGING: "0.8" }),
        row("q3", "p3", { AVERAGE_OF_3_BRIDGING: "0.7" }),
      ],
      {
        config: { number_of_sample_quotes: 2 },
      },
    );

    assert.equal(result.dataStatic.hasDemographics, false);
    const html = Mustache.render(template, result.dataStatic);

    assert.doesNotMatch(html, /class="button-filter-compare"/);
    assert.doesNotMatch(html, /id="filter-compare"/);
    assert.doesNotMatch(html, /class="drawer-quotes-filters"/);
    assert.doesNotMatch(html, /<span>View all quotes and demographics<\/span>/);
    assert.match(html, /<span>View all quotes<\/span>/);
  });

  it("renders Filter & Compare, filter modal, drawer filters, and demographics label when hasDemographics is true", async () => {
    const Mustache = (await import("mustache")).default;
    const { result } = runReport(
      [
        row("q1", "p1", { AVERAGE_OF_3_BRIDGING: "0.9", "demo:Age": "18-29" }),
        row("q2", "p2", { AVERAGE_OF_3_BRIDGING: "0.8", "demo:Age": "30-44" }),
        row("q3", "p3", { AVERAGE_OF_3_BRIDGING: "0.7", "demo:Age": "45+" }),
      ],
      {
        config: { number_of_sample_quotes: 2 },
      },
    );

    assert.equal(result.dataStatic.hasDemographics, true);
    const html = Mustache.render(template, result.dataStatic);

    assert.match(html, /class="button-filter-compare"/);
    assert.match(html, /id="filter-compare"/);
    assert.match(html, /class="drawer-quotes-filters"/);
    assert.match(html, /<span>View all quotes and demographics<\/span>/);
  });
});

describe("processReportData 'Other' category localization and sorting", () => {
  it("localizes 'Other' topics and opinions via i18n.chart.otherCategory and sorts them last", () => {
    const { result } = runReport(
      [
        // Give "Other" the most unique participants so it would sort first without special handling
        row("q1", "p1", { topic: "Other", opinion: "Other", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q2", "p2", { topic: "Other", opinion: "Other", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q3", "p3", { topic: "Other", opinion: "Other", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q4", "p4", { topic: "Economy", opinion: "Other", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q5", "p5", { topic: "Economy", opinion: "Other", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q6", "p6", { topic: "Economy", opinion: "Lower taxes", AVERAGE_OF_3_BRIDGING: "0.5" }),
        row("q7", "p7", { topic: "Healthcare", opinion: "Expand access", AVERAGE_OF_3_BRIDGING: "0.5" }),
      ],
      {
        summary: {
          title: "Test",
          text: "Summary.",
          sub_contents: [
            { title: "## Economy", text: "Economy summary." },
            { title: "## Healthcare", text: "Healthcare summary." },
            { title: "## Other", text: "Other summary." },
          ],
        },
        translations: {
          chart: { otherCategory: "Otros" },
        },
      },
    );

    const topics = result.dataStatic.topics;
    assert.deepEqual(
      topics.map((t) => t.text),
      ["Economy", "Healthcare", "Otros"],
    );
    assert.deepEqual(
      topics.map((t) => t.isOther),
      [false, false, true],
    );
    // English slug is preserved for DOM IDs and quote lookup
    assert.equal(topics[2].topicID, "other");
    assert.equal(topics[2].summary, "Other summary.");

    // Within "Economy", "Other" has 2 quotes and "Lower taxes" has 1 quote,
    // but "Other" ("Otros") is sorted last
    assert.deepEqual(
      topics[0].opinions.map((o) => o.text),
      ["Lower taxes", "Otros"],
    );
    assert.deepEqual(
      topics[0].opinions.map((o) => o.isOther),
      [false, true],
    );
    assert.equal(topics[0].opinions[1].fullID, "economy-other");

    // Conversation overview lead uses the top two non-Other topics
    assert.match(result.dataStatic.conversationOverviewLead, /Economy/);
    assert.match(result.dataStatic.conversationOverviewLead, /Healthcare/);
    assert.doesNotMatch(result.dataStatic.conversationOverviewLead, /Otros/);
  });
});
