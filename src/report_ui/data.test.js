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
import vm from "node:vm";

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

describe("createFilterCompare modal state & reset", () => {
  function createMockDom() {
    class MockElement {
      constructor(tag) {
        this.tag = tag;
        this.attrs = new Map();
        this.classes = new Set();
        this.children = [];
        this.textContent = "";
        this.innerHTML = "";
        this.value = "";
        this.selected = false;
        this.handlers = new Map();
      }

      setAttr(key, val) {
        const str = String(val);
        this.attrs.set(key, str);
        if (key === "class") {
          this.classes = new Set(str.split(/\s+/).filter(Boolean));
        }
      }

      matchesSingle(token) {
        if (token.startsWith("#")) {
          return this.attrs.get("id") === token.slice(1);
        }
        if (token.startsWith(".")) {
          return this.classes.has(token.slice(1));
        }
        return this.tag === token;
      }

      queryAll(selector) {
        const tokens = selector.trim().split(/\s+/);
        const search = (node, idx) => {
          const results = [];
          for (const child of node.children) {
            if (child.matchesSingle(tokens[idx])) {
              if (idx === tokens.length - 1) {
                results.push(child);
              } else {
                results.push(...search(child, idx + 1));
              }
            }
            results.push(...search(child, idx));
          }
          return results;
        };
        return search(this, 0);
      }

      query(selector) {
        return this.queryAll(selector)[0] || null;
      }

      trigger(event, payload = {}) {
        const fn = this.handlers.get(event);
        if (fn) fn.call(this, payload);
      }
    }

    const wrap = (nodes) => {
      const first = nodes[0] || null;
      const api = {
        classed(name, val) {
          if (val === undefined) {
            return first ? first.classes.has(name) : false;
          }
          for (const n of nodes) {
            if (val) n.classes.add(name);
            else n.classes.delete(name);
          }
          return api;
        },
        select(subSelector) {
          return wrap(first ? [first.query(subSelector)].filter(Boolean) : []);
        },
        html(str) {
          if (str === undefined) return first ? first.innerHTML : "";
          for (const n of nodes) {
            n.innerHTML = str;
            if (str === "") n.children = [];
          }
          return api;
        },
        append(tag) {
          if (!first) return wrap([]);
          const child = new MockElement(tag);
          first.children.push(child);
          return wrap([child]);
        },
        attr(key, val) {
          if (val === undefined) return first ? first.attrs.get(key) : null;
          for (const n of nodes) n.setAttr(key, val);
          return api;
        },
        text(str) {
          if (str === undefined) return first ? first.textContent : "";
          for (const n of nodes) n.textContent = String(str);
          return api;
        },
        property(key, val) {
          if (val === undefined) return first ? first[key] : undefined;
          for (const n of nodes) {
            n[key] = val;
            if (key === "selected" && val && first) {
              // Track selected option value if needed
            }
          }
          return api;
        },
        on(event, handler) {
          for (const n of nodes) n.handlers.set(event, handler);
          return api;
        },
      };
      return api;
    };

    const root = new MockElement("body");
    const openBtn = new MockElement("button");
    openBtn.setAttr("class", "button-filter-compare");
    root.children.push(openBtn);

    const modal = new MockElement("div");
    modal.setAttr("id", "filter-compare");
    for (const [tag, cls] of [
      ["div", "filter-compare-bg"],
      ["button", "button-close"],
      ["div", "filter-compare-groups"],
      ["button", "button-add-group"],
      ["button", "button-reset-groups"],
      ["button", "button-apply-groups"],
    ]) {
      const el = new MockElement(tag);
      el.setAttr("class", cls);
      modal.children.push(el);
    }
    root.children.push(modal);

    const docListeners = new Map();
    const document = {
      getElementById(id) {
        return root.query(`#${id}`);
      },
      addEventListener(event, handler) {
        docListeners.set(event, handler);
      },
      triggerKeyup(key) {
        const fn = docListeners.get("keyup");
        if (fn) fn({ key });
      },
    };

    const d3 = {
      select(target) {
        if (typeof target === "string") {
          const found = root.query(target);
          return wrap(found ? [found] : []);
        }
        return wrap(target ? [target] : []);
      },
      selectAll(selector) {
        return wrap(root.queryAll(selector));
      },
    };

    return { root, modal, openBtn, document, d3 };
  }

  async function mountFilterCompare() {
    const scriptSrc = fs.readFileSync(
      path.join(packageRoot, "src", "script.js"),
      "utf-8",
    );
    const startIdx = scriptSrc.indexOf("async function createFilterCompare()");
    const endIdx = scriptSrc.indexOf("function createShare()", startIdx);
    assert.ok(startIdx !== -1 && endIdx !== -1, "Could not locate createFilterCompare in script.js");
    const fnSource = scriptSrc.slice(startIdx, endIdx);

    const dom = createMockDom();
    let renderCalls = 0;
    const sandbox = {
      compareGroups: [[]],
      renderOverviewChart: () => {
        renderCalls += 1;
      },
      i18n: {},
      window: {
        PAYLOAD: {
          demographics: [
            {
              label: "Country",
              values: [
                { value: "United States", count: 10 },
                { value: "United Kingdom", count: 10 },
                { value: "Australia", count: 10 },
              ],
            },
          ],
        },
      },
      document: dom.document,
      d3: dom.d3,
      getCompareGroups: () => sandbox.compareGroups,
    };

    vm.runInNewContext(
      `${fnSource}\nthis.__init = createFilterCompare; this.__readGroups = () => compareGroups;`,
      sandbox,
    );
    await sandbox.__init();

    return {
      ...dom,
      getCompareGroups: () => JSON.parse(JSON.stringify(sandbox.__readGroups())),
      getRenderCalls: () => renderCalls,
    };
  }

  it("resets comparison and re-renders overview chart when Reset is clicked and modal is closed without Apply", async () => {
    const ctx = await mountFilterCompare();

    // 1. Open modal and configure Group A = United States, Group B = United Kingdom
    ctx.openBtn.trigger("click");
    assert.equal(ctx.modal.classes.has("is-open"), true);

    ctx.root.query("#filter-compare .button-add-group").trigger("click");
    const selectA = ctx.root.query("#filter-group-0-country");
    const selectB = ctx.root.query("#filter-group-1-country");
    selectA.value = "United States";
    selectA.trigger("change");
    selectB.value = "United Kingdom";
    selectB.trigger("change");

    // Apply comparison
    ctx.root.query("#filter-compare .button-apply-groups").trigger("click");
    assert.equal(ctx.modal.classes.has("is-open"), false);
    assert.equal(ctx.getRenderCalls(), 1);
    assert.deepEqual(ctx.getCompareGroups(), [
      [{ key: "Country", value: "United States" }],
      [{ key: "Country", value: "United Kingdom" }],
    ]);

    // 2. Reopen modal, click Reset, then close via the X button without clicking Apply
    ctx.openBtn.trigger("click");
    assert.equal(ctx.modal.classes.has("is-open"), true);

    ctx.root.query("#filter-compare .button-reset-groups").trigger("click");
    assert.equal(ctx.getRenderCalls(), 2, "Reset should immediately re-render overview chart");
    assert.deepEqual(ctx.getCompareGroups(), [[]]);

    ctx.root.query("#filter-compare .button-close").trigger("click");
    assert.equal(ctx.modal.classes.has("is-open"), false);
    assert.deepEqual(ctx.getCompareGroups(), [[]]);
  });

  it("discards un-applied draft changes when modal is closed without clicking Apply", async () => {
    const ctx = await mountFilterCompare();

    // Apply Group A = United States vs Group B = United Kingdom
    ctx.openBtn.trigger("click");
    ctx.root.query("#filter-compare .button-add-group").trigger("click");
    const selectA = ctx.root.query("#filter-group-0-country");
    const selectB = ctx.root.query("#filter-group-1-country");
    selectA.value = "United States";
    selectA.trigger("change");
    selectB.value = "United Kingdom";
    selectB.trigger("change");
    ctx.root.query("#filter-compare .button-apply-groups").trigger("click");
    assert.equal(ctx.getRenderCalls(), 1);

    // Reopen, add a 3rd group and edit Group A, then cancel via Escape
    ctx.openBtn.trigger("click");
    ctx.root.query("#filter-compare .button-add-group").trigger("click");
    assert.equal(ctx.root.queryAll(".filter-group-card").length, 3);
    const draftSelectA = ctx.root.query("#filter-group-0-country");
    draftSelectA.value = "Australia";
    draftSelectA.trigger("change");

    ctx.document.triggerKeyup("Escape");
    assert.equal(ctx.modal.classes.has("is-open"), false);
    assert.equal(ctx.getRenderCalls(), 1, "Closing without Apply should not re-render chart");
    assert.deepEqual(ctx.getCompareGroups(), [
      [{ key: "Country", value: "United States" }],
      [{ key: "Country", value: "United Kingdom" }],
    ]);

    // Reopening again should show the 2 applied groups, not the discarded 3rd draft group
    ctx.openBtn.trigger("click");
    assert.equal(ctx.root.queryAll(".filter-group-card").length, 2);
  });

  it("tears down previous resize listener and re-renders on every renderOverviewChart call", () => {
    const scriptSrc = fs.readFileSync(
      path.join(packageRoot, "src", "script.js"),
      "utf-8",
    );
    const startIdx = scriptSrc.indexOf("let activeOverviewChart =");
    const endIdx = scriptSrc.indexOf("function createDemographicChart(", startIdx);
    assert.ok(
      startIdx !== -1 && endIdx !== -1,
      "Could not locate addDebouncedResizeListener/renderOverviewChart in script.js",
    );
    const fnSource = scriptSrc.slice(startIdx, endIdx);

    const activeListeners = new Set();
    let topicChartMounts = 0;
    let opinionChartMounts = 0;

    const sandbox = {
      clearTimeout,
      setTimeout,
      window: {
        addEventListener(event, fn) {
          if (event === "resize") activeListeners.add(fn);
        },
        removeEventListener(event, fn) {
          if (event === "resize") activeListeners.delete(fn);
        },
      },
      createTopicChart() {
        topicChartMounts += 1;
        return sandbox.__addResize(() => {});
      },
      createOpinionChart() {
        opinionChartMounts += 1;
        return sandbox.__addResize(() => {});
      },
    };

    vm.runInNewContext(
      `${fnSource}\nthis.__addResize = addDebouncedResizeListener; this.__render = renderOverviewChart; this.__setMode = (m) => { activeOverviewChart = m; };`,
      sandbox,
    );

    sandbox.__render();
    assert.equal(topicChartMounts, 1);
    assert.equal(opinionChartMounts, 0);
    assert.equal(activeListeners.size, 1);

    // Re-rendering in topics mode cleans up the old listener and mounts a fresh topic chart
    sandbox.__render();
    assert.equal(topicChartMounts, 2);
    assert.equal(activeListeners.size, 1);

    // Switching to opinions mode cleans up topic listener and mounts opinion chart
    sandbox.__setMode("opinions");
    sandbox.__render();
    assert.equal(opinionChartMounts, 1);
    assert.equal(activeListeners.size, 1);
  });
});

describe("createTopicChart comparison topic headers", () => {
  function mountTopicChart(compareGroups) {
    const scriptSrc = fs.readFileSync(
      path.join(packageRoot, "src", "script.js"),
      "utf-8",
    );
    const startIdx = scriptSrc.indexOf("function getTopicData()");
    const endIdx = scriptSrc.indexOf("function createParticipantChart()", startIdx);
    assert.ok(
      startIdx !== -1 && endIdx !== -1,
      "Could not locate getTopicData/createTopicChart in script.js",
    );
    const fnSource = scriptSrc.slice(startIdx, endIdx);

    class MockNode {
      constructor(tag) {
        this.tag = tag;
        this.attrs = new Map();
        this.classes = new Set();
        this.children = [];
        this.textContent = "";
        this.innerHTML = "";
        this.styles = new Map();
        this.handlers = new Map();
      }
      setAttr(k, v) {
        const str = String(v);
        this.attrs.set(k, str);
        if (k === "class") {
          this.classes = new Set(str.split(/\s+/).filter(Boolean));
        }
      }
      getBoundingClientRect() {
        return { width: 800 };
      }
      matchesSingle(token) {
        if (token.startsWith("#")) return this.attrs.get("id") === token.slice(1);
        if (token.startsWith(".")) return this.classes.has(token.slice(1));
        return this.tag === token;
      }
      queryAll(selector) {
        const tokens = selector.trim().split(/\s+/);
        const search = (node, idx) => {
          const out = [];
          for (const child of node.children) {
            if (child.matchesSingle(tokens[idx])) {
              if (idx === tokens.length - 1) out.push(child);
              else out.push(...search(child, idx + 1));
            }
            out.push(...search(child, idx));
          }
          return out;
        };
        return search(this, 0);
      }
    }

    const chartEl = new MockNode("div");
    chartEl.setAttr("id", "conversation-overview-chart");

    const wrap = (nodes, dataItems = null) => {
      const first = nodes[0] || null;
      const api = {
        node: () => first,
        html(str) {
          if (str === "") {
            for (const n of nodes) {
              n.innerHTML = "";
              n.children = [];
            }
          } else {
            for (const n of nodes) n.innerHTML = String(str);
          }
          return api;
        },
        text(str) {
          for (const n of nodes) n.textContent = String(str);
          return api;
        },
        append(tag) {
          if (dataItems) {
            const created = dataItems.map((datum) => {
              const child = new MockNode(tag);
              child.__data__ = datum;
              if (first) first.children.push(child);
              return child;
            });
            return wrap(created);
          }
          if (!first) return wrap([]);
          const child = new MockNode(tag);
          first.children.push(child);
          return wrap([child]);
        },
        attr(k, v) {
          nodes.forEach((n, i) => {
            const val = typeof v === "function" ? v(n.__data__, i) : v;
            n.setAttr(k, val);
          });
          return api;
        },
        style(k, v) {
          nodes.forEach((n, i) => {
            const val = typeof v === "function" ? v(n.__data__, i) : v;
            n.styles.set(k, val);
          });
          return api;
        },
        on(evt, fn) {
          for (const n of nodes) n.handlers.set(evt, fn);
          return api;
        },
        selectAll() {
          return {
            data(arr) {
              return {
                enter() {
                  return wrap(nodes, arr);
                },
              };
            },
          };
        },
      };
      return api;
    };

    const opinions = Array.from({ length: 7 }, (_, idx) => ({
      fullID: `privacy-op-${idx + 1}`,
      text: `Privacy opinion ${idx + 1}`,
      count: 2,
      countFormatted: "2",
    }));

    const quoteMap = new Map();
    opinions.forEach((op, idx) => {
      if (idx < 5) {
        // First 5 opinions appear in both US and UK
        quoteMap.set(op.fullID, [
          { id: op.fullID, quote: "us quote", Country: "United States" },
          { id: op.fullID, quote: "uk quote", Country: "United Kingdom" },
        ]);
      } else {
        // Last 2 opinions only appear in UK (absent from Group A = US)
        quoteMap.set(op.fullID, [
          { id: op.fullID, quote: "uk only quote", Country: "United Kingdom" },
        ]);
      }
    });
    quoteMap.set("ukonly-op-1", [
      { id: "ukonly-op-1", quote: "uk exclusive topic quote", Country: "United Kingdom" },
    ]);

    const sandbox = {
      compareGroups,
      quoteMap,
      i18n: {},
      TOPIC_PALETTE: ["#4285F4", "#34A853"],
      lowSampleThreshold: 30,
      formatNumber: (n) => String(n),
      formatQuotes: (n) => `${n} Quotes`,
      tippy: () => {},
      clearTimeout,
      setTimeout,
      addDebouncedResizeListener: () => () => {},
      window: {
        PAYLOAD: {
          topics: [
            {
              topicID: "privacy",
              text: "Privacy, Cybersecurity, and Fraud",
              opinionCount: 7,
              rawQuoteCount: 12,
              quoteCount: 12,
              quoteCountFormatted: "12",
              opinions,
            },
            {
              topicID: "ukonly",
              text: "UK Specific Topic",
              opinionCount: 3,
              rawQuoteCount: 1,
              quoteCount: 1,
              quoteCountFormatted: "1",
              opinions: [
                {
                  fullID: "ukonly-op-1",
                  text: "UK opinion 1",
                  count: 1,
                  countFormatted: "1",
                },
              ],
            },
          ],
        },
        addEventListener: () => {},
        removeEventListener: () => {},
      },
      d3: {
        select: () => wrap([chartEl]),
        sum: (arr, fn) => arr.reduce((acc, x) => acc + fn(x), 0),
        max: (arr, fn) => Math.max(...arr.map(fn)),
        descending: (a, b) => (b < a ? -1 : b > a ? 1 : 0),
        color: () => ({ darker: () => "#222" }),
      },
    };

    vm.runInNewContext(
      `${fnSource}\nthis.__getTopicData = getTopicData; this.__createTopicChart = createTopicChart;`,
      sandbox,
    );
    sandbox.__createTopicChart();
    return {
      chartEl,
      groups: sandbox.__getTopicData(),
    };
  }

  it("derives topic header opinion counts from the full dataset rather than Group A's filtered opinions", () => {
    const { chartEl, groups } = mountTopicChart([
      [{ key: "Country", value: "United States" }],
      [{ key: "Country", value: "United Kingdom" }],
    ]);

    // Group A only matched 5 opinions in "privacy", while full dataset has 7
    assert.equal(groups[0].topics[0].opinions.length, 5);
    assert.equal(groups[0].topics[0].opinionCount, 7);

    const wrappers = chartEl.queryAll(".topic-wrapper");
    assert.equal(
      wrappers.length,
      2,
      "Both topics with quotes across any active group should render",
    );

    const privacyHeader = wrappers[0].queryAll(".topic-header")[0];
    const privacyMeta = wrappers[0].queryAll(".topic-meta")[0];
    assert.equal(
      privacyHeader.textContent.trim(),
      "Privacy, Cybersecurity, and Fraud",
    );
    assert.equal(privacyMeta.textContent, "(7 opinions)");

    // Second topic had 0 quotes in Group A (US) and 1 quote in Group B (UK), but 3 opinions in full dataset
    const ukHeader = wrappers[1].queryAll(".topic-header")[0];
    const ukMeta = wrappers[1].queryAll(".topic-meta")[0];
    assert.equal(ukHeader.textContent.trim(), "UK Specific Topic");
    assert.equal(ukMeta.textContent, "(3 opinions)");
    assert.equal(wrappers[1].queryAll(".topic-group").length, 2);
  });
});
