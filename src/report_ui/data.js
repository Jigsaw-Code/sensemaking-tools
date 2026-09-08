/**
 * @fileoverview Data script for the report.
 * This script performs the ETL (Extract, Transform, Load) process:
 * 1. Loads raw opinion data, configuration, and AI-generated summaries.
 * 2. Transforms flat opinion lists into a hierarchical structure (Topics -> Opinions -> Quotes).
 * 3. Calculates statistics (participant counts, bridging scores).
 * 4. Generates formatted JSON payloads for the frontend ("static" and "inline" variations).
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const demographics_prefix = "demo:";
/**
 * @typedef {Object} RawOpinion
 * @property {string} topic - The high-level topic category.
 * @property {string} opinion - The specific opinion text.
 * @property {string} quote - The actual quote text.
 * @property {string} participant_id - Representative ID (Participant ID).
 * @property {string|number} [AVERAGE_OF_2_BRIDGING] - Used for sorting.
 */

/**
 * Parses `--flag value` and `--flag=value` tokens from argv (after the command).
 * @param {string[]} args
 * @returns {Map<string, string>}
 */
function parseFlags(args) {
  const flags = new Map();
  for (let i = 0; i < args.length; i++) {
    const token = args[i];
    if (!token.startsWith("--")) continue;
    const eq = token.indexOf("=");
    if (eq !== -1) {
      flags.set(token.slice(2, eq), token.slice(eq + 1));
      continue;
    }
    const key = token.slice(2);
    const next = args[i + 1];
    if (next && !next.startsWith("--")) {
      flags.set(key, next);
      i++;
    } else {
      flags.set(key, "");
    }
  }
  return flags;
}

/**
 * Resolves CLI build options from argv.
 * Defaults match the historical copy-into-`input/` workflow.
 *
 * @param {string[]} [argv=process.argv]
 * @param {string} [cwd=process.cwd()]
 * @returns {Object} Resolved build options.
 */
export function resolveBuildOptions(argv = process.argv, cwd = process.cwd()) {
  const args = argv.slice(2);
  let commandToken = args[0] && !args[0].startsWith("--") ? args[0] : null;
  if (!commandToken) {
    commandToken = "static";
  }

  const buildCommands = new Set(["static", "inline", "build"]);
  const isBuildCommand = buildCommands.has(commandToken);

  const flags = parseFlags(args);
  const hasOutput = flags.has("output");
  const hasOutputDir = flags.has("outputDir");

  if (hasOutput && hasOutputDir) {
    throw new Error(
      "Use either --output (inline) or --outputDir (static), not both.",
    );
  }

  let output = null;
  let outputDir = null;
  const effectiveMode =
    commandToken === "build" ? "inline" : commandToken;

  if (isBuildCommand) {
    if (effectiveMode === "inline") {
      if (hasOutputDir) {
        throw new Error(
          "inline mode writes a single HTML file; use --output <file.html> (not --outputDir).",
        );
      }
      output = path.resolve(
        cwd,
        flags.get("output") || "output/inline/index.html",
      );
    } else if (effectiveMode === "static") {
      if (hasOutput) {
        throw new Error(
          "static mode writes multiple artefacts; use --outputDir <dir> (not --output).",
        );
      }
      outputDir = path.resolve(
        cwd,
        flags.get("outputDir") || "output/static",
      );
    }
  }

  const inputDir = path.resolve(cwd, flags.get("inputDir") || "input");

  const opinionsFlag = flags.get("opinions");
  const bridgingFlag = flags.get("bridging_scores");
  if (opinionsFlag && bridgingFlag) {
    const opinionsResolved = path.resolve(cwd, opinionsFlag);
    const bridgingResolved = path.resolve(cwd, bridgingFlag);
    if (opinionsResolved !== bridgingResolved) {
      throw new Error(
        "--opinions and --bridging_scores both set to different paths; use only one.",
      );
    }
  }
  const opinionsPath = path.resolve(
    cwd,
    opinionsFlag || bridgingFlag || path.join(inputDir, "opinions.csv"),
  );
  const summaryPath = path.resolve(
    cwd,
    flags.get("summary") || path.join(inputDir, "summary.json"),
  );

  const predictedExplicit = flags.has("predicted");
  const configExplicit = flags.has("config");

  let predictedPath = null;
  if (predictedExplicit) {
    predictedPath = path.resolve(cwd, flags.get("predicted"));
  } else {
    const defaultPredicted = path.join(inputDir, "predicted.json");
    if (fs.existsSync(defaultPredicted)) {
      predictedPath = defaultPredicted;
    }
  }

  let configPath = null;
  if (configExplicit) {
    configPath = path.resolve(cwd, flags.get("config"));
  } else {
    const defaultConfig = path.join(inputDir, "config.json");
    if (fs.existsSync(defaultConfig)) {
      configPath = defaultConfig;
    }
  }

  const hasExplicitInputPaths = Boolean(
    flags.has("inputDir") ||
      opinionsFlag ||
      bridgingFlag ||
      flags.has("summary") ||
      predictedExplicit ||
      configExplicit,
  );
  const hasExplicitOutputPaths = hasOutput || hasOutputDir;

  if (isBuildCommand) {
    if (!fs.existsSync(opinionsPath)) {
      throw new Error(`Opinions CSV not found: ${opinionsPath}`);
    }
    if (!fs.existsSync(summaryPath)) {
      throw new Error(`Summary JSON not found: ${summaryPath}`);
    }
    if (predictedExplicit && !fs.existsSync(predictedPath)) {
      throw new Error(`Predicted JSON not found: ${predictedPath}`);
    }
    if (configExplicit && !fs.existsSync(configPath)) {
      throw new Error(`Config JSON not found: ${configPath}`);
    }
  }

  return {
    command: commandToken,
    effectiveMode: isBuildCommand ? effectiveMode : commandToken,
    inputDir,
    output,
    outputDir,
    opinionsPath,
    summaryPath,
    predictedPath,
    configPath,
    predictedExplicit,
    configExplicit,
    hasExplicitInputPaths,
    hasExplicitOutputPaths,
  };
}

/**
 * Recursively deep merges source object into target object.
 * @param {Object} target - The default target object.
 * @param {Object} source - The source object with overrides.
 * @returns {Object} A new deeply merged object.
 */
function deepMerge(target, source) {
  const output = { ...target };
  for (const key of Object.keys(source || {})) {
    if (
      source[key] instanceof Object &&
      key in target &&
      target[key] instanceof Object &&
      !Array.isArray(source[key])
    ) {
      output[key] = deepMerge(target[key], source[key]);
    } else if (source[key] !== undefined) {
      output[key] = source[key];
    }
  }
  return output;
}

/**
 * Processes report inputs into static/inline Mustache payloads and writes
 * workDir JSON artefacts (quotes.json, data-static.json, data-inline.json).
 *
 * @param {Object} params
 * @param {RawOpinion[]} params.opinionsRaw
 * @param {string} params.summaryPath
 * @param {string|null} [params.configPath]
 * @param {string|null} [params.predictedPath]
 * @param {string} params.inputDir - Dir for translations / relative logo paths.
 * @param {string} params.packageRoot - Package root (for default-translations).
 * @param {string} params.workDir - Directory to write quotes + data JSON files.
 * @returns {{ dataStatic: Object, dataInline: Object, quotes: Object[] }}
 */
export function processReportData({
  opinionsRaw,
  summaryPath,
  configPath = null,
  predictedPath = null,
  inputDir,
  packageRoot,
  workDir,
}) {
  const opinions = opinionsRaw.map((d, index) => ({
    ...d,
    index,
  }));

  const config = configPath
    ? JSON.parse(fs.readFileSync(configPath, "utf-8"))
    : {};

  const summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));

  const predictedRaw =
    predictedPath && fs.existsSync(predictedPath)
      ? JSON.parse(fs.readFileSync(predictedPath, "utf-8"))
      : [];

  const defaultI18n = JSON.parse(
    fs.readFileSync(
      path.join(packageRoot, "src", "default-translations.json"),
      "utf-8",
    ),
  );

  const userI18nPath = (() => {
    if (config.translations) {
      const direct = path.join(inputDir, config.translations);
      if (fs.existsSync(direct)) return direct;
      const withJson = path.join(inputDir, `${config.translations}.json`);
      if (fs.existsSync(withJson)) return withJson;
    }
    const defaultPath = path.join(inputDir, "translations.json");
    if (fs.existsSync(defaultPath)) return defaultPath;
    return null;
  })();

  const userI18n = userI18nPath
    ? JSON.parse(fs.readFileSync(userI18nPath, "utf-8"))
    : {};

  const i18n = deepMerge(defaultI18n, userI18n);
  i18n.locale = i18n.locale || "en";
  i18n.direction = i18n.direction || "ltr";

  const numberFormatter = new Intl.NumberFormat(i18n.locale);
  const percentFormatter = new Intl.NumberFormat(i18n.locale, {
    style: "percent",
    maximumFractionDigits: 1,
  });

  /**
   * Formats a value as a localized percentage string.
   * @param {number|string} value
   * @returns {string|null}
   */
  function formatPercent(value) {
    if (value == null || value === "") return null;
    const num = Number(value);
    if (isNaN(num)) return null;
    const ratio = num > 1 ? num / 100 : num;
    return percentFormatter.format(ratio);
  }

  const overviewChart = config.overview_chart || "toggle";
  const options = {
    logo: config.logo || "",
    overviewChart,
    hasToggle: overviewChart === "toggle",
    lowSampleThreshold: config.low_sample_warning_threshold || 30,
    sampleQuoteCount: Math.min(
      Math.max(config.number_of_sample_quotes || 4, 2),
      10,
    ), // between 2 and 10 sample quotes per opinion
    topOpinionCount: Math.min(
      Math.max(config.number_of_top_opinions || 10, 2),
      20,
    ), // between 2 and 20 top opinions
    topicColors: config.chart_colors || [
      "#AFB42B",
      "#F4511E",
      "#3949AB",
      "#E52592",
      "#00897B",
      "#EFB22F",
      "#aaa",
    ],
    demographicColors: config.demographic_colors || [
      "#4886f7",
      "#4071d5",
      "#385db3",
      "#2f4a93",
      "#273874",
      "#1e2656",
    ],
  };

  /**
   * Converts a string of markdown to clean HTML.
   * @param {string} text
   * @returns {string}
   */
  function cleanMarkdown(text) {
    if (!text) return "";
    let html = text;

    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

    return html;
  }

  /**
   * Strips markdown header symbols (e.g. '#', '##') and leading whitespace.
   * @param {string} text
   * @returns {string}
   */
  function stripMarkdownHeader(text) {
    if (!text) return "";
    return text.replace(/^#+\s*/, "");
  }

  /**
   * Sums an array of numbers.
   * @param {number[]} arr
   * @returns {number}
   */
  function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
  }

  /**
   * Formats a number according to the active locale.
   * @param {number} num
   * @returns {string}
   */
  function formatNumber(num) {
    return numberFormatter.format(num);
  }

  /**
   * Groups an array of objects by a specific property key.
   * @param {Object[]} array - The array to group.
   * @param {string} column - The key to group by.
   * @returns {[string, Object[]][]} An array of [key, value[]] pairs.
   */
  function groupBy(array, column) {
    const map = new Map();

    array.forEach((item) => {
      const key = item[column];
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(item);
    });

    return Array.from(map);
  }

  /**
   * Generates a URL-safe slug from a string supporting all Unicode scripts.
   * @param {string} str - The input string.
   * @param {boolean} [useFirstWords=false] - If true, limits the slug to the first 5 words.
   * @returns {string} A lowercase, alphanumeric slug.
   */
  function generateId(str, useFirstWords = false) {
    if (!str) return "item";
    const words = str
      .split(" ")
      .slice(0, useFirstWords ? 5 : undefined)
      .join(" ");
    return (
      words
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "") // Strip diacritics where applicable
        .replace(/[^\p{L}\p{N}]+/gu, "") || "item"
    );
  }

  /**
   * Extracts quotes from raw values, cleans them, and sorts them by bridging score.
   * @param {RawOpinion[]} values
   * @returns {Object[]} Sorted quotes with text, participant_id, and bridging score.
   */
  function sortAndExtractQuotes(values) {
    return (
      values
        .map((v) => {
          const demos = Object.fromEntries(
            Object.entries(v)
              .filter(([k]) => k.startsWith(demographics_prefix))
              .map(([k, val]) => [k.slice(demographics_prefix.length), val]),
          );
          return {
            index: v.index,
            text: v.quote,
            participant_id: v.participant_id,
            avg_bridging: v.AVERAGE_OF_2_BRIDGING
              ? +v.AVERAGE_OF_2_BRIDGING
              : 0,
            ...demos,
          };
        })
        // Sort descending by bridging score (highest first)
        .sort((a, b) => b.avg_bridging - a.avg_bridging)
        .filter((v) => v.text)
    );
  }

  /**
   * Transforms the flat raw opinions list into a hierarchical structure:
   * Topic -> Opinions -> Quotes.
   * Matches topics with summary data and generates unique IDs.
   *
   * @param {RawOpinion[]} opinionsList
   * @returns {Object[]} Structured topic objects.
   */
  function groupOpinions(opinionsList) {
    // 1. Group by high-level Topic
    const byTopic = groupBy(opinionsList, "topic");

    const o = byTopic.map(([topicText, topicOpinions]) => {
      // 2. Find matching AI summary (stripping markdown headers)
      const topicMatch = summary.sub_contents.find(
        (t) => stripMarkdownHeader(t.title) === topicText,
      );

      const topicId = generateId(topicText, true);

      // 3. Group by specific Opinion within the Topic
      const byOpinion = groupBy(topicOpinions, "opinion").map(
        ([_, values]) => ({
          opinionID: generateId(values[0].opinion),
          // fullID is crucial: it links the UI chart to the specific quotes list
          fullID: `${topicId}-${generateId(values[0].opinion)}`,
          text: values[0].opinion,
          count: values.length,
          quotes: sortAndExtractQuotes(values),
        }),
      );

      // 4. Sort opinions: "Other" always last, otherwise by count descending
      byOpinion.sort((a, b) => {
        if (a.text === "Other") return 1;
        if (b.text === "Other") return -1;
        return b.count - a.count;
      });

      return {
        topicID: topicId,
        summary: cleanMarkdown(topicMatch?.text),
        text: topicText,
        count: topicOpinions.length,
        opinions: byOpinion,
      };
    });

    return o;
  }

  /**
   * Creates a flat lookup list of all quotes.
   * Used for the 'quotes.json' file (lazy loading).
   * @param {Object[]} opinionsGrouped - The structured topic hierarchy.
   * @returns {Object[]} Array of { id: string, quote: string }
   */
  function flattenQuotes(opinionsGrouped) {
    const flat = [];
    opinionsGrouped.forEach((topic) => {
      topic.opinions.forEach((opinion, i) => {
        opinion.quotes.forEach((quote) => {
          const {
            index,
            text,
            participant_id,
            avg_bridging,
            fullID,
            ...demos
          } = quote;
          flat.push({
            id: opinion.fullID,
            quote: quote.text,
            ...demos,
          });
        });
      });
    });
    return flat;
  }

  /**
   * Splits summary text into paragraphs based on double newlines.
   * @param {string} text
   * @returns {string[]} Array of paragraph strings.
   */
  function parseSummary(text) {
    return text.split("\n\n").map((p) => p.trim());
  }

  /**
   * A global set to track participants included in sample previews.
   * Used to ensure diversity in the sample quotes.
   * @type {Set<string>}
   */
  const globalSampleParticipants = new Set();

  /**
   * Selects a small subset of quotes for the sample quotes.
   * Attempts to prioritize participants (Participant IDs) who haven't been featured yet
   * to maximize the diversity of voices shown in the initial view.
   *
   * @param {Object[]} topicOpinions - The opinions (with quotes) for a topic.
   * @returns {Object[]} An array of selected quote objects.
   */
  function getSampleQuotes(topicOpinions) {
    // add opinion fulID to quote objects for easier lookup
    const allQuotes = topicOpinions
      .map((o) => o.quotes.map((q) => ({ ...q, fullID: o.fullID })))
      .flat();

    // sort all quotes by avg_bridging (descending) to prioritize more "representative" quotes in the sample
    allQuotes.sort((a, b) => b.avg_bridging - a.avg_bridging);

    // loop through each opinion, and "draft" a sample quote, prioritizing quotes from participants we haven't featured yet in the globalSampleParticipants set, until we hit our options.sampleQuoteCount limit for the opinion. This way we maximize the diversity of voices shown in the sample quotes across opinions, while still prioritizing the most "representative" quotes based on bridging score.

    const selected = [];
    // loop through the number of quotes we want to show in the sample
    for (let i = 0; i < options.sampleQuoteCount; i++) {
      // Loop through the sorted quotes and find the first quote whose
      // participant (Participant ID) hasn't been featured yet in the
      // globalSampleParticipants set.
      // loop through each opinion
      for (let o of topicOpinions) {
        const possible = allQuotes.filter((q) => q.fullID === o.fullID);
        if (!possible.length) continue;
        // find the first quote in this opinion from a participant we haven't featured at all yet
        let newQuote = possible.find(
          (q) =>
            !globalSampleParticipants.has(q.participant_id) &&
            !selected.find((s) => s.participant_id === q.participant_id),
        );
        // now try someone that hasn't been featured in this opinion's sample yet, even if they have been featured in other opinions' samples
        if (!newQuote) {
          newQuote = possible.find(
            (q) => !selected.find((s) => s.participant_id === q.participant_id),
          );
        }

        // now try someone that has already been featured in this same topic
        if (!newQuote)
          newQuote = possible.find(
            (q) => !selected.find((s) => s.index === q.index),
          );

        if (newQuote) {
          selected.push({ ...newQuote });
          globalSampleParticipants.add(newQuote.participant_id);
        }
      }
    }

    // now selected should have up to opinions * options.sampleQuoteCount quotes
    return selected;
  }

  /**
   * Counts the number of unique participants (Participant IDs) in a list of opinions.
   * @param {Object[]} topicOpinions
   * @returns {number} Unique participant count.
   */
  function getUniqueQuoteCount(topicOpinions) {
    const uniqueParticipantIds = new Set();
    topicOpinions.forEach((o) => {
      o.quotes.forEach((q) => {
        uniqueParticipantIds.add(q.participant_id);
      });
    });
    return uniqueParticipantIds.size;
  }

  /**
   * Cleans and shapes raw predicted data for the mustache template.
   * @param {Object|Object[]} raw - Predicted topic objects from predicted.json.
   * @returns {Object} Cleaned predicted topic objects.
   */
  function processPredicted(raw) {
    const data = Array.isArray(raw) ? raw[0] : raw;
    if (!data || !data.sub_contents) return { text: "", topics: [] };
    return {
      text: data.text,
      topics: data.sub_contents.map((s) => ({
        topicID: generateId(s.title || "", true),
        title: stripMarkdownHeader(s.title),
        text: s.text,
        statements: (s.statements || []).map((stmt) => ({
          text: stmt.text,
          predictedAgreement: formatPercent(stmt.predicted_agreement),
          hasPredictedAgreement: stmt.predicted_agreement != null,
        })),
      })),
    };
  }

  // --- Main Execution ---

  // 1. Calculate aggregate statistics
  const byParticipant = groupBy(opinions, "participant_id");
  const totalParticipants = formatNumber(byParticipant.length);
  const totalParticipantsFormatted = formatNumber(byParticipant.length);
  const propositionsGenerated = 0; // Placeholder / Todo

  // 2. Perform transformations
  const opinionsGrouped = groupOpinions(opinions);
  const quotes = flattenQuotes(opinionsGrouped);
  const predicted = processPredicted(predictedRaw);

  // 3. Calculate high-level counts
  const topicsIdentified = summary.sub_contents.length;
  const topicsIdentifiedFormatted = formatNumber(topicsIdentified);
  const opinionsIdentified = opinionsGrouped
    .map((t) => t.opinions.length)
    .reduce((a, b) => a + b, 0);
  const opinionsIdentifiedFormatted = formatNumber(opinionsIdentified);

  // 4. Construct final payload (frontend-ready)
  const topics = opinionsGrouped.map((topic) => {
    // do a draft-style sample of quotes to make sure each opinion gets some of the "top" quotes (based on our sorting by bridging score), while also maximizing the diversity of participants shown in the sample quotes across opinions
    const allSampleQuotes = getSampleQuotes(topic.opinions);

    return {
      topicID: topic.topicID,
      text: topic.text,
      topicCount: topic.count,
      topicCountFormatted: formatNumber(topic.count),
      opinionCount: topic.opinions.length,
      opinionCountFormatted: formatNumber(topic.opinions.length),
      // rawQuoteCount: Sum of all quotes (including duplicates/same user)
      rawQuoteCount: sum(topic.opinions.map((o) => o.count)),
      // quoteCount: Unique participants
      quoteCount: getUniqueQuoteCount(topic.opinions),
      quoteCountFormatted: formatNumber(getUniqueQuoteCount(topic.opinions)),
      summary: topic.summary,
      // Map opinions to frontend structure (only sample quotes included)
      opinions: topic.opinions.map((o) => ({
        text: o.text,
        count: o.count,
        countFormatted: formatNumber(o.count),
        quotesCountFormatted: (
          i18n.sections?.quotesCount || "{count} Quotes"
        ).replace("{count}", formatNumber(o.count)),
        sampleQuotes: allSampleQuotes
          .filter((q) => q.fullID === o.fullID)
          .map((q) => q.text),
        viewAllQuotes: o.quotes.length > options.sampleQuoteCount,
        fullID: o.fullID,
      })),
    };
  });

  // Sort topics by unique quote count (most popular topics first)
  topics.sort((a, b) => b.quoteCount - a.quoteCount);

  // Construct participant overview data for chart of demographics
  const uniqueParticipants = byParticipant.map(([, rows]) => rows[0]);

  const demoKeys = Object.keys(uniqueParticipants[0] || {}).filter((k) =>
    k.startsWith(demographics_prefix),
  );

  const demographics = demoKeys.map((key) => {
    const label = key.slice(demographics_prefix.length);
    const counts = new Map();
    uniqueParticipants.forEach((p) => {
      const val = p[key];
      if (val !== undefined && val !== null && val !== "") {
        counts.set(val, (counts.get(val) || 0) + 1);
      }
    });
    let values = Array.from(counts, ([value, count]) => ({
      value,
      count,
    })).sort((a, b) => b.count - a.count);

    if (values.length > 6) {
      const otherCount = values.slice(5).reduce((acc, v) => acc + v.count, 0);
      const otherCategory = i18n.chart?.otherCategory || "Other";
      values = [
        ...values.slice(0, 5),
        { value: otherCategory, count: otherCount },
      ];
    }

    return { label, values };
  });

  demographics.sort((a, b) => a.label.localeCompare(b.label, i18n.locale));

  // 7. Prepare outputs
  const executiveSummary = parseSummary(cleanMarkdown(summary.text || ""));
  const title = stripMarkdownHeader(summary.title);

  // Dynamic sentence interpolation
  const conversationOverviewLead = (
    i18n.sections?.conversationLeadTemplate ||
    "Below is a high level overview of the topics discussed in the conversation. The most discussed topics were {topTopic1} and {topTopic2}."
  )
    .replace("{topTopic1}", topics[0]?.text || "")
    .replace("{topTopic2}", topics[1]?.text || "");

  const topicsIdentifiedBadge = (
    i18n.sections?.topicsIdentifiedBadge || "{count} topics identified"
  ).replace("{count}", topicsIdentifiedFormatted);

  const baseOutput = {
    ...options,
    title,
    executiveSummary,
    conversationOverviewLead,
    totalParticipants,
    totalParticipantsFormatted,
    topicsIdentified,
    topicsIdentifiedFormatted,
    topicsIdentifiedBadge,
    opinionsIdentified,
    opinionsIdentifiedFormatted,
    propositionsGenerated,
    topics,
    demographics,
    predicted,
    hasPredicted: predicted.topics.length > 0,
    i18n,
  };

  // --- File Writing ---

  if (!fs.existsSync(workDir)) {
    fs.mkdirSync(workDir, { recursive: true });
  }

  // A. Write flat quotes file (for lazy loading in "static" mode)
  fs.writeFileSync(
    path.join(workDir, "quotes.json"),
    JSON.stringify(quotes),
  );

  // B. Write static data (HTML payload contains topics; quotes loaded via fetch)
  const staticOutput = { ...baseOutput };
  // Escape HTML tags to prevent XSS issues when injecting into script tags
  staticOutput.payload = JSON.stringify({
    topics,
    demographics,
    options,
    i18n,
  }).replace(/</g, "\\u003c");
  fs.writeFileSync(
    path.join(workDir, "data-static.json"),
    JSON.stringify(staticOutput),
  );

  // C. Write inline data (HTML payload contains topics AND all quotes)
  const inlineOutput = { ...baseOutput };
  inlineOutput.payload = JSON.stringify({
    topics,
    demographics,
    options,
    quotes,
    i18n,
  }).replace(/</g, "\\u003c");
  fs.writeFileSync(
    path.join(workDir, "data-inline.json"),
    JSON.stringify(inlineOutput),
  );

  console.log("Data processing complete.");

  return { dataStatic: staticOutput, dataInline: inlineOutput, quotes };
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (isMain) {
  // Legacy entry used by `node data.js` / build.js tasks.data — same paths as before.
  const opinionsRaw = JSON.parse(
    fs.readFileSync("./temp/opinions.json", "utf-8"),
  );
  const predictedDefault = "./input/predicted.json";
  processReportData({
    opinionsRaw,
    summaryPath: "./input/summary.json",
    configPath: "./input/config.json",
    predictedPath: fs.existsSync(predictedDefault) ? predictedDefault : null,
    inputDir: "./input",
    packageRoot: ".",
    workDir: "./temp",
  });
}
