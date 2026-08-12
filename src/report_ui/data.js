/**
 * @fileoverview Data script for the report.
 * This script performs the ETL (Extract, Transform, Load) process:
 * 1. Loads raw opinion data, configuration, and AI-generated summaries.
 * 2. Transforms flat opinion lists into a hierarchical structure (Topics -> Opinions -> Quotes).
 * 3. Calculates statistics (participant counts, bridging scores).
 * 4. Generates formatted JSON payloads for the frontend ("static" and "inline" variations).
 */

import fs from "fs";

const args = process.argv.slice(2);
const dev = args[0] && args[0] === "dev";
const prefix = dev ? "test-" : "";
const demographics_prefix = "demo:";
/**
 * @typedef {Object} RawOpinion
 * @property {string} topic - The high-level topic category.
 * @property {string} opinion - The specific opinion text.
 * @property {string} quote - The actual quote text.
 * @property {string} participant_id - Representative ID (Participant ID).
 * @property {string|number} [AVERAGE_OF_2_BRIDGING] - Used for sorting.
 */

// --- Data Loading ---

/**
 * Raw opinions data source.
 * @type {RawOpinion[]}
 */
const opinions = JSON.parse(
  fs.readFileSync("./temp/opinions.json", "utf-8"),
).map((d, index) => ({
  ...d,
  index,
}));

const config = JSON.parse(
  fs.readFileSync(`./input/${prefix}config.json`, "utf-8"),
);

const summary = JSON.parse(
  fs.readFileSync(`./input/${prefix}summary.json`, "utf-8"),
);

const predictedPath = `./input/${prefix}predicted.json`;
const predictedRaw = fs.existsSync(predictedPath)
  ? JSON.parse(fs.readFileSync(predictedPath, "utf-8"))
  : [];

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
// --- Helper Functions ---

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
 * Formats a number with commas (e.g. 1000 -> "1,000").
 * @param {number} num
 * @returns {string}
 */
function addComma(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
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
 * Generates a URL-safe slug from a string.
 * @param {string} str - The input string.
 * @param {boolean} [useFirstWords=false] - If true, limits the slug to the first 5 words.
 * @returns {string} A lowercase, alphanumeric slug.
 */
function generateId(str, useFirstWords = false) {
  // replace anything that isn't a letter with ""
  const words = str
    .split(" ")
    .slice(0, useFirstWords ? 5 : undefined)
    .join(" ");
  return words.toLowerCase().replace(/[^a-z0-9]+/g, "");
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
          avg_bridging: v.AVERAGE_OF_2_BRIDGING ? +v.AVERAGE_OF_2_BRIDGING : 0,
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
 * @param {RawOpinion[]} opinions
 * @returns {Object[]} Structured topic objects.
 */
function groupOpinions(opinions) {
  // 1. Group by high-level Topic
  const byTopic = groupBy(opinions, "topic");

  const o = byTopic.map(([topicText, topicOpinions]) => {
    // 2. Find matching AI summary (stripping markdown headers)
    const topicMatch = summary.sub_contents.find(
      (t) => stripMarkdownHeader(t.title) === topicText,
    );

    const topicId = generateId(topicText, true);

    // 3. Group by specific Opinion within the Topic
    const byOpinion = groupBy(topicOpinions, "opinion").map(([_, values]) => ({
      opinionID: generateId(values[0].opinion),
      // fullID is crucial: it links the UI chart to the specific quotes list
      fullID: `${topicId}-${generateId(values[0].opinion)}`,
      text: values[0].opinion,
      count: values.length,
      quotes: sortAndExtractQuotes(values),
    }));

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
        const { index, text, participant_id, avg_bridging, fullID, ...demos } = quote;
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

// --- Sample Selection Logic ---

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
 * @param {Object[]} quotes - The full list of quotes for an opinion.
 * @returns {string[]} An array of selected quote texts.
 */
function getSampleQuotes(opinions) {
  // add opinion fulID to quote objects for easier lookup
  const allQuotes = opinions
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
    for (let o of opinions) {
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
        newQuote = possible.find((q) => !selected.find((s) => s.participant_id === q.participant_id));
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
 * @param {Object[]} opinions
 * @returns {number} Unique participant count.
 */
function getUniqueQuoteCount(opinions) {
  const uniqueParticipantIds = new Set();
  opinions.forEach((o) => {
    o.quotes.forEach((q) => {
      uniqueParticipantIds.add(q.participant_id);
    });
  });
  return uniqueParticipantIds.size;
}

/**
 * Cleans and shapes raw predicted data for the mustache template.
 * @param {Object[]} raw - Array of predicted topic objects from predicted.json.
 * @returns {Object[]} Cleaned predicted topic objects.
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
      statements: (s.statements || []).map((s) => ({
        text: s.text,
        predictedAgreement:
          s.predicted_agreement != null ? `${s.predicted_agreement}%` : null,
        hasPredictedAgreement: s.predicted_agreement != null,
      })),
    })),
  };
}

// --- Main Execution ---

// 1. Calculate aggregate statistics
const byParticipant = groupBy(opinions, "participant_id");
const totalParticipants = addComma(byParticipant.length);
const propositionsGenerated = 0; // Placeholder / Todo

// 2. Perform transformations
const opinionsGrouped = groupOpinions(opinions);
const quotes = flattenQuotes(opinionsGrouped);
const predicted = processPredicted(predictedRaw);

// 3. Calculate high-level counts
const topicsIdentified = summary.sub_contents.length;
const opinionsIdentified = opinionsGrouped
  .map((t) => t.opinions.length)
  .reduce((a, b) => a + b, 0);

// 4. Construct final payload (frontend-ready)
const topics = opinionsGrouped.map((topic) => {
  // do a draft-style sample of quotes to make sure each opinion gets some of the "top" quotes (based on our sorting by bridging score), while also maximizing the diversity of participants shown in the sample quotes across opinions
  const allSampleQuotes = getSampleQuotes(topic.opinions);

  return {
    topicID: topic.topicID,
    text: topic.text,
    topicCount: topic.count,
    topicCountFormatted: addComma(topic.count),
    opinionCount: topic.opinions.length,
    opinionCountFormatted: addComma(topic.opinions.length),
    // rawQuoteCount: Sum of all quotes (including duplicates/same user)
    rawQuoteCount: sum(topic.opinions.map((o) => o.count)),
    // quoteCount: Unique participants
    quoteCount: getUniqueQuoteCount(topic.opinions),
    quoteCountFormatted: addComma(getUniqueQuoteCount(topic.opinions)),
    summary: topic.summary,
    // Map opinions to frontend structure (only sample quotes included)
    opinions: topic.opinions.map((o) => ({
      text: o.text,
      count: o.count,
      countFormatted: addComma(o.count),
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
    values = [...values.slice(0, 5), { value: "Other", count: otherCount }];
  }

  return { label, values };
});

demographics.sort((a, b) => a.label - b.label);

// 7. Prepare outputs
const executiveSummary = parseSummary(cleanMarkdown(summary.text || ""));
const title = stripMarkdownHeader(summary.title);

const baseOutput = {
  ...options,
  title,
  executiveSummary,
  totalParticipants,
  topicsIdentified,
  opinionsIdentified,
  propositionsGenerated,
  topics,
  demographics,
  predicted,
  hasPredicted: predicted.topics.length > 0,
};

// --- File Writing ---

// A. Write flat quotes file (for lazy loading in "static" mode)
fs.writeFileSync("./temp/quotes.json", JSON.stringify(quotes));

// B. Write static data (HTML payload contains topics; quotes loaded via fetch)
const staticOutput = { ...baseOutput };
// Escape HTML tags to prevent XSS issues when injecting into script tags
staticOutput.payload = JSON.stringify({
  topics,
  demographics,
  options,
}).replace(/</g, "\\u003c");
fs.writeFileSync("./temp/data-static.json", JSON.stringify(staticOutput));

// C. Write inline data (HTML payload contains topics AND all quotes)
const inlineOutput = { ...baseOutput };
inlineOutput.payload = JSON.stringify({
  topics,
  demographics,
  options,
  quotes,
}).replace(/</g, "\\u003c");
fs.writeFileSync("./temp/data-inline.json", JSON.stringify(inlineOutput));

console.log("Data processing complete.");
