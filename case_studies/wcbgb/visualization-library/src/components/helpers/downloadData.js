import { filterByTopic } from "../helpers/filterByTopic.js";
import { extractTopicsSubContent } from "./extractSubcontentSummary.js";

/**
 * Triggers a CSV file download in the browser.
 * Creates a temporary link element to handle the download.
 *
 * @param {string} csvString - The CSV content to download
 * @param {string} filename - The name of the file to download
 */
function triggerCsvDownload(csvString, filename) {
  const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(csvString);
  const link = document.createElement("a");
  link.setAttribute("href", csvContent);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Generates CSV data for topic alignment in solid view.
 * Calculates percentages of statements in different alignment groups.
 *
 * @param {Array} data - Array of statement objects
 * @param {string} topicFilter - Topic to filter statements by
 * @returns {string} CSV formatted string
 */
function csvTopicAlignmentSolid(data, topicFilter) {
  const groupLabels = [
    { key: "high", label: "High Alignment", test: (d) => d.isHighAlignment },
    { key: "low", label: "Low Alignment", test: (d) => d.isLowAlignment },
    { key: "uncertainty", label: "Uncertainty", test: (d) => d.isHighUncertainty },
    {
      key: "uncategorized",
      label: "Uncategorized",
      test: (d) => !d.isHighAlignment && !d.isLowAlignment && !d.isHighUncertainty,
    },
  ];
  let csvRows = [["topic", "group", "percentage of statements"]];
  if (topicFilter) {
    const topicStatements = data.filter((d) => (d.topics || []).includes(topicFilter));
    const total = topicStatements.length;
    groupLabels.forEach(({ key, label, test }) => {
      const groupCount = topicStatements.filter(test).length;
      if (total > 0) {
        csvRows.push([topicFilter, label, (groupCount / total).toFixed(2)]);
      }
    });
  }
  return csvRows.map((row) => row.join(",")).join("\n");
}

/**
 * Generates CSV data for topic alignment in waffle view.
 * Includes detailed statement information and voting data.
 *
 * @param {Array} data - Array of statement objects
 * @param {string} topicFilter - Topic to filter statements by
 * @returns {string} CSV formatted string
 */
function csvTopicAlignmentWaffle(data, topicFilter) {
  const groupLabels = [
    { key: "high", label: "High Alignment", test: (d) => d.isHighAlignment },
    { key: "low", label: "Low Alignment", test: (d) => d.isLowAlignment },
    { key: "uncertainty", label: "Uncertainty", test: (d) => d.isHighUncertainty },
    {
      key: "uncategorized",
      label: "Uncategorized",
      test: (d) => !d.isHighAlignment && !d.isLowAlignment && !d.isHighUncertainty,
    },
  ];
  let csvRows = [
    ["topic", "subtopic", "group", "text", "total votes", "agree", "disagree", "passed"],
  ];
  data.forEach((d) => {
    if (!d.topics) return;
    d.topics.forEach((topic, i) => {
      if (topic !== topicFilter) return;
      const subtopic = (d.subtopics && d.subtopics[i]) || "";
      const group = groupLabels.find((g) => g.test(d));
      // Sum votes across all groups
      let totalVotes = 0,
        agree = 0,
        disagree = 0,
        passed = 0;
      if (d.votes) {
        Object.values(d.votes).forEach((v) => {
          agree += v.agreeCount || 0;
          disagree += v.disagreeCount || 0;
          passed += v.passCount || 0;
        });
        totalVotes = agree + disagree + passed;
      }
      // Properly quote and escape text for CSV
      let text = d.text || "";
      text = text.replace(/\r?\n/g, " ");
      text = '"' + text.replace(/"/g, '""') + '"';
      csvRows.push([
        topic,
        subtopic,
        group ? group.label : "",
        text,
        totalVotes,
        agree,
        disagree,
        passed,
      ]);
    });
  });
  return csvRows.map((row) => row.join(",")).join("\n");
}

/**
 * Extracts prominent themes for a given topic and subtopic from summary data.
 *
 * @param {Object} summaryData - Summary data containing theme information
 * @param {string} topic - Topic to find themes for
 * @param {string} subtopic - Subtopic to find themes for
 * @returns {string} CSV-safe string of prominent themes
 */
function getProminentThemes(summaryData, topic, subtopic) {
  const subcontentSummaryData = extractTopicsSubContent(summaryData);

  if (!Array.isArray(subcontentSummaryData)) return "";
  const topicSummary = subcontentSummaryData.find(
    (t) => t.title && t.title.toLowerCase().includes(topic.toLowerCase())
  );
  if (!topicSummary || !Array.isArray(topicSummary.subContents)) return "";
  const subtopicSummary = topicSummary.subContents.find(
    (s) => s.title && s.title.toLowerCase().includes(subtopic.toLowerCase())
  );
  if (!subtopicSummary || !Array.isArray(subtopicSummary.subContents)) return "";
  const prominent = subtopicSummary.subContents.find(
    (s) => s.title && s.title.toLowerCase().includes("prominent themes")
  );
  return prominent && prominent.text
    ? '"' + prominent.text.replace(/"/g, '""').replace(/\r?\n/g, " ") + '"'
    : "";
}

/**
 * Groups data by topic and subtopic combinations.
 *
 * @param {Array} data - Array of statement objects
 * @returns {Map} Map of topic-subtopic pairs to statement arrays
 */
function groupByTopicSubtopic(data) {
  const groupMap = new Map();
  data.forEach((d) => {
    if (!d.topics || !d.subtopics) return;
    d.topics.forEach((topic, i) => {
      const subtopic = d.subtopics[i] || "";
      const key = topic + "||" + subtopic;
      if (!groupMap.has(key)) groupMap.set(key, []);
      groupMap.get(key).push(d);
    });
  });
  return groupMap;
}

/**
 * Sorts topic-subtopic keys alphabetically.
 *
 * @param {Array} keys - Array of topic-subtopic keys
 * @returns {Array} Sorted array of keys
 */
function sortTopicSubtopicKeys(keys) {
  return keys.sort((a, b) => {
    const [topicA, subA] = a.split("||");
    const [topicB, subB] = b.split("||");
    if (topicA.toLowerCase() < topicB.toLowerCase()) return -1;
    if (topicA.toLowerCase() > topicB.toLowerCase()) return 1;
    if (subA.toLowerCase() < subB.toLowerCase()) return -1;
    if (subA.toLowerCase() > subB.toLowerCase()) return 1;
    return 0;
  });
}

/**
 * Generates CSV data for topic distribution view.
 * Includes statement counts, agreement rates, and prominent themes.
 *
 * @param {Array} data - Array of statement objects
 * @param {Object} summaryData - Summary data containing theme information
 * @param {string} topicFilter - Topic to filter statements by
 * @returns {string} CSV formatted string
 */
function csvTopicDistribution(data, summaryData, topicFilter) {
  const filteredData = filterByTopic(data, topicFilter);
  const groupMap = groupByTopicSubtopic(filteredData);
  let csvRows = [
    ["topic", "subtopic", "statement count", "average agree rate", "prominent themes"],
  ];
  const sortedKeys = sortTopicSubtopicKeys(Array.from(groupMap.keys()));
  sortedKeys.forEach((key) => {
    const items = groupMap.get(key);
    const [topic, subtopic] = key.split("||");
    const count = items.length;
    const avgAgree = (items.reduce((sum, d) => sum + (d.agreeRate || 0), 0) / count).toFixed(8);
    const themes = getProminentThemes(summaryData, topic, subtopic);
    csvRows.push([topic, subtopic, count, avgAgree, themes]);
  });
  return csvRows.map((row) => row.join(",")).join("\n");
}

/**
 * Generates CSV data for topics overview.
 * Includes statement counts and prominent themes for each topic-subtopic pair.
 *
 * @param {Array} data - Array of statement objects
 * @param {Object} summaryData - Summary data containing theme information
 * @returns {string} CSV formatted string
 */
function csvTopicsOverview(data, summaryData) {
  const groupMap = groupByTopicSubtopic(data);
  let csvRows = [["topic", "subtopic", "statement count", "prominent themes"]];
  const sortedKeys = sortTopicSubtopicKeys(Array.from(groupMap.keys()));
  sortedKeys.forEach((key) => {
    const items = groupMap.get(key);
    const [topic, subtopic] = key.split("||");
    const count = items.length;
    const themes = getProminentThemes(summaryData, topic, subtopic);
    csvRows.push([topic, subtopic, count, themes]);
  });
  return csvRows.map((row) => row.join(",")).join("\n");
}

/**
 * Main function to handle data downloads for different chart types and views.
 * Generates appropriate CSV data based on chart type and view.
 *
 * @param {Array} data - Array of statement objects
 * @param {string} chartType - Type of chart (e.g., 'topic-alignment', 'topics-distribution')
 * @param {string} view - View type (e.g., 'solid', 'waffle', 'cluster', 'scatter')
 * @param {string} topicFilter - Topic to filter statements by
 * @param {Object} summaryData - Summary data containing theme information
 */
export function downloadData(data, chartType, view, topicFilter, summaryData) {
  let filename = `${chartType}${view ? `-${view}` : ""}${topicFilter ? `-${topicFilter}` : ""}.csv`;
  let csvString = "";

  switch (`${chartType}|${view}`) {
    case "topic-alignment|solid":
      if (Array.isArray(data)) {
        csvString = csvTopicAlignmentSolid(data, topicFilter);
        triggerCsvDownload(csvString, filename);
        return;
      }
      break;
    case "topic-alignment|waffle":
      if (Array.isArray(data)) {
        csvString = csvTopicAlignmentWaffle(data, topicFilter);
        triggerCsvDownload(csvString, filename);
        return;
      }
      break;
    case "topics-distribution|cluster":
    case "topics-distribution|scatter":
      if (Array.isArray(data)) {
        csvString = csvTopicDistribution(data, summaryData, topicFilter);
        triggerCsvDownload(csvString, filename);
        return;
      }
      break;
    case "topics-overview|undefined":
    case "topics-overview|":
    case "topics-overview|null":
    case "topics-overview|false":
    case "topics-overview|true":
    case "topics-overview|cluster":
    case "topics-overview|scatter":
    case "topics-overview|overview":
    case "topics-overview|summary":
    case "topics-overview|main":
      if (Array.isArray(data)) {
        csvString = csvTopicsOverview(data, summaryData);
        triggerCsvDownload(csvString, filename);
        return;
      }
      break;
    default:
      triggerCsvDownload("", filename);
      return;
  }
}
