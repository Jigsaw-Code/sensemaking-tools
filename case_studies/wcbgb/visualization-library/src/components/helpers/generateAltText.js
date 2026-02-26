/**
 * Constants for alignment group labels and their test functions
 * @type {Array<{key: string, label: string, test: Function}>}
 */
const alignmentGroupLabels = [
  { key: "high", label: "High Alignment", test: (d) => d.isHighAlignment },
  { key: "low", label: "Low Alignment", test: (d) => d.isLowAlignment },
  { key: "uncertainty", label: "Pass/Unsure", test: (d) => d.isHighUncertainty },
  {
    key: "uncategorized",
    label: "Uncategorized",
    test: (d) => !d.isHighAlignment && !d.isLowAlignment && !d.isHighUncertainty,
  },
];

/**
 * Calculates statistics for alignment groups within a topic.
 *
 * @param {Array} data - Array of statement objects
 * @param {string} topicFilter - Topic to filter statements by
 * @returns {Array<{label: string, percent: number, count: number}>} Array of group statistics
 */
function getAlignmentGroupStats(data, topicFilter) {
  const topicStatements = data.filter((d) => (d.topics || []).includes(topicFilter));
  const total = topicStatements.length;
  return alignmentGroupLabels.map(({ key, label, test }) => {
    const count = topicStatements.filter(test).length;
    return {
      label,
      percent: total > 0 ? Math.round((count / total) * 100) : 0,
      count,
    };
  });
}

/**
 * Aggregates statistics for topics and subtopics.
 *
 * @param {Array} data - Array of statement objects
 * @returns {{totalStatements: number, topicCounts: Object, topicSubtopicCounts: Object}} Statistics object
 */
function getTopicAndSubtopicStats(data) {
  const totalStatements = data.length;
  const topicCounts = {};
  data.forEach((d) => {
    if (Array.isArray(d.topics)) {
      d.topics.forEach((topic) => {
        if (!topicCounts[topic]) topicCounts[topic] = 0;
        topicCounts[topic]++;
      });
    } else if (d.topics) {
      if (!topicCounts[d.topics]) topicCounts[d.topics] = 0;
      topicCounts[d.topics]++;
    }
  });
  const topicSubtopicCounts = {};
  data.forEach((d) => {
    if (Array.isArray(d.topics) && Array.isArray(d.subtopics)) {
      d.topics.forEach((topic, i) => {
        const subtopic = d.subtopics[i] || "";
        if (!topicSubtopicCounts[topic]) topicSubtopicCounts[topic] = {};
        if (!topicSubtopicCounts[topic][subtopic]) topicSubtopicCounts[topic][subtopic] = 0;
        topicSubtopicCounts[topic][subtopic]++;
      });
    }
  });
  return { totalStatements, topicCounts, topicSubtopicCounts };
}

/**
 * Formats the top N topics with their largest subtopic.
 *
 * @param {Object} topicCounts - Object mapping topics to their counts
 * @param {Object} topicSubtopicCounts - Object mapping topics to their subtopic counts
 * @param {number} totalStatements - Total number of statements
 * @param {number} [n=3] - Number of top topics to return
 * @returns {Array<string>} Array of formatted topic strings
 */
function formatTopTopics(topicCounts, topicSubtopicCounts, totalStatements, n = 3) {
  const sortedTopics = Object.entries(topicCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
  return sortedTopics.map(([topic, count]) => {
    const percent = totalStatements > 0 ? Math.round((count / totalStatements) * 100) : 0;
    const subtopics = topicSubtopicCounts[topic] || {};
    const largestSubtopic = Object.entries(subtopics).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
    return `- ${topic}, with ${percent}% of statements. Its largest subtopic was ${largestSubtopic}.`;
  });
}

/**
 * Generates alt text for different chart types and views.
 * Provides descriptive text for accessibility purposes.
 *
 * @param {Array} data - Array of statement objects
 * @param {string} chartType - Type of chart (e.g., 'topic-alignment', 'topics-distribution')
 * @param {string} view - View type (e.g., 'solid', 'waffle', 'cluster', 'scatter')
 * @param {string} [topicFilter] - Optional topic to filter by
 * @returns {string} Descriptive alt text for the chart
 */
export function generateAltText(data, chartType, view, topicFilter) {
  if (chartType === "topic-alignment" && view === "solid" && Array.isArray(data) && topicFilter) {
    const percentages = getAlignmentGroupStats(data, topicFilter);
    return `A tree map chart of the ${topicFilter} topic, depicting a percent breakdown of 4 categories: statements with High and Low Alignment, Pass/Unsure, and Uncategorized.\n\nThe High Alignment category was ${percentages[0].percent}%, Low Alignment category ${percentages[1].percent}%, Pass/unsure category ${percentages[2].percent}%, and Uncategorized category ${percentages[3].percent}%.`;
  }
  if (chartType === "topic-alignment" && view === "waffle" && Array.isArray(data) && topicFilter) {
    const percentages = getAlignmentGroupStats(data, topicFilter);
    return `A chart of the ${topicFilter} topic, depicting a percent breakdown of 4 categories: statements with High and Low Alignment, Pass/Unsure, and Uncategorized. Additionally each category is presented as a grid of squares, with each square representing an individual statement within the topic.\n\nThe High Alignment category was ${percentages[0].percent}% (or ${percentages[0].count} statements), Low Alignment category ${percentages[1].percent}% (or ${percentages[1].count} statements), Pass/unsure category ${percentages[2].percent}% (or ${percentages[2].count} statements), and Uncategorized category ${percentages[3].percent}% (or ${percentages[3].count} statements).`;
  }
  if (chartType === "topics-distribution" && view === "cluster" && Array.isArray(data)) {
    const { totalStatements, topicCounts, topicSubtopicCounts } = getTopicAndSubtopicStats(data);
    const topTopicLines = formatTopTopics(topicCounts, topicSubtopicCounts, totalStatements, 3);
    return `A breakdown of the ${totalStatements} statements into ${Object.keys(topicCounts).length} topics, encoding each subtopic's quantity of statements using circle radius. The top 3 topics were:\n\n${topTopicLines.join("\n")}`;
  }
  if (chartType === "topics-distribution" && view === "scatter" && Array.isArray(data)) {
    return `A scatter plot of the average agreement rate for statements in each topic's subtopics. Each subtopic is placed on a scale of 0% to 100% agree, on average.\n\nEach subtopic is depicted as a circle, additionally encoding its quantity of statements using radius size.`;
  }
  if (chartType === "topics-overview" && Array.isArray(data)) {
    const { totalStatements, topicCounts, topicSubtopicCounts } = getTopicAndSubtopicStats(data);
    const topTopicLines = formatTopTopics(topicCounts, topicSubtopicCounts, totalStatements, 3);
    return `A breakdown of the ${totalStatements} statements into ${Object.keys(topicCounts).length} topics, encoding the topic's quantity of statements using rectangle width. The top 3 topics were:\n\n${topTopicLines.join("\n")}`;
  }
  // Default fallback
  return `A data visualization showing data generated from the Sensemaker tools`;
}
