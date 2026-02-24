import * as d3 from "d3";

/**
 * Processes raw data into a format suitable for topic overview visualization.
 * Groups statements by topics and subtopics, calculating counts and totals.
 *
 * @param {Array} data - Raw data array containing statements with topics and subtopics
 * @returns {Array} Processed data array with topic statistics and subtopic groupings
 */
export function processData(data) {
  // Validate input data
  if (!data || !data.length) {
    console.error("No data provided to processData");
    return [];
  }

  // Group statements by topics and their corresponding subtopics
  const topicMap = new Map();
  data.forEach((statement) => {
    const topics = Array.isArray(statement.topics) ? statement.topics : [statement.topics];
    const subtopics = Array.isArray(statement.subtopics)
      ? statement.subtopics
      : [statement.subtopics];

    topics.forEach((topic, index) => {
      if (!topicMap.has(topic)) {
        topicMap.set(topic, []);
      }
      // Add statement with its corresponding subtopic
      topicMap.get(topic).push({
        ...statement,
        _currentSubtopic: subtopics[index] || "Other",
      });
    });
  });

  // Process each topic's data into visualization format
  const processedData = Array.from(topicMap.entries()).map(([topic, statements]) => {
    // Group statements by subtopic
    const subtopics = d3.group(statements, (d) => d._currentSubtopic);

    // Calculate subtopic statistics
    const subtopicData = Array.from(subtopics.entries())
      .map(([name, group]) => ({
        name,
        value: group.length,
      }))
      .sort((a, b) => b.value - a.value);

    return {
      topic,
      subtopics: subtopicData,
      totalStatements: new Set(statements.map((d) => d.id)).size,
      subtopicCount: subtopicData.length,
    };
  });

  // Sort topics by total statement count (descending)
  processedData.sort((a, b) => b.totalStatements - a.totalStatements);

  return processedData;
}
