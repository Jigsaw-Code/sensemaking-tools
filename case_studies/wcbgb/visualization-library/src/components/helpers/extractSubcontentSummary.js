/**
 * Extracts topic-related subcontent from summary data.
 * Finds and returns the subcontents of the "Topics" section.
 *
 * @param {Array} summaryData - Array of summary sections
 * @returns {Array|null} Array of topic subcontents if found, null otherwise
 */
export function extractTopicsSubContent(summaryData) {
  if (!Array.isArray(summaryData)) {
    console.warn("extractTopicsSubContent: summaryData is not an array.", summaryData);
    return null;
  }

  const topicsSummary = summaryData.find(
    (item) => item && item.title && typeof item.title === "string" && item.title.includes("Topics")
  );

  if (topicsSummary && topicsSummary.subContents) {
    return topicsSummary.subContents;
  } else {
    if (!topicsSummary) {
      console.warn("extractTopicsSubContent: 'Topics' section not found in summaryData.");
    }
    return null;
  }
}
