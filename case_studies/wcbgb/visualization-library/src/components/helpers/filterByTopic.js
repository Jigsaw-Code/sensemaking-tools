/**
 * Filters data based on topic and subtopic criteria.
 * Supports both inclusion and exclusion filters using "!" prefix.
 * Filters can be combined using semicolons (e.g., "topic1;!topic2").
 *
 * @param {Array} data - Array of data objects to filter
 * @param {string} topicFilter - Topic filter string (e.g., "topic1;!topic2")
 * @param {Function} [getTopics=(d) => d.topics] - Function to extract topics from data object
 * @param {Function} [getSubtopics=(d) => d.subtopics] - Function to extract subtopics from data object
 * @returns {Array} Filtered array of data objects
 */
export function filterByTopic(
  data,
  topicFilter,
  getTopics = (d) => d.topics,
  getSubtopics = (d) => d.subtopics
) {
  if (!topicFilter) return data;

  const filters = topicFilter.split(";").map((f) => f.trim().toLowerCase());

  const include = filters.filter((f) => !f.startsWith("!"));
  const exclude = filters.filter((f) => f.startsWith("!")).map((f) => f.slice(1));

  return data.filter((d) => {
    const topics = Array.isArray(getTopics(d)) ? getTopics(d) : [getTopics(d) || ""];
    const subtopics = Array.isArray(getSubtopics(d)) ? getSubtopics(d) : [getSubtopics(d) || ""];

    // Create topic:subtopic pairs for comparison
    const topicPairs = topics.map(
      (topic, index) => `${topic.toLowerCase()}:${subtopics[index]?.toLowerCase() || ""}`
    );

    const matchesInclude =
      include.length === 0 ||
      include.some((filter) => {
        const [filterTopic, filterSubtopic] = filter.split(":");
        return topicPairs.some((pair) => {
          const [topic, subtopic] = pair.split(":");
          return (!filterSubtopic || subtopic === filterSubtopic) && topic === filterTopic;
        });
      });

    const matchesExclude = exclude.some((filter) => {
      const [filterTopic, filterSubtopic] = filter.split(":");
      return topicPairs.some((pair) => {
        const [topic, subtopic] = pair.split(":");
        return (!filterSubtopic || subtopic === filterSubtopic) && topic === filterTopic;
      });
    });

    return matchesInclude && !matchesExclude;
  });
}
