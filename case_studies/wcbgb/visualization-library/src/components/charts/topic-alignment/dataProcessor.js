/**
 * Processes and groups data for the topic alignment visualization.
 * Groups statements into high alignment, low alignment, uncertainty, and uncategorized categories.
 * Calculates percentages for each group.
 * 
 * @param {Array<Object>} data - Array of statement objects with alignment and uncertainty scores
 * @param {string} [topicFilter] - Optional topic to filter statements by
 * @returns {Object} Processed data with filtered results, grouped data, and percentages
 */
export function processData(data, topicFilter) {
    if (!data || !data.length) {
        console.error("No data provided to process");
        return {
            filteredData: [],
            groupedData: {
                uncertainty: [],
                highAlignment: [],
                lowAlignment: [],
                uncategorized: [],
            },
            percentages: {
                high: "0",
                low: "0",
                uncertainty: "0",
                uncategorized: "0",
            },
            agreementPercentages: {
                high: "0",
                low: "0",
            },
            disagreementPercentages: {
                high: "0",
                low: "0",
            },
        };
    }

    // Filter statements by topic if provided
    const filteredData = topicFilter
        ? data.filter(
            (statement) => statement.topics && 
            Array.isArray(statement.topics) && 
            statement.topics.some(topic => topic.toLowerCase() === topicFilter.toLowerCase())
        )
        : data;

    // Group statements by their alignment and uncertainty scores
    const groupedData = {
        uncertainty: [],
        highAlignment: [],
        lowAlignment: [],
        uncategorized: [],
    };

    filteredData.forEach((statement) => {
        if (statement.isHighAlignment) {
            groupedData.highAlignment.push({
                ...statement,
                alignmentType: "highAlignment",
                alignmentValue: statement.highAlignmentScore,
            });
        } else if (statement.isLowAlignment) {
            groupedData.lowAlignment.push({
                ...statement,
                alignmentType: "lowAlignment",
                alignmentValue: statement.lowAlignmentScore,
            });
        } else if (statement.isHighUncertainty) {
            groupedData.uncertainty.push({
                ...statement,
                alignmentType: "uncertainty",
                alignmentValue: statement.highUncertaintyScore,
            });
        } else {
            groupedData.uncategorized.push({
                ...statement,
                alignmentType: "uncategorized"
            });
        }
    });

    // Calculate percentages for each group
    const total = filteredData.length;
    const percentages = {
        high: ((groupedData.highAlignment.length / total) * 100).toFixed(0),
        low: ((groupedData.lowAlignment.length / total) * 100).toFixed(0),
        uncertainty: ((groupedData.uncertainty.length / total) * 100).toFixed(0),
        uncategorized: ((groupedData.uncategorized.length / total) * 100).toFixed(0),
    };

    return {
        filteredData,
        groupedData,
        percentages
    };
}
