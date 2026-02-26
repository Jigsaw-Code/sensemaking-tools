import * as d3 from "d3";

/**
 * Calculates the x-position for a column in the scatter plot view.
 * Takes into account the column width, shift offset, and x-offset.
 *
 * @param {number} index - The column index
 * @param {Object} dimensions - Dimensions object containing view parameters
 * @param {Object} dimensions.views - View-specific dimensions
 * @param {Object} dimensions.views.scatter - Scatter view dimensions
 * @param {number} dimensions.views.scatter.innerWidth - Inner width of the scatter view
 * @param {number} dimensions.views.scatter.maxColumns - Maximum number of columns
 * @param {number} dimensions.views.scatter.shiftIndex - Index for shifting columns
 * @param {number} dimensions.views.scatter.xOffset - X-axis offset
 * @returns {number} The calculated x-position for the column
 */
export function calculateColumnPosition(index, dimensions) {
  const columnWidth = dimensions.views.scatter.innerWidth / dimensions.views.scatter.maxColumns;
  const shiftOffset = columnWidth * dimensions.views.scatter.shiftIndex;
  return dimensions.views.scatter.xOffset + columnWidth * index + columnWidth / 2 - shiftOffset;
}

/**
 * Processes raw data into a format suitable for visualization.
 * Groups statements by topic and calculates various metrics for each topic.
 * Creates bubble data for both cluster and scatter views.
 *
 * @param {Object[]} rawData - Array of statement objects to process
 * @param {Object} dimensions - Dimensions object containing view parameters
 * @param {Function} getColor - Function to get color for a group index
 * @param {string[]} [visibleTopics] - Optional array of topics to include
 * @returns {Object} Processed data containing grouped data and bubble data
 * @returns {Map} returns.groupedData - Map of topics to their items
 * @returns {Array} returns.allBubbleData - Array of bubble data for visualization
 */
export function processData(rawData, dimensions, getColor, visibleTopics = null) {
  const { width, margin, views } = dimensions;

  // Group items by topic and subtopic
  const topicMap = new Map();
  rawData.forEach((item) => {
    const topics = Array.isArray(item.topics) ? item.topics : [item.topics];
    const subtopics = Array.isArray(item.subtopics) ? item.subtopics : [item.subtopics];

    topics.forEach((topic, index) => {
      if (!visibleTopics || visibleTopics.includes(topic)) {
        if (!topicMap.has(topic)) {
          topicMap.set(topic, []);
        }
        topicMap.get(topic).push({
          ...item,
          _currentSubtopic: subtopics[index] || "Other",
        });
      }
    });
  });

  // Initialize data structures for visualization
  const allBubbleData = [];
  const allValues = [];

  // Calculate value ranges for scaling
  Array.from(topicMap.entries()).forEach(([topic, items]) => {
    const subgroups = d3.rollup(
      items,
      (v) => v.length,
      (d) => d._currentSubtopic
    );
    subgroups.forEach((value) => {
      allValues.push(value);
    });
  });

  // Create scales for different visual properties
  const clusterRadiusScale = d3
    .scaleSqrt()
    .domain([Math.max(1, d3.min(allValues)), d3.max(allValues)])
    .range([10, 60]);

  const alignmentScale = d3.scaleLinear().domain([1, 0]).range([0, views.scatter.innerHeight]);

  const scatterRadiusScale = d3
    .scaleSqrt()
    .domain([Math.max(1, d3.min(allValues)), d3.max(allValues)])
    .range([5, views.scatter.innerWidth / (topicMap.size * 2)]);

  // Process each topic group
  Array.from(topicMap.entries()).forEach(([topic, items], groupIndex) => {
    // Calculate positions for cluster view
    const clusterView = views.cluster;
    const col = groupIndex % clusterView.columns;
    const row = Math.floor(groupIndex / clusterView.columns);
    const groupX =
      col * clusterView.groupWidth + clusterView.groupPadding + clusterView.bubbleSize / 2;
    const groupY = row * clusterView.groupHeight + clusterView.groupPadding;

    // Calculate position for scatter view
    const scatterX = calculateColumnPosition(groupIndex, dimensions);

    // Group items by subtopic and calculate metrics
    const subgroups = d3.rollup(
      items,
      (v) => ({
        count: v.length,
        alignment: d3.mean(v, (d) => d.agreeRate),
      }),
      (d) => d._currentSubtopic
    );

    // Convert to hierarchy data
    const hierarchyData = Array.from(subgroups, ([name, data]) => ({
      name,
      value: Math.max(1, data.count),
      alignment: data.alignment,
    }));

    if (hierarchyData.length === 0) return;

    // Create and pack hierarchy
    const root = d3.hierarchy({ children: hierarchyData }).sum((d) => d.value);

    const pack = d3.pack().size([clusterView.bubbleSize, clusterView.bubbleSize]).padding(0);

    // Generate initial node positions
    const initialNodes = pack(root)
      .leaves()
      .map((node) => ({
        x: node.x,
        y: node.y,
        name: node.data.name,
        value: node.data.value,
        alignment: node.data.alignment,
        r: clusterRadiusScale(node.data.value),
      }));

    // Resolve node collisions
    d3.forceSimulation(initialNodes)
      .force(
        "collide",
        d3.forceCollide((d) => d.r + 2)
      )
      .force("x", d3.forceX(clusterView.bubbleSize / 2).strength(0.05))
      .force("y", d3.forceY(clusterView.bubbleSize).strength(0.05))
      .stop()
      .tick(100);

    // Align nodes to baseline
    const topEdge = d3.min(initialNodes, (d) => d.y - d.r);
    initialNodes.forEach((d) => {
      d.y -= topEdge;
    });

    // Sort nodes by value
    initialNodes.sort((a, b) => b.value - a.value);

    // Create bubble data for each node
    initialNodes.forEach((node) => {
      const scatterY = alignmentScale(node.alignment);

      const elementData = {
        node: {
          x: node.x,
          y: node.y,
          r: node.r,
        },
        name: node.name,
        value: node.value,
        alignment: node.alignment,
        topic: topic,
        groupIndex: groupIndex,
        color: getColor(groupIndex),
        positions: {
          cluster: {
            x: groupX - clusterView.bubbleSize / 2 + node.x,
            y: groupY + 50 + node.y,
            r: node.r,
          },
          scatter: {
            x: scatterX,
            y: scatterY,
            r: scatterRadiusScale(node.value),
          },
        },
        items: items.filter((item) => item._currentSubtopic === node.name),
      };

      allBubbleData.push(elementData);
    });
  });

  return { groupedData: topicMap, allBubbleData };
}
