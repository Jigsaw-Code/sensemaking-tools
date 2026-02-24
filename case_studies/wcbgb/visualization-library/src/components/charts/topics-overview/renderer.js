import * as d3 from "d3";

import { formatProminentThemes } from "../../helpers/formatProminentThemes.js";

/**
 * Renders a single topic row with stacked bar segments for subtopics.
 * Creates interactive elements with tooltips showing detailed information.
 *
 * @param {Object} params - Configuration parameters
 * @param {HTMLElement} params.container - Container element for the row
 * @param {Object} params.topicData - Processed data for the topic
 * @param {number} params.index - Index of the topic for color selection
 * @param {Array} params.colorArray - Array of colors for topics
 * @param {Object} params.tooltip - Tooltip instance
 * @param {number} params.width - Width of the visualization
 * @param {number} params.barHeight - Height of each bar segment
 * @param {number} params.cornerRadius - Radius for rounded corners
 * @param {number} params.segmentPadding - Padding between segments
 * @param {number} params.maxValue - Maximum value for scaling
 * @param {Array} params.summaries - Summary data for tooltips
 * @returns {HTMLElement} The rendered row container
 */
export function renderTopicRow({
  container,
  topicData,
  index,
  colorArray,
  tooltip,
  width,
  barHeight,
  cornerRadius,
  segmentPadding,
  maxValue,
  summaries,
}) {
  // Create row container
  const rowContainer = document.createElement("div");
  rowContainer.className = "chart-row";

  // Create header with title and subtitle
  const headerContainer = document.createElement("div");
  headerContainer.className = "chart-header";

  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = topicData.topic;

  const subtitle = document.createElement("div");
  subtitle.className = "chart-subtitle";
  subtitle.textContent = `(${topicData.subtopicCount} subtopics, ${topicData.totalStatements} total statements)`;

  headerContainer.appendChild(title);
  headerContainer.appendChild(subtitle);
  rowContainer.appendChild(headerContainer);

  // Create content container for bars
  const content = d3.select(rowContainer).append("div").attr("class", "chart-content");

  // Get color for this topic
  const topicColor = colorArray[index % colorArray.length];

  // Create scale for bar widths
  const xScale = d3.scaleLinear().domain([0, maxValue]).range([0, width]);

  // Create stacked bar segments
  let cumulative = 0;
  topicData.subtopics.forEach((subtopic) => {
    const segmentWidth = xScale(subtopic.value);

    // Create bar segment
    const bar = content
      .append("div")
      .attr("class", "chart-bar")
      .style("width", `${Math.max(0, segmentWidth)}px`)
      .style("height", `${barHeight}px`)
      .style("background", topicColor)
      .on("mouseover", function (event) {
        // Find relevant summary data
        const topicSummaries = summaries.find((s) =>
          s.title.toLowerCase().includes(topicData.topic.toLowerCase())
        );

        const subtopicSummary =
          topicSummaries && topicSummaries.subContents
            ? topicSummaries.subContents.find((s) =>
                s.title.toLowerCase().includes(subtopic.name.toLowerCase())
              )
            : null;

        // Extract prominent themes if available
        let prominentThemesText = "";
        if (
          topicSummaries &&
          topicSummaries.subContents &&
          subtopicSummary &&
          subtopicSummary.subContents
        ) {
          const prominentThemes = subtopicSummary.subContents.find(
            (s) => s.title && s.title.toLowerCase().includes("prominent themes")
          );
          if (prominentThemes && prominentThemes.text) {
            prominentThemesText = formatProminentThemes(prominentThemes.text);
          }
        }

        // Show tooltip with topic and subtopic information
        const tooltipContent = `
                    <div class="sm-tooltip-topic">${topicData.topic} &gt;</div>
                    <div class="sm-tooltip-subtopic">${subtopic.name} (${subtopic.value} statements)</div>
                    ${prominentThemesText ? `<div class="sm-tooltip-subtopic-summary">${prominentThemesText}</div>` : ""}
                `;
        tooltip.show(tooltipContent, event.clientX, event.clientY);
      })
      .on("mousemove", function (event) {
        tooltip.move(event.clientX, event.clientY);
      })
      .on("mouseout", function () {
        tooltip.hide();
      });

    // Add subtopic label
    bar
      .append("span")
      .attr("class", "chart-text")
      .text(subtopic.name)
      .classed("is-small", segmentWidth < 64);

    cumulative += subtopic.value;
  });

  return rowContainer;
}
