import * as d3 from "d3";
import { createTooltip } from "../../helpers/tooltip.js";
import { wrap } from "../../helpers/wrap.js";
import { formatProminentThemes } from "../../helpers/formatProminentThemes.js";
import { calculateColumnPosition } from "./dataProcessor.js";

/**
 * Tooltip content for different alignment levels
 * @type {Object}
 */
const alignmentInfo = {
  highAgree:
    "<b>High alignment (Agree)</b><br/><br/>On average, 70% or more of participants agreed with statements in this subtopic.",
  low: "<b>Low alignment</b><br/><br/>Opinions were split. On average, 40–60% of voters either agreed or disagreed with statements in this subtopic.",
  highDisagree:
    "<b>High alignment (Disagree)</b><br/><br/>On average, 70% or more of participants disagreed with statements in this subtopic on average.",
};

/**
 * Renders the visualization elements for both cluster and scatter views.
 * Creates interactive elements with tooltips and handles view transitions.
 *
 * @param {Object} params - Configuration parameters
 * @param {d3.Selection} params.container - D3 selection of the container element
 * @param {Array} params.data - Array of bubble data for visualization
 * @param {Map} params.groupedData - Map of topics to their items
 * @param {Object} params.dimensions - Dimensions object containing view parameters
 * @param {string} params.view - Current view type ('cluster' or 'scatter')
 * @param {Object} params.tooltip - Tooltip instance for visualization elements
 * @param {Array} params.summaries - Array of summary data for tooltips
 * @param {Array} [params.tooltips] - Additional tooltips (unused)
 * @param {number} [params.extraTopSpace] - Additional space at the top
 * @returns {Object} Visualization elements and containers
 */
export function renderVisualization({
  container,
  data,
  groupedData,
  dimensions,
  view,
  tooltip: vizTooltip,
  summaries,
  tooltips = [],
  extraTopSpace,
}) {
  // Initialize tooltips
  const labelTooltip = createTooltip();
  const { margin } = dimensions;

  // Create main visualization container
  const vizContainer = container.append("g").attr("class", "viz-container");

  // Create scatter view container
  const scatterContainer = vizContainer
    .append("g")
    .attr("class", "scatter-view-container")
    .attr("transform", `translate(0, ${extraTopSpace})`)
    .style("opacity", view === "scatter" ? 1 : 0);

  // Create highlights for alignment regions
  const highlightsContainer = scatterContainer
    .append("g")
    .attr("class", "highlights-container")
    .attr("transform", `translate(${dimensions.views["scatter"].xOffset}, 0)`);

  const highlights = highlightsContainer.append("g").attr("class", "highlights");

  // Add high alignment (agree) region
  highlights
    .append("rect")
    .attr("class", "highlight")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", dimensions.views["scatter"].innerWidth)
    .attr("height", dimensions.views["scatter"].innerHeight * 0.3)
    .attr("fill", "#FFFBD7B2");

  highlights
    .append("line")
    .attr("class", "highlight-line")
    .attr("x1", 0)
    .attr("y1", dimensions.views["scatter"].innerHeight * 0.3)
    .attr("x2", dimensions.views["scatter"].innerWidth)
    .attr("y2", dimensions.views["scatter"].innerHeight * 0.3)
    .attr("stroke", "#606060")
    .attr("stroke-width", 1);

  // Add low alignment region
  highlights
    .append("line")
    .attr("class", "highlight-line")
    .attr("x1", 0)
    .attr("y1", dimensions.views["scatter"].innerHeight * 0.4)
    .attr("x2", dimensions.views["scatter"].innerWidth)
    .attr("y2", dimensions.views["scatter"].innerHeight * 0.4)
    .attr("stroke", "#606060")
    .attr("stroke-width", 1);

  highlights
    .append("rect")
    .attr("class", "highlight")
    .attr("x", 0)
    .attr("y", dimensions.views["scatter"].innerHeight * 0.4)
    .attr("width", dimensions.views["scatter"].innerWidth)
    .attr("height", dimensions.views["scatter"].innerHeight * 0.2)
    .attr("fill", "#ABABAB")
    .attr("opacity", 0.2);

  highlights
    .append("line")
    .attr("class", "highlight-line")
    .attr("x1", 0)
    .attr("y1", dimensions.views["scatter"].innerHeight * 0.6)
    .attr("x2", dimensions.views["scatter"].innerWidth)
    .attr("y2", dimensions.views["scatter"].innerHeight * 0.6)
    .attr("stroke", "#606060")
    .attr("stroke-width", 1);

  // Add high alignment (disagree) region
  highlights
    .append("rect")
    .attr("class", "highlight")
    .attr("x", 0)
    .attr("y", dimensions.views["scatter"].innerHeight * 0.7)
    .attr("width", dimensions.views["scatter"].innerWidth)
    .attr("height", dimensions.views["scatter"].innerHeight * 0.3)
    .attr("fill", "#FFFBD7B2");

  highlights
    .append("line")
    .attr("class", "highlight-line")
    .attr("x1", 0)
    .attr("y1", dimensions.views["scatter"].innerHeight * 0.7)
    .attr("x2", dimensions.views["scatter"].innerWidth)
    .attr("y2", dimensions.views["scatter"].innerHeight * 0.7)
    .attr("stroke", "#606060")
    .attr("stroke-width", 1);

  // Create labels container
  const scatterLabelsContainer = scatterContainer
    .append("g")
    .attr("class", "scatter-labels-container");

  // Create grid container
  const gridContainer = scatterContainer
    .append("g")
    .attr("class", "grid-container")
    .attr("transform", `translate(${dimensions.views["scatter"].xOffset}, 0)`);

  // Add vertical grid lines
  const gridLines = gridContainer.append("g").attr("class", "grid-lines");

  // Add x-axis labels container
  const xAxisLabels = gridContainer
    .append("g")
    .attr("class", "axis-labels")
    .attr("transform", `translate(0, ${dimensions.views["scatter"].innerHeight - margin / 2})`);

  // Add grid lines and labels for each topic
  const topics = Array.from(groupedData.keys());
  topics.forEach((topic, index) => {
    const xPos = calculateColumnPosition(index, dimensions) - dimensions.views["scatter"].xOffset;

    // Add vertical grid line
    gridLines
      .append("line")
      .attr("class", "vertical grid-line")
      .attr("x1", xPos)
      .attr("y1", 0)
      .attr("x2", xPos)
      .attr("y2", dimensions.views["scatter"].innerHeight)
      .attr("stroke", "#ddd")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,2")
      .attr("data-original-x", xPos);

    // Add topic label
    xAxisLabels
      .append("text")
      .attr("class", "topic-label")
      .attr("x", xPos - 20)
      .attr("y", margin - 10)
      .attr("dy", "0.7em")
      .attr("text-anchor", "end")
      .attr("transform-origin", `middle`)
      .attr("transform", `rotate(-45, ${xPos}, 0)`)
      .attr("data-original-x", xPos)
      .text(topic)
      .call(wrap, margin * 3)
      .style("font-size", "12px")
      .style("fill", "#666");
  });

  // Add alignment labels
  const scatterLabels = scatterLabelsContainer.append("g").attr("class", "labels");

  // Add top label (100% Agree)
  scatterLabels
    .append("text")
    .attr("class", "highlight-text")
    .attr("x", dimensions.views["scatter"].xOffset - 10)
    .attr("y", 0)
    .text("100% Agree")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "before-edge")
    .attr("font-size", "12px");

  // Add high alignment (agree) label
  scatterLabels
    .append("text")
    .attr("class", "highlight-text")
    .attr("x", dimensions.views["scatter"].xOffset - 10)
    .attr("y", dimensions.views["scatter"].innerHeight * 0.1)
    .text("High Alignment (Agree)")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "middle")
    .attr("font-size", "12px")
    .attr("fill", "#8D8740")
    .attr("font-weight", "700")
    .call(wrap, dimensions.views["scatter"].xOffset, 0.3)
    .call(addUnderline, { color: "#8D8740" })
    .on("mouseover", function (event) {
      labelTooltip.show(alignmentInfo.highAgree, event.clientX, event.clientY, "is-invert");
    })
    .on("mousemove", function (event) {
      labelTooltip.move(event.clientX, event.clientY);
    })
    .on("mouseout", function () {
      labelTooltip.hide();
    });

  // Add low alignment label
  scatterLabels
    .append("text")
    .attr("class", "highlight-text")
    .attr("x", dimensions.views["scatter"].xOffset - 10)
    .attr("y", dimensions.views["scatter"].innerHeight * 0.5)
    .text("Low Alignment")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "middle")
    .attr("font-size", "12px")
    .attr("fill", "#84847D")
    .attr("font-weight", "700")
    .call(wrap, dimensions.views["scatter"].xOffset, 0.5)
    .call(addUnderline, { color: "#84847D" })
    .on("mouseover", function (event) {
      labelTooltip.show(alignmentInfo.low, event.clientX, event.clientY, "is-invert");
    })
    .on("mousemove", function (event) {
      labelTooltip.move(event.clientX, event.clientY);
    })
    .on("mouseout", function () {
      labelTooltip.hide();
    });

  // Add high alignment (disagree) label
  scatterLabels
    .append("text")
    .attr("class", "highlight-text")
    .attr("x", dimensions.views["scatter"].xOffset - 10)
    .attr("y", dimensions.views["scatter"].innerHeight * 0.9)
    .text("High Alignment (Disagree)")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "middle")
    .attr("font-size", "12px")
    .attr("fill", "#8D8740")
    .attr("font-weight", "700")
    .call(wrap, dimensions.views["scatter"].xOffset, 0.3)
    .call(addUnderline, { color: "#8D8740" })
    .on("mouseover", function (event) {
      labelTooltip.show(alignmentInfo.highDisagree, event.clientX, event.clientY, "is-invert");
    })
    .on("mousemove", function (event) {
      labelTooltip.move(event.clientX, event.clientY);
    })
    .on("mouseout", function () {
      labelTooltip.hide();
    });

  // Add bottom label (0% Agree)
  scatterLabels
    .append("text")
    .attr("class", "highlight-text")
    .attr("x", dimensions.views["scatter"].xOffset - 10)
    .attr("y", dimensions.views["scatter"].innerHeight)
    .text("0% Agree")
    .attr("text-anchor", "end")
    .attr("font-size", "12px");

  // Create and style visualization elements
  const elementsContainer = vizContainer.append("g").attr("class", "elements-container");

  const elements = elementsContainer
    .selectAll("circle")
    .data(data)
    .join("circle")
    .attr("class", "element")
    .attr("r", (d) => d.positions[view].r)
    .attr("fill", (d) => d.color)
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .attr("data-name", (d) => d.name)
    .attr("data-topic", (d) => d.topic)
    .attr("cx", (d) => d.positions[view].x)
    .attr("cy", (d) => d.positions[view].y)
    .attr("opacity", (d) => (view === "scatter" ? 0.7 : 1));

  // Add tooltip event handlers
  elements
    .on("mouseover", function (event, d) {
      let prominentThemesText = "";

      if (summaries) {
        const topicSummaries = summaries.find((s) =>
          s.title.toLowerCase().includes(d.topic.toLowerCase())
        );

        const subtopicSummary =
          topicSummaries && topicSummaries.subContents
            ? topicSummaries.subContents.find((s) =>
                s.title.toLowerCase().includes(d.name.toLowerCase())
              )
            : null;

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
      }

      const tooltipContent = `
                <div class="sm-tooltip-topic">${d.topic} &gt;</div>
                <div class="sm-tooltip-subtopic">${d.name} (${d.value} statements)</div>
                ${prominentThemesText ? `<div class="sm-tooltip-subtopic-summary">${prominentThemesText}</div>` : ""}
            `;
      vizTooltip.show(tooltipContent, event.clientX, event.clientY);
    })
    .on("mousemove", function (event) {
      vizTooltip.move(event.clientX, event.clientY);
    })
    .on("mouseout", function () {
      vizTooltip.hide();
    });

  // Create cluster view container
  const clusterContainer = vizContainer
    .append("g")
    .attr("class", "cluster-view-container")
    .style("opacity", view === "cluster" ? 1 : 0);

  // Create label groups
  const clusterLabelContainer = clusterContainer.append("g").attr("class", "label-container");

  // Add headers for each group
  Array.from(groupedData.entries()).forEach(([topic, items], groupIndex) => {
    // Calculate group position
    const col = groupIndex % dimensions.views["cluster"].columns;
    const row = Math.floor(groupIndex / dimensions.views["cluster"].columns);
    const groupX =
      col * dimensions.views["cluster"].groupWidth +
      dimensions.views["cluster"].groupPadding +
      dimensions.views["cluster"].bubbleSize / 2;
    const groupY =
      row * dimensions.views["cluster"].groupHeight + dimensions.views["cluster"].groupPadding;

    // Add title for this group
    clusterLabelContainer
      .append("text")
      .attr("class", "group-title")
      .attr("data-topic", topic)
      .attr("x", groupX)
      .attr("y", groupY)
      .text(topic);

    // Get unique subtopics for this group
    const subtopics = Array.from(
      new Set(
        items.flatMap((d) => {
          if (Array.isArray(d.topics) && Array.isArray(d.subtopics)) {
            const topicIndex = d.topics.indexOf(topic);
            if (topicIndex >= 0 && d.subtopics[topicIndex]) {
              return [d.subtopics[topicIndex]];
            }
          }
          return [];
        })
      )
    ).filter(Boolean);

    // Add subtopic count
    clusterLabelContainer
      .append("text")
      .attr("class", "group-count")
      .attr("data-topic", topic)
      .attr("x", groupX)
      .attr("y", groupY + 20)
      .text(`${subtopics.length} subtopic${subtopics.length === 1 ? "" : "s"}`);

    // Add statement count
    clusterLabelContainer
      .append("text")
      .attr("class", "group-count")
      .attr("data-topic", topic)
      .attr("x", groupX)
      .attr("y", groupY + 35)
      .text(`${new Set(items.map((d) => d.id)).size} statements`);
  });

  // Setup cleanup function
  const cleanup = () => {
    vizTooltip.cleanup();
    labelTooltip.cleanup();
  };
  container._cleanup = cleanup;

  return {
    elements,
    cluster: clusterContainer,
    scatter: scatterContainer,
  };
}

/**
 * Adds an underline to text elements with optional styling.
 *
 * @param {d3.Selection} selection - D3 selection of text elements
 * @param {Object} [options] - Underline styling options
 * @param {string} [options.color="#000"] - Underline color
 * @param {number} [options.thickness=1] - Underline thickness
 * @param {number} [options.offset=-1] - Vertical offset from text
 * @param {string} [options.dasharray="2,2"] - Dash pattern for the line
 */
function addUnderline(selection, options = {}) {
  const { color = "#000", thickness = 1, offset = -1, dasharray = "2,2" } = options;

  selection.each(function () {
    const text = d3.select(this);
    const tspans = text.selectAll("tspan");

    if (tspans.empty()) return;

    requestAnimationFrame(() => {
      tspans.each(function () {
        const tspan = d3.select(this);
        const bbox = this.getBBox();

        if (bbox.width === 0 || bbox.height === 0) return;

        const parent = d3.select(text.node().parentNode);
        parent
          .insert("line", "text")
          .attr("x1", bbox.x)
          .attr("x2", bbox.x + bbox.width)
          .attr("y1", bbox.y + bbox.height + offset)
          .attr("y2", bbox.y + bbox.height + offset)
          .attr("stroke", color)
          .attr("stroke-width", thickness)
          .attr("stroke-dasharray", dasharray);
      });

      text
        .attr("paint-order", "stroke")
        .attr("stroke", "#fff")
        .attr("stroke-width", 3)
        .attr("stroke-linejoin", "round");
    });
  });
}
