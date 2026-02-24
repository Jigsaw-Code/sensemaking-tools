import * as d3 from "d3";
import { processData } from "./dataProcessor.js";
import { renderTopicRow } from "./renderer.js";
import { getStyles } from "./styles.js";
import { createTooltip } from "../../helpers/tooltip.js";
import { extractTopicsSubContent } from "../../helpers/extractSubcontentSummary.js";

/**
 * Renders a stacked bar chart visualization for topic overview.
 * Creates interactive bars with tooltips and handles data processing.
 *
 * @param {HTMLElement} container - Container element for the visualization
 * @param {Array} data - Raw data array containing topic information
 * @param {Object} theme - Theme configuration object
 * @param {string} title - Chart title
 * @param {Object} summaryData - Summary data for tooltips
 * @returns {Function} Cleanup function
 */
export function render(container, data, theme, title, summaryData) {
  // Validate required inputs
  if (!data || !data.length) {
    console.error("No data provided to renderStackedBars");
    return;
  }

  if (!theme) {
    console.error("No theme provided to renderStackedBars");
    return;
  }

  // Extract relevant sub-content from summary data
  const subcontentSummaryData = extractTopicsSubContent(summaryData);

  // Initialize chart dimensions and styling
  const margin = { top: 0, right: 0, bottom: 0, left: 0 };
  const width = 1000; // Fixed width that will be scaled by viewBox
  const barHeight = (theme.bars && theme.bars.barHeight) || 30;
  const barPadding = (theme.bars && theme.bars.barPadding) || 20;
  const segmentPadding = 2; // Space between segments
  const cornerRadius = 4; // Rounded corners radius
  const colorArray = theme.colors || ["#FFE0B2", "#FFCDD2", "#B3E5FC"];

  // Add chart styles
  const style = document.createElement("style");
  style.textContent = getStyles(theme, barHeight, barPadding);
  container.appendChild(style);

  // Initialize tooltip
  const tooltip = createTooltip();

  // Create chart container
  const chartContainer = document.createElement("div");
  chartContainer.className = "stacked-bar-container";

  // Process and prepare data
  const processedData = processData(data);
  const maxValue = d3.max(processedData, (d) => d.totalStatements);

  // Render topic rows
  processedData.forEach((topicData, index) => {
    const row = renderTopicRow({
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
      summaries: subcontentSummaryData,
    });

    container.appendChild(row);
  });

  container.appendChild(chartContainer);

  // Return cleanup function
  return function cleanup() {
    // tooltip.cleanup();
  };
}
