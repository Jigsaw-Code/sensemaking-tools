import { processData } from "./dataProcessor.js";
import { createWaffle, createSolid } from "./renderer.js";
import { getStyles } from "./styles.js";
import { createTooltip } from "../../helpers/tooltip.js";

/**
 * Renders the topic alignment visualization with both solid and waffle views.
 * The visualization shows the distribution of statements across different alignment categories.
 *
 * @param {HTMLElement} container - DOM element to render the visualization in
 * @param {Array<Object>} data - Array of statement objects to visualize
 * @param {Object} theme - Theme configuration for colors and styling
 * @param {string} view - Current view mode ('solid' or 'waffle')
 * @param {string} [topicFilter] - Optional topic to filter statements by
 * @param {Object} [chartOptions] - Additional chart configuration options
 */
export function render(container, data, theme, view, topicFilter, chartOptions) {
  if (!data || !data.length) {
    console.error("No data provided to render function");
    return;
  }

  if (!theme) {
    console.error("No theme provided to render function");
    return;
  }

  // Tooltip content definitions for different categories
  const tooltipInfo = {
    alignment:
      "<b>Alignment</b><br/><br/>These statements showed an especially high or especially low level of alignment amongst participants",
    high: "<b>High Alignment</b><br/><br/>70% or more of participants agreed or disagreed with these statements.",
    low: "<b>Low Alignment</b><br/><br/>Opinions were split. 40–60% of voters either agreed or disagreed with these statements.",
    uncategorized:
      "<b>Uncategorized</b><br/><br/>These statements do not meet criteria for high alignment, low alignment, or high uncertainty.",
    uncertainty:
      "<b>Uncertainty</b><br/><br/>Statements in this category were among the 25% most passed on in the conversation as a whole or were passed on by at least 20% of participants.",
  };

  const height = 500;
  const width = container.getBoundingClientRect().width;

  // Add styles
  const style = document.createElement("style");
  style.textContent = getStyles(height);

  // Process data into groups and calculate percentages
  const { filteredData, groupedData, percentages } = processData(data, topicFilter);

  // Clear container and add style
  container.innerHTML = "";
  container.appendChild(style);

  // Create chart container
  const chartContainer = document.createElement("div");
  chartContainer.className = "topic-alignment-container";
  chartContainer.style.position = "relative";

  // Create containers for both views
  const solidContainer = document.createElement("div");
  solidContainer.className = "topic-alignment-view solid-view";
  solidContainer.style.width = "100%";
  solidContainer.style.display = view === "solid" ? "block" : "none";
  if (view !== "solid") {
    solidContainer.style.position = "absolute";
    solidContainer.style.top = "0";
    solidContainer.style.left = "0";
  }

  const waffleContainer = document.createElement("div");
  waffleContainer.className = "topic-alignment-view waffle-view";
  waffleContainer.style.width = "100%";
  waffleContainer.style.display = view === "waffle" ? "block" : "none";
  if (view !== "waffle") {
    waffleContainer.style.position = "absolute";
    waffleContainer.style.top = "0";
    waffleContainer.style.left = "0";
  }

  // Create tooltips at the chart level
  const labelTooltip = createTooltip();
  const vizTooltip = createTooltip();

  // Create solid view
  const { solidViz } = createSolid({
    width,
    height,
    data: {
      percentages: {
        high: percentages.high,
        low: percentages.low,
        uncertainty: percentages.uncertainty,
        uncategorized: percentages.uncategorized,
      },
      counts: {
        high: groupedData.highAlignment.length,
        low: groupedData.lowAlignment.length,
        uncertainty: groupedData.uncertainty.length,
        uncategorized: groupedData.uncategorized.length,
      },
    },
    info: tooltipInfo,
    theme,
    labelTooltip,
  });
  solidContainer.appendChild(solidViz);

  // Create waffle view
  const { waffleViz } = createWaffle({
    width,
    data: {
      high: groupedData.highAlignment,
      low: groupedData.lowAlignment,
      uncertainty: groupedData.uncertainty,
      uncategorized: groupedData.uncategorized,
      percentages: {
        high: percentages.high,
        low: percentages.low,
        uncertainty: percentages.uncertainty,
        uncategorized: percentages.uncategorized,
      },
    },
    info: tooltipInfo,
    theme,
    labelTooltip,
    vizTooltip,
  });
  waffleContainer.appendChild(waffleViz);

  // Add both view containers to chart container
  chartContainer.appendChild(solidContainer);
  chartContainer.appendChild(waffleContainer);

  // Add chart container to main container
  container.appendChild(chartContainer);

  // Store initial view
  let currentView = view;

  // Update function for view changes
  container._updateView = (newView) => {
    currentView = newView;
    // Toggle visibility and positioning of views
    if (newView === "solid") {
      solidContainer.style.display = "block";
      solidContainer.style.position = "static";
      waffleContainer.style.display = "none";
      waffleContainer.style.position = "absolute";
    } else {
      solidContainer.style.display = "none";
      solidContainer.style.position = "absolute";
      waffleContainer.style.display = "block";
      waffleContainer.style.position = "static";
    }
  };

  // Store initial width for resize handling
  let previousWidth = container.getBoundingClientRect().width;
  let isRendering = false;

  /**
   * Debounces a function to prevent rapid repeated calls
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Debounced render function for resizing
  const debouncedRender = debounce(() => {
    if (isRendering) return;
    isRendering = true;

    const currentWidth = container.getBoundingClientRect().width;
    if (Math.abs(currentWidth - previousWidth) > 1) {
      // Clean up current visualization
      if (container._cleanup) {
        container._cleanup();
      }

      // Re-render with current view and settings
      container.innerHTML = "";
      render(container, data, theme, currentView, topicFilter, chartOptions);
      previousWidth = currentWidth;
    }

    isRendering = false;
  }, 250);

  // Add resize observer to handle container width changes
  const resizeObserver = new ResizeObserver(() => {
    debouncedRender();
  });

  resizeObserver.observe(container);

  // Cleanup function
  const cleanup = () => {
    resizeObserver.disconnect();
    labelTooltip.cleanup();
    vizTooltip.cleanup();
  };

  // Store cleanup function
  container._cleanup = cleanup;
}
