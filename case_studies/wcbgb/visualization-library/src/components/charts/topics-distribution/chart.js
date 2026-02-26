import * as d3 from "d3";
import { filterByTopic } from "../../helpers/filterByTopic.js";
import { processData } from "./dataProcessor.js";
import { renderVisualization } from "./renderer.js";
import { updateView } from "./viewTransitions.js";
import { getStyles } from "./styles.js";
import { createTooltip } from "../../helpers/tooltip.js";
import { extractTopicsSubContent } from "../../helpers/extractSubcontentSummary.js";

/**
 * Renders a topics distribution visualization that shows the distribution of statements across topics.
 * Supports both cluster and scatter views, with interactive features like pagination and tooltips.
 *
 * @param {HTMLElement} container - The container element to render the visualization in
 * @param {Object[]} data - Array of statement objects to visualize
 * @param {Object} theme - Theme configuration object containing colors and styles
 * @param {string} view - Current view type ('cluster' or 'scatter')
 * @param {string} [topicFilter] - Optional topic filter string
 * @param {Object} [chartOptions] - Additional chart configuration options
 * @param {Object} [summaryData] - Summary data for tooltips and additional context
 */
export function render(container, data, theme, view, topicFilter, chartOptions, summaryData) {
  // Validate required inputs
  if (!data || !data.length) {
    console.error("No data provided to render topics-distribution");
    return;
  }

  if (!theme) {
    console.error("No theme provided to render topics-distribution");
    return;
  }

  // Extract relevant sub-content from summary data
  const subcontentSummaryData = extractTopicsSubContent(summaryData);

  // Filter and process data
  const filteredData = filterByTopic(data, topicFilter);
  const uniqueTopics = new Set(
    filteredData.flatMap((d) => (Array.isArray(d.topics) ? d.topics : [d.topics]))
  );

  // Apply topic filtering if specified
  const visibleTopics = topicFilter
    ? Array.from(uniqueTopics).filter((topic) => {
        const filters = topicFilter.split(";").map((f) => f.trim().toLowerCase());
        const include = filters.filter((f) => !f.startsWith("!"));
        const exclude = filters.filter((f) => f.startsWith("!")).map((f) => f.slice(1));

        const topicLower = topic.toLowerCase();
        const matchesInclude = include.length === 0 || include.includes(topicLower);
        const matchesExclude = exclude.includes(topicLower);
        return matchesInclude && !matchesExclude;
      })
    : Array.from(uniqueTopics);

  // Calculate dimensions and layout parameters
  const numGroups = visibleTopics.length;
  const width = container.getBoundingClientRect().width;
  if (!width) return;

  const margin = 50;
  const minGroupWidth = 250;
  const minGroupHeight = 350;
  const legendHeight = 52;

  // Calculate cluster view dimensions
  const maxCols = Math.floor(width / minGroupWidth);
  const clusterColumns = Math.max(1, Math.min(maxCols, Math.ceil(Math.sqrt(numGroups))));
  const clusterRows = Math.ceil(numGroups / clusterColumns);
  const clusterHeight = clusterRows * minGroupHeight;
  const clusterInnerWidth = width;
  const clusterGroupWidth = Math.max(50, clusterInnerWidth / clusterColumns);
  const clusterGroupHeight = Math.max(50, clusterHeight / clusterRows);
  const clusterGroupPadding = Math.min(20, Math.min(clusterGroupWidth, clusterGroupHeight) / 4);
  const clusterBubbleSize = Math.max(
    20,
    Math.min(clusterGroupWidth, clusterGroupHeight) - clusterGroupPadding * 2
  );

  // Calculate scatter view dimensions
  const scatterLabelWidth = 130;
  const scatterHeight = 650;
  const minColumnWidth = 80;
  const maxColumnsRaw = Math.floor((width - scatterLabelWidth) / minColumnWidth);
  const scatterColumns = numGroups <= maxColumnsRaw ? numGroups : maxColumnsRaw;
  const maxColumns = scatterColumns;
  const needsPagination = numGroups > maxColumnsRaw;
  const scatterRows = Math.ceil(numGroups / scatterColumns);
  const scatterInnerWidth = width - scatterLabelWidth;
  const scatterInnerHeight = scatterHeight - margin * 2;
  const scatterGroupWidth = Math.max(50, scatterInnerWidth / scatterColumns);
  const scatterGroupHeight = Math.max(50, scatterInnerHeight / scatterRows);
  const scatterGroupPadding = Math.min(30, Math.min(scatterGroupWidth, scatterGroupHeight) / 4);
  const scatterBubbleSize = Math.max(
    20,
    Math.min(scatterGroupWidth, scatterGroupHeight) - scatterGroupPadding * 2
  );

  // Combine dimensions into a single configuration object
  const dimensions = {
    width,
    margin,
    views: {
      cluster: {
        height: clusterHeight,
        innerWidth: clusterInnerWidth,
        innerHeight: clusterHeight,
        columns: clusterColumns,
        rows: clusterRows,
        groupWidth: clusterGroupWidth,
        groupHeight: clusterGroupHeight,
        groupPadding: clusterGroupPadding,
        bubbleSize: clusterBubbleSize,
      },
      scatter: {
        xOffset: scatterLabelWidth,
        height: scatterHeight,
        innerWidth: scatterInnerWidth,
        innerHeight: scatterInnerHeight,
        columns: scatterColumns,
        rows: scatterRows,
        groupWidth: scatterGroupWidth,
        groupHeight: scatterGroupHeight,
        groupPadding: scatterGroupPadding,
        bubbleSize: scatterBubbleSize,
        maxColumns,
        shiftIndex: 0,
      },
    },
  };

  // Initialize visualization elements
  const style = document.createElement("style");
  style.textContent = getStyles();
  container.innerHTML = "";
  container.appendChild(style);

  // Create chart container with legend
  const chartContainer = document.createElement("div");
  chartContainer.className = "chart-container";
  chartContainer.style.height =
    view == "cluster" ? `${dimensions.views[view].height + legendHeight}px` : "auto";
  chartContainer.style.position = "relative";
  chartContainer.style.width = "100%";

  // Create and add legend
  const legend = document.createElement("div");
  legend.className = "bubble-legend";
  legend.style.display = "flex";
  legend.style.alignItems = "center";
  legend.style.margin = "0 0 12px 0";
  legend.innerHTML = `
        <div style="margin-left: 12px; display: flex; align-items: center;" aria-hidden="true">
            <span style="margin-right:12px; text-align:right; min-width:60px; font-size:12px; color:#555;">
                Fewer<br>Statements
            </span>
            <svg width="100" height="40" style="vertical-align:middle;">
                <circle cx="10" cy="20" r="10" fill="#ccc"/>
                <circle cx="40" cy="20" r="15" fill="#ccc"/>
                <circle cx="80" cy="20" r="20" fill="#ccc"/>
            </svg>
            <span style="margin-left:12px; text-align:left; min-width:60px; font-size:12px; color:#555;">
                More<br>Statements
            </span>
        </div>
    `;
  chartContainer.appendChild(legend);

  // Initialize tooltip and visualization elements
  const tooltip = createTooltip();
  let paginationControls;

  // Create SVG container
  const svg = d3
    .create("svg")
    .attr("width", "100%")
    .attr("height", `${dimensions.views[view].height}px`)
    .attr("viewBox", `0 0 ${width} ${dimensions.views[view].height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  // Set up color cycling
  const colorArray = theme.colors || ["#3B4998", "#5C6BC0", "#9FA8DA", "#C5CAE9"];
  const getColor = (index) => colorArray[index % colorArray.length];

  // Create main container and process data
  const mainContainer = svg.append("g").attr("class", "main-container");
  const { groupedData, allBubbleData } = processData(
    filteredData,
    dimensions,
    getColor,
    visibleTopics
  );

  // Render visualization elements
  const { elements, cluster, scatter } = renderVisualization({
    container: mainContainer,
    data: allBubbleData,
    groupedData,
    dimensions,
    view,
    tooltip,
    summaries: subcontentSummaryData,
  });

  // Add SVG to DOM
  chartContainer.appendChild(svg.node());

  /**
   * Creates pagination controls for the scatter view
   * @param {boolean} isScatter - Whether the current view is scatter
   */
  const createPaginationControls = (isScatter) => {
    if (paginationControls) {
      paginationControls.remove();
    }

    if (!isScatter || !needsPagination) {
      return;
    }

    paginationControls = document.createElement("div");
    paginationControls.className = "pagination-controls";
    paginationControls.style.display = "flex";
    paginationControls.style.justifyContent = "flex-end";
    paginationControls.style.gap = "10px";
    paginationControls.style.marginTop = "12px";

    // Create previous button
    const prevButton = document.createElement("button");
    prevButton.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="12" fill="#ccc"/><path d="M14.5 7L10 12L14.5 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    prevButton.style.width = "40px";
    prevButton.style.height = "40px";
    prevButton.style.border = "none";
    prevButton.style.borderRadius = "50%";
    prevButton.style.background = "none";
    prevButton.style.display = "flex";
    prevButton.style.alignItems = "center";
    prevButton.style.justifyContent = "center";
    prevButton.style.boxShadow = "0 1px 4px rgba(0,0,0,0.08)";
    prevButton.style.cursor = "pointer";
    prevButton.style.padding = "0";
    prevButton.disabled = true;
    prevButton.style.opacity = prevButton.disabled ? "1" : "0.8";
    prevButton.onmouseover = () => !prevButton.disabled && (prevButton.style.opacity = "1");
    prevButton.onmouseout = () => !prevButton.disabled && (prevButton.style.opacity = "0.8");

    // Create next button
    const nextButton = document.createElement("button");
    nextButton.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="12" fill="#222"/><path d="M9.5 7L14 12L9.5 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    nextButton.style.width = "40px";
    nextButton.style.height = "40px";
    nextButton.style.border = "none";
    nextButton.style.borderRadius = "50%";
    nextButton.style.background = "none";
    nextButton.style.display = "flex";
    nextButton.style.alignItems = "center";
    nextButton.style.justifyContent = "center";
    nextButton.style.boxShadow = "0 1px 4px rgba(0,0,0,0.08)";
    nextButton.style.cursor = "pointer";
    nextButton.style.padding = "0";
    nextButton.style.opacity = nextButton.disabled ? "1" : "0.8";
    nextButton.onmouseover = () => !nextButton.disabled && (nextButton.style.opacity = "1");
    nextButton.onmouseout = () => !nextButton.disabled && (nextButton.style.opacity = "0.8");

    /**
     * Updates the state of pagination buttons based on current position
     */
    const updateButtonStates = () => {
      prevButton.disabled = dimensions.views.scatter.shiftIndex <= 0;
      nextButton.disabled = dimensions.views.scatter.shiftIndex >= numGroups - maxColumns;

      const prevCircle = prevButton.querySelector("circle");
      prevCircle.setAttribute("fill", prevButton.disabled ? "#ccc" : "#222");
      prevButton.style.opacity = prevButton.disabled ? "1" : "0.8";
      prevButton.style.cursor = prevButton.disabled ? "default" : "pointer";

      const nextCircle = nextButton.querySelector("circle");
      nextCircle.setAttribute("fill", nextButton.disabled ? "#ccc" : "#222");
      nextButton.style.opacity = nextButton.disabled ? "1" : "0.8";
      nextButton.style.cursor = nextButton.disabled ? "default" : "pointer";
    };

    // Add click handlers for pagination
    prevButton.onclick = () => {
      if (dimensions.views.scatter.shiftIndex > 0) {
        dimensions.views.scatter.shiftIndex--;
        updateButtonStates();

        const columnWidth =
          dimensions.views.scatter.innerWidth / dimensions.views.scatter.maxColumns;
        const shiftOffset = columnWidth * dimensions.views.scatter.shiftIndex;

        // Update grid lines and labels
        mainContainer
          .selectAll(".vertical.grid-line")
          .transition()
          .duration(300)
          .attr("x1", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX - shiftOffset;
          })
          .attr("x2", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX - shiftOffset;
          })
          .style("opacity", function () {
            const originalX = d3.select(this).attr("data-original-x");
            const newX = originalX - shiftOffset;
            return newX >= 0 && newX <= dimensions.views.scatter.innerWidth ? 1 : 0;
          });

        mainContainer
          .selectAll(".topic-label")
          .transition()
          .duration(300)
          .attr("transform", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return `translate(${-shiftOffset}, 0) rotate(-45, ${originalX}, 0)`;
          })
          .style("opacity", function () {
            const originalX = d3.select(this).attr("data-original-x");
            const newX = originalX - shiftOffset;
            return newX >= 0 && newX <= dimensions.views.scatter.innerWidth ? 1 : 0;
          })
          .selectAll("tspan")
          .transition()
          .duration(300)
          .attr("x", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX;
          });

        // Update bubbles
        mainContainer
          .selectAll("circle.element")
          .transition()
          .duration(300)
          .attr("cx", (d) => d.positions.scatter.x - shiftOffset)
          .style("opacity", function (d) {
            const newX = d.positions.scatter.x - shiftOffset;
            return newX >= dimensions.views.scatter.xOffset &&
              newX <= dimensions.views.scatter.xOffset + dimensions.views.scatter.innerWidth
              ? 0.7
              : 0;
          });
      }
    };

    nextButton.onclick = () => {
      if (dimensions.views.scatter.shiftIndex < numGroups - maxColumns) {
        dimensions.views.scatter.shiftIndex++;
        updateButtonStates();

        const columnWidth =
          dimensions.views.scatter.innerWidth / dimensions.views.scatter.maxColumns;
        const shiftOffset = columnWidth * dimensions.views.scatter.shiftIndex;

        // Update grid lines and labels
        mainContainer
          .selectAll(".vertical.grid-line")
          .transition()
          .duration(300)
          .attr("x1", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX - shiftOffset;
          })
          .attr("x2", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX - shiftOffset;
          })
          .style("opacity", function () {
            const originalX = d3.select(this).attr("data-original-x");
            const newX = originalX - shiftOffset;
            return newX >= 0 && newX <= dimensions.views.scatter.innerWidth ? 1 : 0;
          });

        mainContainer
          .selectAll(".topic-label")
          .transition()
          .duration(300)
          .attr("transform", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return `translate(${-shiftOffset}, 0) rotate(-45, ${originalX}, 0)`;
          })
          .style("opacity", function () {
            const originalX = d3.select(this).attr("data-original-x");
            const newX = originalX - shiftOffset;
            return newX >= 0 && newX <= dimensions.views.scatter.innerWidth ? 1 : 0;
          })
          .selectAll("tspan")
          .transition()
          .duration(300)
          .attr("x", function () {
            const originalX = d3.select(this).attr("data-original-x");
            return originalX;
          });

        // Update bubbles
        mainContainer
          .selectAll("circle.element")
          .transition()
          .duration(300)
          .attr("cx", (d) => d.positions.scatter.x - shiftOffset)
          .style("opacity", function (d) {
            const newX = d.positions.scatter.x - shiftOffset;
            return newX >= dimensions.views.scatter.xOffset &&
              newX <= dimensions.views.scatter.xOffset + dimensions.views.scatter.innerWidth
              ? 0.7
              : 0;
          });
      }
    };

    paginationControls.appendChild(prevButton);
    paginationControls.appendChild(nextButton);
    chartContainer.appendChild(paginationControls);

    updateButtonStates();
  };

  // Initialize pagination controls
  createPaginationControls(view == "scatter");
  container.appendChild(chartContainer);

  // Store initial view and setup view update functionality
  let currentView = view;
  container._updateView = (newView) => {
    const newHeight = dimensions.views[newView].height;
    svg.attr("viewBox", `0 0 ${width} ${newHeight}`).attr("height", `${newHeight}px`);

    chartContainer.style.height = newView == "cluster" ? `${newHeight + legendHeight}px` : "auto";

    currentView = updateView({
      currentView,
      newView,
      elements,
      cluster,
      scatter,
      dimensions,
    });

    createPaginationControls(newView == "scatter");
  };

  // Setup resize handling
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

  // Setup debounced render for resize handling
  const debouncedRender = debounce(() => {
    if (isRendering) return;
    isRendering = true;

    const currentWidth = container.getBoundingClientRect().width;
    if (Math.abs(currentWidth - previousWidth) > 1) {
      if (container._cleanup) {
        container._cleanup();
      }

      container.innerHTML = "";
      render(container, data, theme, currentView, topicFilter, chartOptions, summaryData);
      previousWidth = currentWidth;
    }

    isRendering = false;
  }, 250);

  // Setup resize observer
  const resizeObserver = new ResizeObserver(() => {
    debouncedRender();
  });
  resizeObserver.observe(container);

  // Setup cleanup function
  const cleanup = () => {
    resizeObserver.disconnect();
  };
  container._cleanup = cleanup;
}
