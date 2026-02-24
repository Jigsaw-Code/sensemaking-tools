import * as d3 from "d3";

/**
 * Creates a tooltip instance with show, move, hide, and cleanup functionality.
 * Handles intelligent positioning and content display.
 *
 * @returns {Object} Tooltip instance with methods for interaction
 */
export function createTooltip() {
  // Create tooltip element
  const tooltip = d3.select("body").append("div").attr("class", "sm-tooltip");

  let hideTimeout = null;

  /**
   * Calculates optimal tooltip position based on viewport and content size
   * @param {number} x - Target X position
   * @param {number} y - Target Y position
   * @param {number} tooltipWidth - Width of tooltip content
   * @param {number} tooltipHeight - Height of tooltip content
   * @returns {{x: number, y: number, anchorClass: string}} Position and anchor class
   */
  function calculatePosition(x, y, tooltipWidth, tooltipHeight) {
    const OFFSET = 10;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let tooltipX = x + OFFSET;
    let tooltipY = y - tooltipHeight - OFFSET;
    let isRight = false;
    let isBelow = false;

    // Adjust horizontal position if it would overflow the right edge
    if (tooltipX + tooltipWidth > viewportWidth - 20) {
      tooltipX = x - tooltipWidth - OFFSET;
      isRight = true;
    }

    // Adjust vertical position if it would overflow the top edge
    if (tooltipY < 10) {
      tooltipY = y + OFFSET;
      isBelow = true;
    }

    // Ensure it's not clipped at the left
    tooltipX = Math.max(10, tooltipX);

    // Determine anchor class based on position
    let anchorClass = "bottom-left";
    if (isRight && !isBelow) anchorClass = "bottom-right";
    else if (!isRight && isBelow) anchorClass = "top-left";
    else if (isRight && isBelow) anchorClass = "top-right";

    return { x: tooltipX, y: tooltipY, anchorClass };
  }

  /**
   * Updates tooltip position and anchor class
   * @param {number} x - New X position
   * @param {number} y - New Y position
   * @param {string} anchorClass - New anchor class
   */
  function updatePosition(x, y, anchorClass) {
    // Remove previous anchor classes
    tooltip
      .classed("bottom-left", false)
      .classed("bottom-right", false)
      .classed("top-left", false)
      .classed("top-right", false);

    // Add new anchor class
    tooltip.classed(anchorClass, true);

    // Update position
    tooltip.style("left", `${x}px`).style("top", `${y}px`);
  }

  return {
    /**
     * Shows the tooltip with specified content and intelligent positioning
     * @param {string} html - HTML content to display
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {string} [c] - Optional class to add to tooltip
     */
    show: function (html, x, y, c) {
      // Clear any pending hide timeout
      if (hideTimeout) {
        clearTimeout(hideTimeout);
        hideTimeout = null;
      }

      // Set content and optional class
      tooltip.html(html);
      if (c) tooltip.classed(c, true);

      // First display the tooltip to get its dimensions
      tooltip.style("opacity", 1).style("left", "-1000px").style("top", "-1000px");

      // Get tooltip dimensions
      const tooltipNode = tooltip.node();
      const tooltipRect = tooltipNode.getBoundingClientRect();
      const { width: tooltipWidth, height: tooltipHeight } = tooltipRect;

      // Calculate optimal position
      const {
        x: tooltipX,
        y: tooltipY,
        anchorClass,
      } = calculatePosition(x, y, tooltipWidth, tooltipHeight);

      // Update position
      updatePosition(tooltipX, tooltipY, anchorClass);
    },

    /**
     * Updates the tooltip position with intelligent positioning
     * @param {number} x - X position
     * @param {number} y - Y position
     */
    move: function (x, y) {
      if (parseFloat(tooltip.style("opacity")) < 0.1) return;

      // Get tooltip dimensions
      const tooltipNode = tooltip.node();
      const tooltipRect = tooltipNode.getBoundingClientRect();
      const { width: tooltipWidth, height: tooltipHeight } = tooltipRect;

      // Calculate optimal position
      const {
        x: tooltipX,
        y: tooltipY,
        anchorClass,
      } = calculatePosition(x, y, tooltipWidth, tooltipHeight);

      // Update position
      updatePosition(tooltipX, tooltipY, anchorClass);
    },

    /**
     * Hides the tooltip with optional class removal
     * @param {string} [c] - Optional class to remove from tooltip
     */
    hide: function (c) {
      tooltip.style("opacity", 0);

      // Clear any existing timeout
      if (hideTimeout) {
        clearTimeout(hideTimeout);
      }

      // Set new timeout for class removal
      hideTimeout = setTimeout(() => {
        if (c) tooltip.classed(c, false);
        hideTimeout = null;
      }, 200); // Match the transition duration in the CSS
    },

    /**
     * Removes the tooltip from the DOM and cleans up timeouts
     */
    cleanup: function () {
      if (hideTimeout) {
        clearTimeout(hideTimeout);
        hideTimeout = null;
      }
      tooltip.remove();
    },
  };
}
