/**
 * Updates the visualization view between cluster and scatter layouts.
 * Handles transitions for elements and containers, including opacity and position changes.
 *
 * @param {Object} params - Configuration parameters
 * @param {string} params.currentView - Current view type ('cluster' or 'scatter')
 * @param {string} params.newView - Target view type ('cluster' or 'scatter')
 * @param {d3.Selection} params.elements - D3 selection of visualization elements
 * @param {d3.Selection} params.cluster - D3 selection of cluster view container
 * @param {d3.Selection} params.scatter - D3 selection of scatter view container
 * @param {Object} params.dimensions - Dimensions object containing view parameters
 * @returns {string} The new view type
 */
export function updateView({ currentView, newView, elements, cluster, scatter, dimensions }) {
  // Skip if no change
  if (newView === currentView) {
    return currentView;
  }

  console.log(`Transitioning from ${currentView} to ${newView} view`);

  if (newView === "cluster") {
    // Show cluster view, hide scatter view
    cluster.transition().style("opacity", 1);
    scatter.transition().style("opacity", 0);

    // Transition elements to cluster positions
    elements
      .transition()
      .duration(500)
      .attr("cx", (d) => d.positions[newView].x)
      .attr("cy", (d) => d.positions[newView].y)
      .attr("r", (d) => d.positions[newView].r)
      .style("opacity", 1);
  } else if (newView === "scatter") {
    // Show scatter view, hide cluster view
    cluster.transition().style("opacity", 0);
    scatter.transition().style("opacity", 1);

    // Calculate positions for scatter view
    const columnWidth = dimensions.views.scatter.innerWidth / dimensions.views.scatter.maxColumns;
    const shiftOffset = columnWidth * dimensions.views.scatter.shiftIndex;

    // Transition elements to scatter positions
    elements
      .transition()
      .duration(500)
      .attr("cx", (d) => d.positions[newView].x - shiftOffset)
      .attr("cy", (d) => d.positions[newView].y)
      .attr("r", (d) => d.positions[newView].r)
      .style("opacity", function (d) {
        const newX = d.positions.scatter.x - shiftOffset;
        return newX >= dimensions.views.scatter.xOffset &&
          newX <= dimensions.views.scatter.xOffset + dimensions.views.scatter.innerWidth
          ? 0.7
          : 0;
      });
  }

  return newView;
}
