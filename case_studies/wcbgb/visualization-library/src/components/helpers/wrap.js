import * as d3 from "d3";

/**
 * Wraps text within a specified width, breaking words at spaces.
 * Creates multiple tspan elements for each line of wrapped text.
 * 
 * @param {d3.Selection} text - D3 selection of the text element
 * @param {number} width - Maximum width for text wrapping
 * @param {number} [offset=0] - Optional line height offset multiplier
 * @returns {number} Total number of lines created
 */
export function wrap(text, width, offset = 0) {
  // Create a temporary SVG element for text measurement
  const temp = d3
    .select("body")
    .append("svg")
    .attr("class", "temporary-svg")
    .style("visibility", "hidden")
    .style("position", "absolute")
    .style("width", "0px")
    .style("height", "0px");

  let maxLineNumber = 0;

  text.each(function () {
    const text = d3.select(this);
    const words = text.text().split(/\s+/).reverse();
    const lineHeight = 1.1 + offset; // ems
    const x = text.attr("x");
    const y = text.attr("y");
    const dy = 0;

    // Create initial tspan
    let tspan = text
      .text(null)
      .append("tspan")
      .attr("x", x)
      .attr("y", y)
      .attr("dy", dy + "em")
      .attr("data-original-x", x);

    // Create temporary text element for measurement
    const tempText = temp
      .append("text")
      .style("font-size", text.style("font-size"))
      .style("font-weight", text.style("font-weight"))
      .style("font-family", text.style("font-family"));

    let line = [];
    let lineNumber = 0;

    // Process each word
    while (words.length > 0) {
      const word = words.pop();
      line.push(word);
      
      // Measure current line
      tempText.text(line.join(" "));
      
      // If line exceeds width, create new line
      if (tempText.node().getComputedTextLength() > width) {
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        
        // Create new tspan for next line
        tspan = text
          .append("tspan")
          .attr("x", x)
          .attr("y", y)
          .attr("dy", ++lineNumber * lineHeight + dy + "em")
          .attr("data-original-x", x)
          .text(word);
      } else {
        tspan.text(line.join(" "));
      }
    }

    maxLineNumber = Math.max(maxLineNumber, lineNumber);
  });

  // Clean up temporary SVG
  temp.remove();
  
  // Return total number of lines (add 1 since lineNumber is 0-based)
  return maxLineNumber + 1;
}
