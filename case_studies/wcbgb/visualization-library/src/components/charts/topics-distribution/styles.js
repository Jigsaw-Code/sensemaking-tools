/**
 * Returns CSS styles for the topics distribution visualization.
 * Includes styles for containers, nodes, text elements, and interactive elements.
 *
 * @returns {string} CSS styles as a template literal
 */
export function getStyles() {
  return `
/* Container styles */
.chart-container {
		user-select: none;
}
.main-container {
		width: 100%;
		height: 100%;
}

/* Node styles */
.node {
		cursor: pointer;
}

/* Text styles */
.group-title {
		font-size: 14px;
		font-weight: 500;
		text-anchor: middle;
}
.group-count {
		font-size: 12px;
		fill: #666;
		text-anchor: middle;
}
.scatter-label {
		font-size: 12px;
		text-anchor: end;
		dominant-baseline: middle;
}

/* Element styles */
.elements-container .element {
		cursor: pointer;
		transform-origin: center center;
}
.elements-container .element:hover {
		stroke-width: 1px;
		stroke: #000;
}
.element-label {
		pointer-events: none;
		transition: opacity 0.5s ease;
		/*
		paint-order: stroke;
  	stroke: #fff;
  	stroke-width: 2px;
		*/
}
.element-label.small {
		opacity: 0;
		display: none;
}
.element-count {
		pointer-events: none;
		opacity: 0.9;
		transition: opacity 0.5s ease;
}
.element-count.small {
		opacity: 0;
		display: none;
}

/* Interactive elements */
.highlight-text {
		cursor: pointer;
}
`;
}
