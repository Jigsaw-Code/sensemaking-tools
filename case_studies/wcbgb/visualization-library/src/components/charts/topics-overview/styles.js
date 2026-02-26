/**
 * Returns CSS styles for the topics overview visualization.
 * Includes styles for headers, chart content, bars, and text elements.
 * 
 * @param {Object} theme - Theme configuration object
 * @param {number} barHeight - Height of each bar segment
 * @param {number} barPadding - Padding between bars
 * @returns {string} CSS styles as a template literal
 */
export function getStyles(theme, barHeight, barPadding) {
  return `
/* Header styles */
.chart-header {
	display: flex;
	align-items: baseline;
	gap: 8px;
	margin-bottom: 4px;
}

.chart-title {
	font-size: 12px;
	font-weight: var(--sm-title-font-weight, 600);
	color: var(--sm-title-color, #333);
	margin: 0;
	text-transform: uppercase;
}

.chart-subtitle {
	font-size: var(--sm-subtitle-font-size, 12px);
	color: var(--sm-subtitle-color, #666);
	text-transform: uppercase;
	margin: 0;
}

/* Chart content */
.chart-content {
	width: 100%;
	position: relative;
	display: flex;
	margin-bottom: 12px;
}

/* Bar styles */
.chart-bar {
	transition: opacity 0.2s ease;
	cursor: pointer;
	border: 2px solid #fff;
	border-radius: 4px;
	display: flex;
	align-items: center;
}

.chart-bar:hover {
	opacity: 0.8;
}

/* Text styles */
.chart-text {
	display: inline-block;
	width: calc(100% - 16px);
	padding: 0 8px;
	color: #fff;
	font-size: 11px;
	font-weight: 600;
	pointer-events: none;
	text-overflow: ellipsis;
	overflow-x: clip;
	white-space: nowrap;
	line-height: 1;
	vertical-align: middle;
}

.chart-text.is-small {
	display: none;
}
  `;
}
