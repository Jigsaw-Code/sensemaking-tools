/**
 * Returns CSS styles for the topic alignment visualization components.
 * Includes styles for both solid and waffle views, as well as responsive behavior.
 * @returns {string} CSS styles as a template literal
 */
export function getStyles() {
  return `
/* Container styles */
.topic-alignment-container {
	width: 100%;
	display: flex;
	gap: 5px;
	position: relative;
}

.viz-container {
	position: relative;
	flex-direction: column;
	transition: background 0.4s ease;
}

/* Grid layouts */
.solid-grid,
.waffle-grid {
	display: flex;
	justify-content: space-between;
	flex-direction: column;
	gap: 5px;
	position: relative;
}

.solid-grid {
	height: 500px;
}

.waffle-grid {
	gap: 25px;
}

/* Group layouts */
.solid-group,
.waffle-group {
	width: 100%;
	display: flex;
	gap: 10px;
}

.solid-group.alignment,
.waffle-group.alignment {
	padding-top: 24px;
}

.solid-group.is-short {
	align-items: center;
}

/* Group boxes */
.solid-group-box,
.waffle-group-box {
	position: relative;
	height: 100%;
	width: 100%;
	display: flex;
	gap: 5px;
}

.waffle-group-box {
	gap: 25px;
	justify-content: space-between;
}

/* Labels */
.group-label {
	min-width: 100px;	
	height: fit-content;
	text-align: right;
	font-size: 14px;
	text-decoration: underline;
	text-decoration-style: dotted;
	text-underline-offset: 2px;
	cursor: pointer;
}

.section-label {
	position: absolute;
	top: -28px;
	left: 0;
	font-size: 14px;
	font-weight: 500;
	text-decoration: underline;
	text-decoration-style: dotted;
	text-underline-offset: 2px;
	cursor: pointer;
}

/* Solid view specific */
.solid-group-box-section {
	border: 1px solid #ddd;
	border-radius: 8px;
	position: relative;
	height: 100%;
}

.solid-group-box-section-text {
	padding: 8px;
}

.solid-group-box-section-text-subtitle {
	font-size: 12px;
	opacity: 0.9;
}

/* Responsive styles */
.is-narrow .solid-group-box-section-text .percentage {
	font-size: 18px;
}

.is-narrow .solid-group-box-section-text-subtitle {
	display: none;
}

.is-hidden .solid-group-box-section-text {
	display: none;
}

/* Waffle view specific */
.waffle-group-box-section {
	position: relative;
	height: 100%;
	display: flex;
	flex-wrap: wrap;
}

.waffle-square {
	width: 25px;
	height: 25px;
	transition: background-color 0.2s ease, outline 0.2s ease;
	position: relative;
	box-sizing: border-box;
	border: 1px solid #fff;
}

.waffle-square.invert {
	box-shadow: inset 0 0 0 1px #00000060; 
}

.waffle-square:hover {
	border: 2px solid #000;
	box-shadow: none;
}

/* Visualization containers */
.waffle-viz, 
.solid-viz {
	margin-top: 1rem;
	width: 100%;
}
`;
}
