/**
 * Returns base CSS styles for chart containers.
 * Defines styles for the host element and template container.
 * 
 * @returns {string} CSS styles as a template literal
 */
export function chartStyle() {
  return `
:host {
		display: block;
		width: 100%;
	}

.sm-template-container {
	width: 100%;
	font-family: var(--sm-font-family, sans-serif);
	position: relative;
}
`;
}
