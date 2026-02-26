/**
 * Returns CSS custom properties (variables) for styling components.
 * Defines font families, sizes, weights, and colors used throughout the application.
 *
 * @param {string} [fontFamily="Noto Sans"] - Primary font family to use
 * @returns {string} CSS custom properties as a template literal
 */
export function styleVariables(fontFamily = "Noto Sans") {
  return `
.sm-template-container {
	/* Font settings */
	--sm-font-family: ${fontFamily}, Arial, sans-serif;
	--sm-title-font-size: 18px;
	--sm-title-font-weight: 600;
	--sm-title-color: #333;
	--sm-subtitle-color: #666;
	--sm-font-size: 14px;
	
	/* Label settings */
	--sm-label-font-size: 12px;
	--sm-label-font-weight: 500;
	--sm-label-color: #333;
	
	/* Value settings */
	--sm-value-font-size: 11px;
	--sm-value-font-weight: 400;
	--sm-value-color: #666;
	
	/* Theme colors */
	--sm-color-primary: #0B57D0;
}
	`;
}
