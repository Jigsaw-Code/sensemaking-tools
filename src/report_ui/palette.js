/**
 * @fileoverview Sequential color palette generation for the report UI.
 *
 * Builds a light-to-dark palette from a single base hex color using CSS
 * Relative Color Syntax (`oklch(from <color> l c h)`), delegating color-space
 * conversion and gamut mapping to the browser's CSS engine.
 */

/**
 * Number of colors generated for the demographic chart. The chart shows at
 * most five values plus "Other" (see data.js), so six colors are enough.
 * @type {number}
 */
export const DEMOGRAPHIC_PALETTE_STEPS = 6;

/**
 * Base color for the default demographic palette.
 * @type {string}
 */
export const DEFAULT_DEMOGRAPHIC_BASE_COLOR = "#4886f7";

/**
 * Lightest OKLCH lightness used in generated palettes.
 *
 * Palettes deliberately use a wide lightness range: lightness differences are
 * what keep segments distinguishable for color-blind readers. The lightest
 * segment is therefore a tint with less than 3:1 contrast against the chart
 * background; segments are separated by gaps and labelled in the legend, so
 * color is not the only way to tell values apart.
 * @type {number}
 */
export const PALETTE_MAX_LIGHTNESS = 0.85;

/**
 * Darkest OKLCH lightness used in generated palettes. Dark enough for a strong
 * end to the ramp while staying visibly tinted rather than black.
 * @type {number}
 */
export const PALETTE_MIN_LIGHTNESS = 0.22;

/**
 * Fraction of the base color's chroma kept at PALETTE_MAX_LIGHTNESS. Steps
 * lighter than the base taper towards this, so pale steps are soft tints
 * rather than fluorescent.
 * @type {number}
 */
export const LIGHTEST_CHROMA_FRACTION = 0.5;

const HEX_COLOR_RE = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i;

/**
 * Returns whether `value` is a `#rgb` or `#rrggbb` hex color string.
 * @param {unknown} value
 * @returns {boolean}
 */
export function isHexColor(value) {
  return typeof value === "string" && HEX_COLOR_RE.test(value.trim());
}

/**
 * Generates a light-to-dark palette of CSS `oklch(from ...)` strings from a
 * single base hex color.
 *
 * Every step keeps the base hue (`h`), spaces lightness (`targetL`) evenly
 * from PALETTE_MAX_LIGHTNESS down to PALETTE_MIN_LIGHTNESS, and scales the
 * base chroma (`c`) in CSS:
 * - Steps darker than the base scale chroma proportionally (`targetL / l`).
 * - Steps lighter than the base taper chroma linearly toward
 *   `LIGHTEST_CHROMA_FRACTION * c` at `PALETTE_MAX_LIGHTNESS`.
 *
 * @param {string} [baseHex=DEFAULT_DEMOGRAPHIC_BASE_COLOR] - `#rgb` or `#rrggbb`.
 * @param {number} [steps=DEMOGRAPHIC_PALETTE_STEPS] - Number of steps (>= 2).
 * @returns {string[]} CSS `oklch(from ...)` color strings, lightest to darkest.
 * @throws {Error} If `baseHex` is not a valid hex color or `steps` < 2.
 */
export function generateSequentialPalette(
  baseHex = DEFAULT_DEMOGRAPHIC_BASE_COLOR,
  steps = DEMOGRAPHIC_PALETTE_STEPS,
) {
  if (!isHexColor(baseHex)) {
    throw new Error(`Invalid hex color: ${baseHex}`);
  }
  if (!Number.isInteger(steps) || steps < 2) {
    throw new Error(`steps must be an integer >= 2, got ${steps}`);
  }

  const base = baseHex.trim().toLowerCase();
  const lightTaper = +(1 - LIGHTEST_CHROMA_FRACTION).toFixed(4);

  return Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1);
    const targetL = +(
      PALETTE_MAX_LIGHTNESS -
      t * (PALETTE_MAX_LIGHTNESS - PALETTE_MIN_LIGHTNESS)
    ).toFixed(4);

    const darkScale = `${targetL} / max(l, 0.001)`;
    const lightProgress = `clamp(0, (${targetL} - l) / max(0.001, ${PALETTE_MAX_LIGHTNESS} - l), 1)`;
    const lightScale = `1 - ${lightTaper} * ${lightProgress}`;

    return `oklch(from ${base} ${targetL} calc(c * min(${darkScale}, ${lightScale})) h)`;
  });
}

/**
 * Resolves the `demographic_colors` config value into a palette.
 *
 * Accepts an explicit array of colors (used as-is), a single hex color (a
 * palette is generated from it), or nothing (a palette is generated from
 * DEFAULT_DEMOGRAPHIC_BASE_COLOR).
 *
 * @param {string|string[]|undefined} configValue
 * @returns {string[]}
 */
export function resolveDemographicColors(configValue) {
  if (Array.isArray(configValue)) return configValue;
  if (isHexColor(configValue)) return generateSequentialPalette(configValue);
  if (configValue != null && configValue !== "") {
    console.warn(
      `Warning: demographic_colors must be an array or a hex color such as ` +
        `"${DEFAULT_DEMOGRAPHIC_BASE_COLOR}"; got ` +
        `${JSON.stringify(configValue)}. Using default colors.`,
    );
  }
  return generateSequentialPalette(DEFAULT_DEMOGRAPHIC_BASE_COLOR);
}
