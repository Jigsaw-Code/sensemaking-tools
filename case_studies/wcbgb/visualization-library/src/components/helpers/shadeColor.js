/**
 * Adjusts the shade of a color by a specified percentage.
 * Positive percentages lighten the color, negative percentages darken it.
 *
 * @param {string} color - Hex color code (e.g., "#FF0000")
 * @param {number} percent - Percentage to adjust the color (-100 to 100)
 * @returns {string} Adjusted hex color code
 */
export function shadeColor(color, percent) {
  // Convert hex to RGB
  var R = parseInt(color.substring(1, 3), 16);
  var G = parseInt(color.substring(3, 5), 16);
  var B = parseInt(color.substring(5, 7), 16);

  // Adjust RGB values
  R = parseInt((R * (100 + percent)) / 100);
  G = parseInt((G * (100 + percent)) / 100);
  B = parseInt((B * (100 + percent)) / 100);

  // Ensure values stay within valid range
  R = R < 255 ? R : 255;
  G = G < 255 ? G : 255;
  B = B < 255 ? B : 255;

  // Round values
  R = Math.round(R);
  G = Math.round(G);
  B = Math.round(B);

  // Convert back to hex with padding
  var RR = R.toString(16).length == 1 ? "0" + R.toString(16) : R.toString(16);
  var GG = G.toString(16).length == 1 ? "0" + G.toString(16) : G.toString(16);
  var BB = B.toString(16).length == 1 ? "0" + B.toString(16) : B.toString(16);

  return "#" + RR + GG + BB;
}
