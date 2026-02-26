/**
 * Formats prominent themes text into an HTML list.
 * Processes markdown-style text with asterisks and bold formatting.
 *
 * @param {string} text - Raw text containing prominent themes with markdown formatting
 * @returns {string} HTML formatted string with theme count and bulleted list
 */
export function formatProminentThemes(text) {
  if (!text) return "";

  // Split by bullet points and filter out empty lines
  const themes = text
    .split("*")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter((line) => !line.includes(":"))
    .map((line) => {
      // Extract the theme name (text before the colon)
      const match = line.match(/\*\*(.*?)\*\*:/);
      return match ? match[1].trim() : line;
    });

  // Create HTML unordered list
  return `<b>${themes.length} prominent themes identified:</b>
    <ul class="sm-tooltip-themes">
        ${themes.map((theme) => `<li>${theme}</li>`).join("")}
    </ul>`;
}
