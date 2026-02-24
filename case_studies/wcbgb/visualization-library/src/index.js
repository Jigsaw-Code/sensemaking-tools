import "./assets/fonts/fonts.css";
import "./sensemaker-chart.js";

// Export the custom element for TypeScript support
export const SensemakerChart = customElements.get("sensemaker-chart");

// Auto-register the component if not already registered
if (!customElements.get("sensemaker-chart")) {
  customElements.define("sensemaker-chart", SensemakerChart);
}
