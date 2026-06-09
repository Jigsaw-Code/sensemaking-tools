import "./assets/fonts/fonts.css";
import "./sensemaker-chart.js";

// Export the custom element for TypeScript support - safe for Node
export const SensemakerChart = (typeof customElements !== "undefined")
  ? customElements.get("sensemaker-chart")
  : null;

// Auto-register the component if not already registered - safe for Node
if (typeof customElements !== "undefined" && !customElements.get("sensemaker-chart")) {
  customElements.define("sensemaker-chart", SensemakerChart);
}
