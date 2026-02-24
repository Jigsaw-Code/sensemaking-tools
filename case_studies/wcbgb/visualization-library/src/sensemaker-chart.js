/**
 * Sensemaker Chart Web Component
 * A custom element for rendering various types of data visualizations for the Jigsaw Sensemaker tool.
 * Supports topics distribution, topics overview, and topic alignment visualizations.
 */
import "./assets/fonts/fonts.css";
import { render as renderTopicsDistribution } from "./components/charts/topics-distribution/chart.js";
import { render as renderTopicsOverview } from "./components/charts/topics-overview/chart.js";
import { render as renderTopicAlignment } from "./components/charts/topic-alignment/chart.js";
import { DataLoader } from "./loaders/data-loader.js";
import { styleVariables } from "./components/helpers/styleVariables.js";
import { globalStyle } from "./components/helpers/globalStyle.js";
import { chartStyle } from "./components/helpers/chartStyle.js";
import { downloadData } from "./components/helpers/downloadData.js";
import { generateAltText } from "./components/helpers/generateAltText.js";

const defaultTheme = {
  colors: [
    "#AFB42B",
    "#129EAF",
    "#F4511E",
    "#3949AB",
    "#5F8F35",
    "#9334E6",
    "#E52592",
    "#00897B",
    "#E8710A",
    "#1A73E8",
  ],
  fontFamily: "Noto Sans",
};

class SensemakerChart extends HTMLElement {
  static get observedAttributes() {
    return [
      "data-source",
      "summary-source",
      "chart-type",
      "colors",
      "font-family",
      "view",
      "topic-filter",
      "chart-options",
      "alt-text",
    ];
  }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    // Initialize data properties
    this._rawDataInput = null;
    this._rawSummaryInput = null;
    this._data = [];
    this._summaryData = null;

    // Initialize theme properties
    this._colors = [...defaultTheme.colors];
    this._fontFamily = defaultTheme.fontFamily;
    this._view = this.getAttribute("view");
    this._topicFilter = null;
    this._chartOptions = {};
  }

  // Property getters and setters
  set colors(value) {
    if (Array.isArray(value)) {
      this._colors = [...value];
      this.render();
    }
  }
  get colors() {
    return this._colors;
  }

  set fontFamily(value) {
    if (typeof value === "string") {
      this._fontFamily = value;
      this.render();
    }
  }
  get fontFamily() {
    return this._fontFamily;
  }

  set data(value) {
    this._rawDataInput = value;
    this.loadData();
  }
  get data() {
    return this._rawDataInput;
  }

  set summaryData(value) {
    this._rawSummaryInput = value;
    this.loadSummaryData();
  }
  get summaryData() {
    return this._rawSummaryInput;
  }

  /**
   * Parses chart options from JSON string
   */
  _parseChartOptions(optionsString) {
    if (!optionsString) return {};
    try {
      return JSON.parse(optionsString);
    } catch (error) {
      console.warn("Failed to parse chart options:", error);
      return {};
    }
  }

  /**
   * Handles attribute changes and triggers appropriate updates
   */
  async attributeChangedCallback(name, oldValue, newValue) {
    if (oldValue === newValue) return;

    switch (name) {
      case "data-source":
        await this.loadData();
        break;
      case "summary-source":
        await this.loadSummaryData();
        break;
      case "colors":
        try {
          this._colors = JSON.parse(newValue);
          this.render();
        } catch (error) {
          console.warn("Failed to parse colors:", error);
        }
        break;
      case "font-family":
        this._fontFamily = newValue;
        this.render();
        break;
      case "view":
        this._view = newValue;
        const container = this.shadowRoot.querySelector(".sm-template-container");
        if (container && container._updateView) {
          container._updateView(this._view);
        } else {
          this.render();
        }
        break;
      case "topic-filter":
        this._topicFilter = newValue;
        this.render();
        break;
      case "chart-options":
        this._chartOptions = this._parseChartOptions(newValue);
        this.render();
        break;
    }

    // Update accessibility attributes
    if (["view", "chart-type", "topic-filter", "alt-text"].includes(name)) {
      const customAlt = this.getAttribute("alt-text");
      this.setAttribute(
        "aria-label",
        customAlt ||
          generateAltText(
            this._data,
            this.getAttribute("chart-type"),
            this._view,
            this._topicFilter
          )
      );
    }
  }

  /**
   * Loads data from either direct input or source URL
   */
  async loadData() {
    if (this._rawDataInput) {
      this._data = await DataLoader.load(this._rawDataInput, true);
    } else {
      const source = this.getAttribute("data-source");
      this._data = source ? await DataLoader.load(source, true) : [];
    }
    this.render();
  }

  /**
   * Loads summary data from either direct input or source URL
   */
  async loadSummaryData() {
    if (this._rawSummaryInput) {
      this._summaryData = await DataLoader.load(this._rawSummaryInput, false);
    } else {
      const source = this.getAttribute("summary-source");
      this._summaryData = source ? await DataLoader.load(source, false) : null;
    }
    this.render();
  }

  /**
   * Initializes the component when added to DOM
   */
  async connectedCallback() {
    // Initialize properties from attributes
    this._chartOptions = this._parseChartOptions(this.getAttribute("chart-options"));
    this._view = this.getAttribute("view");
    this._topicFilter = this.getAttribute("topic-filter");
    this._fontFamily = this.getAttribute("font-family") || defaultTheme.fontFamily;

    // Initialize colors
    const colorsAttr = this.getAttribute("colors");
    this._colors = colorsAttr
      ? (() => {
          try {
            return JSON.parse(colorsAttr);
          } catch (error) {
            console.warn("Failed to parse colors:", error);
            return [...defaultTheme.colors];
          }
        })()
      : [...defaultTheme.colors];

    // Load initial data
    await this.loadData();
    await this.loadSummaryData();
  }

  /**
   * Main rendering function that creates the chart visualization
   */
  render() {
    if (!this._colors || !this._fontFamily) return;

    // Clear and setup shadow DOM
    this.shadowRoot.innerHTML = "";

    // Add styles
    const styleChart = document.createElement("style");
    styleChart.textContent = `
            ${styleVariables(this._fontFamily)}
            ${chartStyle()}
            .download-button {
                display: block;
                background-color: transparent;
                border: none;
                cursor: pointer;
                font-family: ${this._fontFamily};
                font-size: 12px;
                text-decoration: underline;
                margin-top: 1rem;
                color: #666;
                padding: 0px;
            }
        `;
    this.shadowRoot.appendChild(styleChart);

    // Add global styles
    const styleGlobal = document.createElement("style");
    styleGlobal.textContent = `
            ${styleVariables(this._fontFamily)}
            ${globalStyle()}
        `;
    document.head.appendChild(styleGlobal);

    // Create chart container
    const container = document.createElement("div");
    container.className = "sm-template-container";
    this.shadowRoot.appendChild(container);

    // Render appropriate chart type
    const chartType = this.getAttribute("chart-type");
    switch (chartType) {
      case "topics-distribution":
        renderTopicsDistribution(
          container,
          this._data,
          { colors: this._colors },
          this._view,
          this._topicFilter,
          this._chartOptions,
          this._summaryData
        );
        break;
      case "topics-overview":
        renderTopicsOverview(
          container,
          this._data,
          { colors: this._colors },
          this._chartOptions,
          this._summaryData
        );
        break;
      case "topic-alignment":
        renderTopicAlignment(
          container,
          this._data,
          { colors: this._colors },
          this._view,
          this._topicFilter,
          this._chartOptions
        );
        break;
      default:
        console.warn(`Unsupported chart type: ${chartType}`);
    }

    // Add download button
    const downloadButton = document.createElement("button");
    downloadButton.className = "download-button";
    downloadButton.textContent = "Download Data";
    downloadButton.setAttribute("tabindex", "0");
    downloadButton.setAttribute("aria-label", "Download data for this chart");
    downloadButton.addEventListener("click", () => {
      downloadData(this._data, chartType, this._view, this._topicFilter, this._summaryData);
    });
    this.shadowRoot.appendChild(downloadButton);

    // Set accessibility attributes
    const customAlt = this.getAttribute("alt-text");
    this.setAttribute(
      "aria-label",
      customAlt || generateAltText(this._data, chartType, this._view, this._topicFilter)
    );
  }
}

// Safe registration function
function registerComponent() {
  console.log("Attempting to register component...");
  console.log("Window defined:", typeof window !== "undefined");
  console.log(
    "CustomElements available:",
    typeof window !== "undefined" && !!window.customElements
  );

  if (typeof window !== "undefined" && window.customElements) {
    if (!customElements.get("sensemaker-chart")) {
      console.log("Registering sensemaker-chart component");
      customElements.define("sensemaker-chart", SensemakerChart);
      console.log("Component registered successfully");
    } else {
      console.log("Component already registered");
    }
  } else {
    console.log("Cannot register - customElements not available");
  }
}

// Self-executing function to handle registration
(function () {
  console.log("SensemakerChart initialization starting");

  // Try to register immediately
  registerComponent();

  // Also try on window load
  if (typeof window !== "undefined") {
    window.addEventListener("load", () => {
      console.log("Window load event fired");
      registerComponent();
    });
  }
})();

// Export initialization function for manual control if needed
export function init() {
  console.log("Manual init called");
  registerComponent();
  return true;
}

// For backward compatibility
export { SensemakerChart };
