import "../src/sensemaker-chart.js";
import { scriptWarning } from "./shared/scriptWarning.js";

const basePath =
  process.env.NODE_ENV === "production" ? "/sensemaking-tools/visualization-docs" : "";
const defaultColors = [
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
];

const getDataSource = (path) => {
  // If it's a remote URL (starts with http:// or https://), return as is
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  // Otherwise, prepend the base path for local files
  return `${basePath}${path}`;
};

export default {
  title: "Charts/TopicsDistribution",
  tags: ["autodocs"],
  argTypes: {
    dataSource: {
      name: "data-source",
      control: "text",
      description: "Local path or remote URL to the data source JSON.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
        category: "Required",
      },
    },
    summarySource: {
      name: "summary-source",
      control: "text",
      description:
        "Local path or remote URL to the summary data JSON. Optional, but required for theme summaries.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
        category: "Optional",
      },
    },
    view: {
      control: "select",
      options: ["cluster", "scatter"],
      description:
        'Display mode: "cluster" (circle packing) or "scatter" (distributed). Can be set statically or dynamically via DOM manipulation.',
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "cluster" },
      },
    },
    id: {
      control: "text",
      description:
        "Unique identifier for the chart element. Primarily used to target the chart for DOM manipulation.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
      },
    },
    topicFilter: {
      name: "topic-filter",
      control: "text",
      description:
        "Semicolon-separated list of topics to filter data. Can also prefix with '!' to exclude topics.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
        category: "Required",
      },
    },
    colors: {
      name: "colors",
      control: "object",
      description: "Array of colors to use in the chart.",
      table: {
        type: { summary: "string[]" },
        defaultValue: { summary: JSON.stringify(defaultColors) },
        category: "Style",
      },
    },
    fontFamily: {
      name: "font-family",
      control: "text",
      description: "Font family to use in the chart.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "Noto Sans" },
        category: "Style",
      },
    },
    altText: {
      name: "alt-text",
      control: "text",
      description:
        "Manually set alternative text description for accessibility purposes. This will overwrite the programmatically generated alt text.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "undefined" },
        category: "Accessibility",
      },
    },
  },
  parameters: {
    docs: {
      description: {
        component: `
The topic alignment chart displays agreement/disagreement percentages with options for different view options (cluster or scatter). The view can be set statically or dynamically via DOM manipulation.

The radius of each subtopic is determined by the number of statements in the subtopic using a square root scale.
                
${scriptWarning}
                `,
      },
    },
  },
};

// Basic template
const Template = ({
  id,
  dataSource,
  summarySource,
  view,
  topicFilter,
  colors,
  fontFamily,
  altText,
}) => {
  return `
    <sensemaker-chart
      id="${id}"
      data-source="${getDataSource(dataSource)}"
      summary-source="${getDataSource(summarySource)}"
      chart-type="topics-distribution"
      view="${view}"
      colors='${JSON.stringify(colors || defaultColors)}'
      font-family="${fontFamily || "Noto Sans"}"
      ${topicFilter ? `topic-filter="${topicFilter}"` : ""}
      ${altText ? `alt-text="${altText}"` : ""}
    ></sensemaker-chart>
  `;
};

// Example with cluster view
export const clusterView = Template.bind({});
clusterView.args = {
  dataSource: "/comments.json",
  summarySource: "/summary.json",
  view: "cluster",
  topicFilter: "!other",
};
clusterView.parameters = {
  docs: {
    description: {
      story: `The cluster view shows the subtopics as circles grouped by topic, with the size of the circle indicating the number of statements in the subtopic.`,
    },
    source: {
      code: `<sensemaker-chart
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="cluster"
  topic-filter="!other">
</sensemaker-chart>`,
      language: "html",
      type: "code",
    },
  },
};

// Example with scatter view
export const scatterView = Template.bind({});
scatterView.args = {
  dataSource: "/comments.json",
  summarySource: "/summary.json",
  view: "scatter",
  topicFilter: "!other",
};
scatterView.parameters = {
  docs: {
    description: {
      story: `The scatter view shows the subtopics as circles distributed by topic and alignment rate, with the size of the circle indicating the number of statements in the subtopic.`,
    },
    source: {
      code: `<sensemaker-chart
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="scatter"
  topic-filter="!other">
</sensemaker-chart>`,
      language: "html",
      type: "code",
    },
  },
};

// Interactive template with view toggle
const ViewToggleTemplate = ({
  dataSource,
  summarySource,
  topicFilter,
  colors,
  fontFamily,
  altText,
}) => {
  // This will run after the component is added to the DOM
  setTimeout(() => {
    const chart = document.getElementById("topics-distribution-chart-with-toggle");
    const viewInputs = document.querySelectorAll('input[name="view"]');

    // Add event listeners to view controls
    viewInputs.forEach((input) => {
      input.addEventListener("change", (e) => {
        if (e.target.checked) {
          if (e.target.value === "scatter") {
            chart.setAttribute("view", "scatter");
          } else {
            chart.setAttribute("view", "cluster");
          }
        }
      });
    });
  }, 100);

  return `
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <div class="view-controls" style="margin-bottom: 20px;">
        <label style="margin-right: 15px; cursor: pointer;">
          <input type="radio" name="view" value="cluster" checked /> Cluster View
        </label>
        <label style="cursor: pointer;">
          <input type="radio" name="view" value="scatter" /> Scatter View
        </label>
      </div>
      
      <sensemaker-chart
        id="topics-distribution-chart-with-toggle"
        data-source="${getDataSource(dataSource)}"
        summary-source="${getDataSource(summarySource)}"
        chart-type="topics-distribution"
        view="cluster"
        colors='${JSON.stringify(colors || defaultColors)}'
        font-family="${fontFamily || "Noto Sans"}"
        ${topicFilter ? `topic-filter="${topicFilter}"` : ""}
        ${altText ? `alt-text="${altText}"` : ""}
      ></sensemaker-chart>
    </div>
  `;
};

// Story with view toggle
export const WithViewToggle = ViewToggleTemplate.bind({});
WithViewToggle.args = {
  dataSource: "/comments.json",
  summarySource: "/summary.json",
  topicFilter: "!other",
};
WithViewToggle.parameters = {
  docs: {
    description: {
      story: `The chart view updates via external controls, with animated transitions that preserve DOM state.`,
    },
    source: {
      code: `<!-- Toggle controls -->
<div class="view-controls">
  <label>
    <input type="radio" name="view" value="cluster" checked /> cluster View
  </label>
  <label>
    <input type="radio" name="view" value="scatter" /> scatter View
  </label>
</div>

<!-- Chart component -->
<sensemaker-chart
  id="topics-distribution-chart-with-toggle"
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="cluster"
  topic-filter="!other">
</sensemaker-chart>

<script>
  // Add event listeners to view controls
  const chart = document.getElementById("alignment-chart-with-toggle");
  const viewInputs = document.querySelectorAll('input[name="view"]');
  
  viewInputs.forEach((input) => {
    input.addEventListener("change", (e) => {
      if (e.target.checked) {
        chart.setAttribute("view", e.target.value);
      }
    });
  });
</script>`,
      language: "html",
      type: "code",
    },
  },
};
