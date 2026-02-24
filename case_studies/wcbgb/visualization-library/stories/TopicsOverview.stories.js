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
  title: "Charts/TopicsOverview",
  tags: ["autodocs"],
  argTypes: {
    dataSource: {
      name: "data-source",
      control: "text",
      description: "Local or remote URL to the data source CSV. HTML attribute: `data-source`",
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
    id: {
      control: "text",
      description:
        "Unique identifier for the chart element. Primarily used to target the chart for DOM manipulation. HTML attribute: `id`",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
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
The topics overview chart displays the number of statements per topic and subtopic as a stacked bar chart.
                
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
      chart-type="topics-overview"
      colors='${JSON.stringify(colors || defaultColors)}'
      font-family="${fontFamily || "Noto Sans"}"
      ${altText ? `alt-text="${altText}"` : ""}
    ></sensemaker-chart>
  `;
};

// Example with solid view
export const Base = Template.bind({});
Base.args = {
  id: "topics-overview-chart",
  dataSource: "/comments.json",
  summarySource: "/summary.json",
};
Base.parameters = {
  docs: {
    description: {
      story: ``,
    },
    source: {
      code: `<sensemaker-chart
  id="topics-overview-chart"
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-overview"
  >
</sensemaker-chart>`,
      language: "html",
      type: "code",
    },
  },
};
