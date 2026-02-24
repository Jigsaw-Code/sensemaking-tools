import "../src/sensemaker-chart.js";
import { scriptWarning } from "./shared/scriptWarning.js";
import comments from "./data/comments.json";

const defaultColors = ["#3A708A", "#589AB7", "#8bc3da", "#757575"];
const basePath =
  process.env.NODE_ENV === "production" ? "/sensemaking-tools/visualization-docs" : "";

const getDataSource = (path) => {
  // If it's a remote URL (starts with http:// or https://), return as is
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  // Otherwise, prepend the base path for local files
  return `${basePath}${path}`;
};

export default {
  title: "Charts/TopicAlignment",
  tags: ["autodocs"],
  argTypes: {
    dataSource: {
      name: "data-source",
      control: "text",
      description:
        "Local path or remote URL to the main data source. Not used if data is provided directly via property.",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
        category: "Data Attributes",
      },
    },
    view: {
      control: "select",
      options: ["solid", "waffle"],
      description:
        'Display mode: "solid" (bar) or "waffle" (squares). Can be set statically or dynamically via DOM manipulation.',
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "solid" },
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
      description: "Topic to filter data .",
      table: {
        type: { summary: "string" },
        defaultValue: { summary: "none" },
        category: "Required",
      },
    },
    colors: {
      name: "colors",
      control: "object",
      description:
        "Array of colors to use in the chart rendered in order of high, medium, low, and uncertain.",
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
The topic alignment chart displays agreement/disagreement percentages with options for different view options (solid or waffle). The view can be set statically or dynamically via DOM manipulation.
                
${scriptWarning}
                `,
      },
    },
  },
};

// Basic template
const Template = ({ id, dataSource, view, topicFilter, colors, fontFamily, altText }) => {
  return `
    <sensemaker-chart
      id="${id}"
      data-source="${getDataSource(dataSource)}"
      chart-type="topic-alignment"
      view="${view}"
      colors='${JSON.stringify(colors || defaultColors)}'
      font-family="${fontFamily || "Noto Sans"}"
      ${topicFilter ? `topic-filter="${topicFilter}"` : ""}
      ${altText ? `alt-text="${altText}"` : ""}
    ></sensemaker-chart>
  `;
};

// Example with solid view
export const SolidView = Template.bind({});
SolidView.args = {
  id: "topic-alignment-chart-solid",
  dataSource: `/comments.json`,
  view: "solid",
  topicFilter: "Education",
  colors: defaultColors,
};
SolidView.parameters = {
  docs: {
    description: {
      story: `The solid view shows the aggregated agreement/disagreement as stacked bars.`,
    },
    source: {
      code: `<sensemaker-chart
  data-source="/comments.json"
  chart-type="topic-alignment"
  view="solid"
  colors='["#3A708A", "#589AB7", "#8bc3da", "#757575"]'
  topic-filter="education">
</sensemaker-chart>`,
      language: "html",
      type: "code",
    },
  },
};

// Example with waffle view
export const WaffleView = Template.bind({});
WaffleView.args = {
  id: "topic-alignment-chart-waffle",
  dataSource: `/comments.json`,
  view: "waffle",
  topicFilter: "Education",
  colors: defaultColors,
};
WaffleView.parameters = {
  docs: {
    description: {
      story: `The waffle view shows individual statements as squares.`,
    },
    source: {
      code: `<sensemaker-chart
  data-source="stories/data/comments.json"
  chart-type="topic-alignment"
  view="waffle"
  colors='["#3A708A", "#589AB7", "#8bc3da", "#757575"]'
  topic-filter="education">
</sensemaker-chart>`,
      language: "html",
      type: "code",
    },
  },
};

// Interactive template with view toggle
const ViewToggleTemplate = ({ dataSource, topicFilter, colors, fontFamily, altText }) => {
  // This will run after the component is added to the DOM
  setTimeout(() => {
    const chart = document.getElementById("topic-alignment-chart-with-toggle");
    const viewInputs = document.querySelectorAll(
      '#topic-alignment-chart-with-toggle-controls input[name="view"]'
    );

    // Add event listeners to view controls
    viewInputs.forEach((input) => {
      input.addEventListener("change", (e) => {
        if (e.target.checked) {
          chart.setAttribute("view", e.target.value);
        }
      });
    });
  }, 100);

  return `
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <div id="topic-alignment-chart-with-toggle-controls" class="view-controls" style="margin-bottom: 20px;">
        <label style="margin-right: 15px; cursor: pointer;">
          <input type="radio" name="view" value="solid" checked /> Solid View
        </label>
        <label style="cursor: pointer;">
          <input type="radio" name="view" value="waffle" /> Waffle View
        </label>
      </div>
      
      <sensemaker-chart
        id="topic-alignment-chart-with-toggle"
        data-source="${dataSource}"
        chart-type="topic-alignment"
        view="solid"
        colors='${JSON.stringify(colors || defaultColors)}'
        font-family="${fontFamily || "Noto Sans"}"
        ${topicFilter ? `topic-filter="${topicFilter}"` : ""}
        ${altText ? `alt-text="${altText}"` : ""}>
      </sensemaker-chart>
    </div>
  `;
};

// Story with view toggle
export const WithViewToggle = ViewToggleTemplate.bind({});
WithViewToggle.args = {
  id: "topic-alignment-chart-with-toggle",
  dataSource: `${basePath}/comments.json`,
  topicFilter: "education",
  colors: defaultColors,
};
WithViewToggle.parameters = {
  docs: {
    description: {
      story: `The chart view updates via external controls.`,
    },
    source: {
      code: `<!-- Toggle controls -->
<div class="view-controls">
  <label>
    <input type="radio" name="view" value="solid" checked /> Solid View
  </label>
  <label>
    <input type="radio" name="view" value="waffle" /> Waffle View
  </label>
</div>

<!-- Chart component -->
<sensemaker-chart
  id="topic-alignment-chart-with-toggle"
  data-source="stories/data/comments.json"
  chart-type="topic-alignment"
  view="solid"
  colors='["#3A708A", "#589AB7", "#8bc3da", "#757575"]'
  topic-filter="education">
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

// Template for direct data injection with view toggle
const DirectDataTemplate = ({ topicFilter, colors, fontFamily, altText }) => {
  const chartId = "topic-alignment-direct-data";

  // This script will run after the component is added to the DOM
  setTimeout(() => {
    const chart = document.getElementById(chartId);
    const viewInputs = document.querySelectorAll(
      '#topic-alignment-direct-data-controls input[name="view"]'
    );

    if (chart) {
      // Provide the imported data directly to the component properties
      chart.data = comments;

      // Add event listeners to view controls
      viewInputs.forEach((input) => {
        input.addEventListener("change", (e) => {
          if (e.target.checked) {
            chart.setAttribute("view", e.target.value);
          }
        });
      });
    } else {
      console.error(`Element with id '${chartId}' not found for direct data assignment.`);
    }
  }, 0);

  return `
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <div id="topic-alignment-direct-data-controls" class="view-controls" style="margin-bottom: 20px;">
        <label style="margin-right: 15px; cursor: pointer;">
          <input type="radio" name="view" value="solid" checked /> Solid View
        </label>
        <label style="cursor: pointer;">
          <input type="radio" name="view" value="waffle" /> Waffle View
        </label>
      </div>
      
      <sensemaker-chart
        id="${chartId}"
        chart-type="topic-alignment"
        view="solid"
        colors='${JSON.stringify(colors || defaultColors)}'
        font-family="${fontFamily || "Noto Sans"}"
        ${topicFilter ? `topic-filter="${topicFilter}"` : ""}
        ${altText ? `alt-text="${altText}"` : ""}>
      </sensemaker-chart>
    </div>
  `;
};

export const WithDirectData = DirectDataTemplate.bind({});
WithDirectData.args = {
  id: "topic-alignment-direct-data",
  topicFilter: "Education",
  colors: defaultColors,
};
WithDirectData.parameters = {
  docs: {
    description: {
      story: `This example demonstrates providing data directly to the component via JavaScript properties (.data) instead of using URL attributes, while also including interactive view controls to toggle between solid and waffle views. The data is imported from local JSON files within the story itself.`,
    },
    source: {
      code: `// Data is imported in the story script:
// import comments from "../stories/data/includesTopics-comments-with-scores.json";

// HTML:
<div class="view-controls">
  <label>
    <input type="radio" name="view" value="solid" checked /> Solid View
  </label>
  <label>
    <input type="radio" name="view" value="waffle" /> Waffle View
  </label>
</div>

<sensemaker-chart
  id="topic-alignment-direct-data"
  chart-type="topic-alignment"
  view="solid"
  colors='["#3A708A", "#589AB7", "#8bc3da", "#757575"]'
  topic-filter="Education">
</sensemaker-chart>

<script>
  const chart = document.getElementById("topic-alignment-direct-data");
  // Assign data directly
  chart.data = comments;
  
  // Add view toggle functionality
  const viewInputs = document.querySelectorAll('input[name="view"]');
  viewInputs.forEach((input) => {
    input.addEventListener("change", (e) => {
      if (e.target.checked) {
        chart.setAttribute("view", e.target.value);
      }
    });
  });
</script>`,
      language: "js",
      type: "code",
    },
  },
};
