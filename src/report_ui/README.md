# Jigsaw Sensemaking Report Generator

This tool automatically generates interactive HTML reports from structured opinion data and AI-generated summaries. It takes CSV and JSON inputs and transforms them into a visual report featuring topic clusters, opinion distributions, and representative quotes.

## Quick start (Report Generation)

If you are just here to generate a report from new data, follow these steps.

### Prerequisites
*   **Node.js**: Ensure you have [Node.js](https://nodejs.org/) installed on your machine.

### Setup
* Download a [zip](https://github.com/polygraph-cool/jigsaw-sensemaking-generator/archive/refs/heads/main.zip) or
* Use this repo as [a template](https://github.com/new?owner=polygraph-cool&template_name=jigsaw-sensemaking-generator&template_owner=polygraph-cool) *also via button in top right*

### Prepare your data
Navigate to the `input/` folder. You must place the following files there, replacing any existing ones:

1.  **`opinions.csv`**: The raw data containing participant quotes.
    *   *Required Columns:* `topic`, `opinion`, `representative_text` (the quote), `rid` (participant ID).
    *   *Optional:* `AVERAGE_OF_2_BRIDGING` (used for sorting quotes by importance).
2.  **`summary.json`**: The AI-generated summary of the conversation.
    *   *Structure:* Must contain a `title`, `text` (executive summary), and `sub_contents` (array of topic objects with `title` and `text`).
3.  **`config.json`**: [Basic configuration](#configurationcustomization) and custimazation options (e.g., logo path).
4.  **Logo**: Place an image file (e.g., `logo.png` or `logo.svg`) in the `input/` folder.


#### Configuration/customization
In config.json, optionally add these properties

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `logo` | `string` | `""` | Header image file. options: `"logo.png"` or `"logo.svg"`. |
| `overview_chart` | `string` | `"toggle"` | Overview chart display mode. Options: `"toggle"`, `"topics"`, or `"opinions"`. |
| `number_of_top_opinions` | `number` | `10` | The number of items to show in the opinions overview chart. |
| `number_of_sample_quotes` | `number` | `4` | The number of quote previews to display for each opinion. |
| `chart_colors` | `array` | `["#AFB42B", "#F4511E", "#3949AB", "#E52592", "#00897B", "#EFB22F", "#aaa"]` | Array of color codes. |

### 3. Generate the Report
No dependecies are required to generate the report. Open your terminal/command prompt in the project folder and run:

```bash
# Static (default): Best for hosting
npm run static

# OR

# Inline: best for emailing/offline use
npm run inline
```

### 4. Results
All results are in the `output` folder ready to deploy or share. To preview:
*   **Static report**: `npm run preview`
*   **Inline report**: Located in `output/inline/`. This is a single `index.html` file containing all data, scripts, and styles.

---

## Configuration and input details

### `opinions.csv` Format
The logic relies on specific headers. Ensure your CSV looks like this:

| topic | opinion | representative_text | rid |

### `summary.json` Format
This file maps the visual topics to the text summaries.
```json
{
  "title": "# Conversation Title",
  "text": "Executive summary paragraph...",
  "sub_contents": [
    {
      "title": "## Topic", 
      "text": "Summary of the topic..."
    }
  ]
}
```

---

## Development guide

If you are a developer looking to modify the report or build process, here is the architectural overview.

### Project Structure

*   **`input/`**: Raw data entry point.
*   **`src/`**: Source code for the report.
    *   `script.js`: Frontend logic and charts.
    *   `style.css`: Visual styling.
    *   `index.mustache`: HTML template used during the build.
*   **`data.js`**: The ETL (Extract, Transform, Load) script. It converts the flat CSV into a hierarchical JSON structure (`Topic -> Opinions -> Quotes`).
*   **`build.js`**: The orchestration script. It handles file cleaning, data processing, templating, and asset copying.

### Key Commands

| Command | Description |
| :--- | :--- |
| `npm run static` | Builds the report separating HTML, CSS, JS, and JSON. Loads quotes lazily. |
| `npm run inline` | Builds a single HTML file. Inlines all CSS, JS, and the full dataset. |
| `npm run preview` | Starts a local `browser-sync` server to view the `output/static` build with live reloading. |
| `npm run dev` | Runs the data processing steps without a full build cleanup (useful for debugging data logic and a live server for testing and developing features). |

### Data Pipeline (`data.js`)
1.  **Ingestion**: Reads `opinions.csv` via `csvtojson`.
2.  **Grouping**: Groups raw rows by `topic`, then by `opinion`.
4.  **Output**: Generates `data-static.json` (lightweight payload) and `data-inline.json` (heavy payload with all quotes).

### Visualization Logic (`script.js`)
*   **Frameworks**: D3.v7 (charts), Tippy.js (tooltips), Mustache (templating).
*   **Charts**:
    *   *Topic Chart*: A stacked horizontal bar chart summarizing opinion distribution.
    *   *Opinion Chart*: A flattened bar chart of the top opinions across all topics.
    *   *Donut Charts*: Per-topic visualization of opinion breakdown.
*   **Data binding**: Data is injected into `window.PAYLOAD` during the build process.

### Customizing the build
The `build.js` file contains a `tasks` object. You can add new build steps here.