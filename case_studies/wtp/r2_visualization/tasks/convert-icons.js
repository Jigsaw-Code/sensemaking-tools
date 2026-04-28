import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// --- ESM path resolution fix ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// -------------------------------


// --- Configuration ---
const ICONS_DIR = path.resolve(__dirname, '../static/icons/');
// Change output file extension to .json
const OUTPUT_FILE = path.join(ICONS_DIR, 'data-uri-output.json');

const MIME_TYPES = {
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
};

/**
 * Converts a local file to a Data URI and returns its base64 string.
 * @param {string} filePath - The full path to the image file.
 * @param {string} mimeType - The MIME type of the image.
 * @returns {string} The complete Data URI string.
 */
function fileToDataUri(filePath, mimeType) {
  // 1. Read the file into a Buffer
  const fileBuffer = fs.readFileSync(filePath);

  // 2. Base64 Encode the Buffer
  const base64Data = fileBuffer.toString('base64');

  // 3. Construct the Data URI
  return `data:${mimeType};base64,${base64Data}`;
}

/**
 * Logs a visual preview of the icon to the console using CSS.
 * @param {string} dataUri - The Data URI string of the image.
 * @param {string} fileName - The original filename.
 */
function logIconPreview(dataUri, fileName) {
  const escapedUri = dataUri.replace(/"/g, '\\"');
  const cssStyle = `background-image: url("${escapedUri}"); background-size: contain; padding: 12px;`;

  console.log(`\n--- ${fileName} ---`);
  console.log('%c ', cssStyle);
  console.log(`URI String: ${dataUri}`);
}

// --- Main Execution ---
function processIcons() {
  console.log(`Processing icons in: ${ICONS_DIR}`);

  if (!fs.existsSync(ICONS_DIR)) {
    console.error(`\n❌ Error: Directory not found at ${ICONS_DIR}`);
    console.error('Please ensure the script is run from a location where ../static/icons/ exists.');
    return;
  }

  try {
    const files = fs.readdirSync(ICONS_DIR);
    // Initialize an object to hold the key-value pairs for the JSON output
    const dataUriMap = {};
    let convertedCount = 0;

    for (const file of files) {
      const ext = path.extname(file).toLowerCase();
      const mimeType = MIME_TYPES[ext];

      if (mimeType) {
        const filePath = path.join(ICONS_DIR, file);
        const dataUri = fileToDataUri(filePath, mimeType);

        // Use the filename (without extension) as the JSON key
        const keyName = path.parse(file).name;
        dataUriMap[keyName] = dataUri;
        convertedCount++;

        // Log to console
        logIconPreview(dataUri, file);
      }
    }

    // Write the dataUriMap object to the output file as formatted JSON
    if (convertedCount > 0) {
      // Use JSON.stringify for proper JSON formatting and readability
      const jsonOutput = JSON.stringify(dataUriMap, null, 2);
      fs.writeFileSync(OUTPUT_FILE, jsonOutput, 'utf8');
      console.log(`\n✅ Successfully converted and wrote ${convertedCount} URIs to: ${OUTPUT_FILE}`);
    } else {
      console.log('\n⚠️ No supported image files found to convert.');
    }

  } catch (error) {
    console.error('\nA file system error occurred:', error.message);
  }
}

processIcons();