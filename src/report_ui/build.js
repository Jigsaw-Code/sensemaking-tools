/**
 * @fileoverview Build automation script for generating static and inlined HTML reports.
 * This script orchestrates the entire build pipeline including directory cleaning,
 * data conversion (CSV to JSON), template rendering (Mustache), asset copying,
 * and deployment tasks.
 *
 * @module BuildScript
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

/**
 * Executes a shell command synchronously and captures the output.
 * Captures stdout/stderr and returns stdout as a string.
 * Exits the process with code 1 on failure.
 *
 * @param {string} cmd - The shell command to execute.
 * @returns {string} The standard output from the command.
 * @throws {Error} If the command fails, logs error and exits process.
 */
const run = (cmd) => {
  try {
    return execSync(cmd, {
      stdio: "pipe",
      encoding: "utf-8",
      maxBuffer: 1024 * 1024 * 50, // 50MB buffer
    });
  } catch (e) {
    console.error(`Error running: ${cmd}`);
    console.error(e.stderr || e.message);
    process.exit(1);
  }
};

/**
 * Executes a shell command synchronously while inheriting stdio.
 * Useful for commands that require user interaction or live logging to the console.
 *
 * @param {string} cmd - The shell command to execute.
 */
const runInherit = (cmd) => {
  try {
    execSync(cmd, { stdio: "inherit" });
  } catch (e) {
    process.exit(1);
  }
};

/**
 * Recursively removes a directory or file if it exists.
 * Equivalent to `rm -rf`.
 *
 * @param {string} dirPath - The path to remove.
 */
const rm = (dirPath) => {
  if (fs.existsSync(dirPath))
    fs.rmSync(dirPath, { recursive: true, force: true });
};

/**
 * Creates a directory recursively if it does not already exist.
 * Equivalent to `mkdir -p`.
 *
 * @param {string} dirPath - The directory path to create.
 */
const mkdir = (dirPath) => {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
};

/**
 * Copies a source file or directory to a destination.
 * If the destination is an existing directory, the source is copied into it.
 *
 * @param {string} src - The source file or directory path.
 * @param {string} dest - The destination path.
 */
const cp = (src, dest) => {
  if (!fs.existsSync(src)) return;

  let destination = dest;

  // If dest is a folder, join the source filename to the dest path
  if (fs.existsSync(dest) && fs.statSync(dest).isDirectory())
    destination = path.join(dest, path.basename(src));

  fs.cpSync(src, destination, { recursive: true });
};

/**
 * Collection of build tasks available to the CLI.
 * @namespace
 */
const tasks = {
  /**
   * Prepares the workspace by cleaning temp folders.
   * @param {boolean} [dev=false] - If true, skips cleaning the 'output' directory.
   */
  start: (dev) => {
    console.log("...preparing directory");
    rm("temp");
    mkdir("temp");
    if (!dev) {
      rm("output");
      mkdir("output");
    }
  },

  /**
   * Converts input CSV data to JSON and runs the data processing script.
   * Uses `csvtojson` for conversion and executes `data.js`.
   */
  data: () => {
    console.log("...converting opinions csv to json and processing data");
    const fileDescriptor = fs.openSync("temp/opinions.json", "w");

    try {
      // Stream output of csvtojson directly to the file descriptor
      execSync("npx -y -q csvtojson input/opinions.csv", {
        stdio: ["ignore", fileDescriptor, "inherit"],
      });
    } catch (e) {
      console.error("Error converting CSV");
      process.exit(1);
    } finally {
      fs.closeSync(fileDescriptor);
    }

    runInherit("node data.js");
  },

  /**
   * Generates the HTML using the static data JSON file and Mustache templates.
   * Result is saved to `temp/raw.html`.
   */
  htmlStatic: () => {
    console.log("...generating HTML (Static)");
    const html = run(
      "npx -y -q mustache temp/data-static.json src/index.mustache",
    );
    fs.writeFileSync("temp/raw.html", html);
  },

  /**
   * Generates the HTML using the inline data JSON file and Mustache templates.
   * Result is saved to `temp/raw.html`.
   */
  htmlInline: () => {
    console.log("...generating HTML (Inline)");
    const html = run(
      "npx -y -q mustache temp/data-inline.json src/index.mustache",
    );
    fs.writeFileSync("temp/raw.html", html);
  },

  /**
   * Copies all static assets (CSS, JS, logos, JSON) to the output directory.
   * Filters input files to only include logos.
   */
  assets: () => {
    console.log("...copying assets");
    mkdir("output/static");

    // Copy individual files from src/ to output/static/
    if (fs.existsSync("src")) {
      const srcFiles = fs.readdirSync("src");
      srcFiles.forEach((file) => {
        cp(path.join("src", file), "output/static/");
      });
    }

    cp("temp/quotes.json", "output/static/");
    cp("temp/raw.html", "output/static/index.html");

    // Copy logo files from input/
    if (fs.existsSync("input")) {
      const inputFiles = fs.readdirSync("input");
      inputFiles
        .filter((f) => f.startsWith("logo."))
        .forEach((f) => {
          cp(path.join("input", f), "output/static/");
        });
    }

    // Cleanup artifacts
    const mustacheFile = "output/static/index.mustache";
    if (fs.existsSync(mustacheFile)) fs.rmSync(mustacheFile);
  },

  /**
   * Inlines external resources (CSS/JS) into the HTML file for a single-file output.
   * Uses `inline-source-cli`.
   */
  inlineAssets: () => {
    console.log("...inlining data and assets");

    mkdir("temp/svg");

    // copy everything recursively from src to temp (except index.mustache)
    if (fs.existsSync("src")) {
      const srcFiles = fs.readdirSync("src");
      srcFiles.forEach((file) => {
        if (file !== "index.mustache") {
          cp(path.join("src", file), "temp");
        }
      });
    }

    // Copy logo files from input/
    if (fs.existsSync("input")) {
      const inputFiles = fs.readdirSync("input");
      inputFiles
        .filter((f) => f.startsWith("logo."))
        .forEach((f) => {
          cp(path.join("input", f), "temp");
        });
    }

    mkdir("output/inline");

    run(
      "npx -y -q inline-source-cli --root temp temp/raw.html > temp/index.html",
    );

    // Replace font file references with base64 data URIs
    let html = fs.readFileSync("temp/index.html", "utf-8");

    if (fs.existsSync("temp/fonts")) {
      const fontFiles = fs
        .readdirSync("temp/fonts")
        .filter((f) => f.endsWith(".woff2"));
      for (const fontFile of fontFiles) {
        const txtFile = path.join(
          "temp/fonts",
          fontFile.replace(/\.woff2$/, ".txt"),
        );
        if (!fs.existsSync(txtFile)) continue;

        const dataUri = fs.readFileSync(txtFile, "utf-8").trim();
        const fontUrl = `fonts/${fontFile}`;

        // Replace all occurrences (single-quoted, double-quoted, or unquoted)
        html = html.split(fontUrl).join(dataUri);
      }
    }

    fs.writeFileSync("output/inline/index.html", html);
  },

  /**
   * Cleans up temporary working directories.
   */
  end: () => {
    console.log("...clean up");
    rm("temp");
  },

  /**
   * Main Pipeline: Builds the project using the Static strategy.
   * Standard build with separate CSS/JS files.
   */
  static: () => {
    console.log("\n** BUILDING REPORT (STATIC) **\n");
    tasks.start();
    tasks.data();
    tasks.htmlStatic();
    tasks.assets();
    tasks.end();
    console.log("\n** BUILD COMPLETE! **\n");
  },

  /**
   * Main Pipeline: Builds the project using the Inline strategy.
   * Single-file HTML build containing all assets.
   */
  inline: () => {
    console.log("\n** BUILDING REPORT (INLINE) **\n");
    tasks.start();
    tasks.data();
    tasks.htmlInline();
    tasks.inlineAssets();
    tasks.end();
    console.log("\n** BUILD COMPLETE! **\n");
  },

  /**
   * Starts a local development server to preview the static output.
   * Uses `browser-sync`.
   */
  preview: () => {
    runInherit(
      'npx -y -q browser-sync start --server ./output/static --files "./output/static/**"',
    );
  },

  /**
   * Helper task for development; runs data conversion without full cleanup.
   */
  dev: () => {
    tasks.start(true);
    tasks.data();
  },

  /**
   * Deploys the static output to a 'docs' folder and commits to Git.
   * Intended for GitHub Pages deployment.
   */
  github: () => {
    console.log("...deploying to github docs");

    rm("docs");
    mkdir("docs");

    const staticFiles = fs.readdirSync("output/static");
    staticFiles.forEach((file) => {
      cp(path.join("output/static", file), "docs");
    });

    // Create .nojekyll to bypass GitHub Pages Jekyll processing
    fs.closeSync(fs.openSync("docs/.nojekyll", "w"));

    runInherit("git add -A");
    runInherit('git commit -m "update github pages"');
    runInherit("git push");
  },
};

// -----------------------------------------------------------------------------
// CLI Argument Parsing
// -----------------------------------------------------------------------------

const args = process.argv.slice(2);
const command = args[0] || "static";

if (command === "build") {
  // Alias 'build' to 'inline'
  tasks.inline();
} else if (command === "static") {
  tasks.static();
} else if (command === "inline") {
  tasks.inline();
} else if (tasks[command]) {
  // Run any specific task by name
  tasks[command]();
} else {
  console.log(`Unknown command: ${command}`);
}
