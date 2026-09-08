/**
 * @fileoverview Build automation script for generating static and inlined HTML reports.
 * This script orchestrates the entire build pipeline including directory cleaning,
 * data conversion (CSV to JSON), template rendering (Mustache), asset copying,
 * and deployment tasks.
 *
 * Supports the historical copy-into-`input/` workflow and optional explicit
 * input/output path flags (see resolveBuildOptions).
 *
 * @module BuildScript
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { processReportData, resolveBuildOptions } from "./data.js";

const packageRoot = path.dirname(fileURLToPath(import.meta.url));

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
      cwd: packageRoot,
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
    execSync(cmd, { stdio: "inherit", cwd: packageRoot });
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
 * Active build paths for the current run. Defaults match the historical layout.
 * @type {{
 *   inputDir: string,
 *   workDir: string,
 *   staticOutDir: string,
 *   inlineOutFile: string,
 *   opinionsCsv: string,
 *   summaryPath: string|null,
 *   configPath: string|null,
 *   predictedPath: string|null,
 *   useLegacyDataJs: boolean,
 * }}
 */
let ctx = {
  inputDir: path.join(packageRoot, "input"),
  workDir: path.join(packageRoot, "temp"),
  staticOutDir: path.join(packageRoot, "output", "static"),
  inlineOutFile: path.join(packageRoot, "output", "inline", "index.html"),
  opinionsCsv: path.join(packageRoot, "input", "opinions.csv"),
  summaryPath: null,
  configPath: null,
  predictedPath: null,
  useLegacyDataJs: true,
};

/**
 * Relativize a path for shell commands executed with cwd=packageRoot.
 * @param {string} absPath
 * @returns {string}
 */
function rel(absPath) {
  return path.relative(packageRoot, absPath) || ".";
}

/**
 * Resets ctx to the historical default layout under packageRoot.
 */
function resetDefaultCtx() {
  ctx = {
    inputDir: path.join(packageRoot, "input"),
    workDir: path.join(packageRoot, "temp"),
    staticOutDir: path.join(packageRoot, "output", "static"),
    inlineOutFile: path.join(packageRoot, "output", "inline", "index.html"),
    opinionsCsv: path.join(packageRoot, "input", "opinions.csv"),
    summaryPath: null,
    configPath: null,
    predictedPath: null,
    useLegacyDataJs: true,
  };
}

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
    rm(ctx.workDir);
    mkdir(ctx.workDir);
    if (!dev) {
      const defaultOutput = path.join(packageRoot, "output");
      const defaultStatic = path.join(defaultOutput, "static");
      const defaultInline = path.join(defaultOutput, "inline", "index.html");
      const usingDefaultStatic =
        path.resolve(ctx.staticOutDir) === defaultStatic;
      const usingDefaultInline =
        path.resolve(ctx.inlineOutFile) === defaultInline;
      // Only wipe the package output/ tree when writing into the default layout.
      if (usingDefaultStatic && usingDefaultInline) {
        rm(defaultOutput);
        mkdir(defaultOutput);
      } else if (usingDefaultStatic) {
        rm(ctx.staticOutDir);
      }
    }
  },

  /**
   * Converts input CSV data to JSON and runs the data processing script.
   * Uses `csvtojson` for conversion and executes `data.js` (legacy) or
   * processReportData (explicit paths).
   */
  data: () => {
    console.log("...converting opinions csv to json and processing data");
    const opinionsJson = path.join(ctx.workDir, "opinions.json");
    const fileDescriptor = fs.openSync(opinionsJson, "w");

    try {
      execSync(`npx -y -q csvtojson "${ctx.opinionsCsv}"`, {
        stdio: ["ignore", fileDescriptor, "inherit"],
        cwd: packageRoot,
      });
    } catch (e) {
      console.error("Error converting CSV");
      process.exit(1);
    } finally {
      fs.closeSync(fileDescriptor);
    }

    if (ctx.useLegacyDataJs) {
      runInherit("node data.js");
      return;
    }

    const opinionsRaw = JSON.parse(fs.readFileSync(opinionsJson, "utf-8"));
    const summaryPath =
      ctx.summaryPath || path.join(ctx.inputDir, "summary.json");
    const configPath =
      ctx.configPath ||
      (fs.existsSync(path.join(ctx.inputDir, "config.json"))
        ? path.join(ctx.inputDir, "config.json")
        : null);
    const predictedPath =
      ctx.predictedPath ||
      (fs.existsSync(path.join(ctx.inputDir, "predicted.json"))
        ? path.join(ctx.inputDir, "predicted.json")
        : null);

    processReportData({
      opinionsRaw,
      summaryPath,
      configPath,
      predictedPath,
      inputDir: ctx.inputDir,
      packageRoot,
      workDir: ctx.workDir,
    });
  },

  /**
   * Generates the HTML using the static data JSON file and Mustache templates.
   * Result is saved to `temp/raw.html`.
   */
  htmlStatic: () => {
    console.log("...generating HTML (Static)");
    const html = run(
      `npx -y -q mustache "${rel(path.join(ctx.workDir, "data-static.json"))}" src/index.mustache`,
    );
    fs.writeFileSync(path.join(ctx.workDir, "raw.html"), html);
  },

  /**
   * Generates the HTML using the inline data JSON file and Mustache templates.
   * Result is saved to `temp/raw.html`.
   */
  htmlInline: () => {
    console.log("...generating HTML (Inline)");
    const html = run(
      `npx -y -q mustache "${rel(path.join(ctx.workDir, "data-inline.json"))}" src/index.mustache`,
    );
    fs.writeFileSync(path.join(ctx.workDir, "raw.html"), html);
  },

  /**
   * Copies all static assets (CSS, JS, logos, JSON) to the output directory.
   * Filters input files to only include logos.
   */
  assets: () => {
    console.log("...copying assets");
    mkdir(ctx.staticOutDir);

    const srcDir = path.join(packageRoot, "src");
    if (fs.existsSync(srcDir)) {
      const srcFiles = fs.readdirSync(srcDir);
      srcFiles.forEach((file) => {
        cp(path.join(srcDir, file), ctx.staticOutDir);
      });
    }

    cp(path.join(ctx.workDir, "quotes.json"), ctx.staticOutDir);
    cp(
      path.join(ctx.workDir, "raw.html"),
      path.join(ctx.staticOutDir, "index.html"),
    );

    if (fs.existsSync(ctx.inputDir)) {
      const inputFiles = fs.readdirSync(ctx.inputDir);
      inputFiles
        .filter((f) => f.startsWith("logo."))
        .forEach((f) => {
          cp(path.join(ctx.inputDir, f), ctx.staticOutDir);
        });
    }

    const mustacheFile = path.join(ctx.staticOutDir, "index.mustache");
    if (fs.existsSync(mustacheFile)) fs.rmSync(mustacheFile);
  },

  /**
   * Inlines external resources (CSS/JS) into the HTML file for a single-file output.
   * Uses `inline-source-cli`.
   */
  inlineAssets: () => {
    console.log("...inlining data and assets");

    mkdir(path.join(ctx.workDir, "svg"));

    const srcDir = path.join(packageRoot, "src");
    if (fs.existsSync(srcDir)) {
      const srcFiles = fs.readdirSync(srcDir);
      srcFiles.forEach((file) => {
        if (file !== "index.mustache") {
          cp(path.join(srcDir, file), ctx.workDir);
        }
      });
    }

    if (fs.existsSync(ctx.inputDir)) {
      const inputFiles = fs.readdirSync(ctx.inputDir);
      inputFiles
        .filter((f) => f.startsWith("logo."))
        .forEach((f) => {
          cp(path.join(ctx.inputDir, f), ctx.workDir);
        });
    }

    mkdir(path.dirname(ctx.inlineOutFile));

    const rawRel = rel(path.join(ctx.workDir, "raw.html"));
    const workRel = rel(ctx.workDir);
    const tempIndex = path.join(ctx.workDir, "index.html");
    run(
      `npx -y -q inline-source-cli --root "${workRel}" "${rawRel}" > "${rel(tempIndex)}"`,
    );

    let html = fs.readFileSync(tempIndex, "utf-8");

    const fontsDir = path.join(ctx.workDir, "fonts");
    if (fs.existsSync(fontsDir)) {
      const fontFiles = fs
        .readdirSync(fontsDir)
        .filter((f) => f.endsWith(".woff2"));
      for (const fontFile of fontFiles) {
        const txtFile = path.join(
          fontsDir,
          fontFile.replace(/\.woff2$/, ".txt"),
        );
        if (!fs.existsSync(txtFile)) continue;

        const dataUri = fs.readFileSync(txtFile, "utf-8").trim();
        const fontUrl = `fonts/${fontFile}`;
        html = html.split(fontUrl).join(dataUri);
      }
    }

    fs.writeFileSync(ctx.inlineOutFile, html);
  },

  /**
   * Cleans up temporary working directories.
   */
  end: () => {
    console.log("...clean up");
    rm(ctx.workDir);
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
    resetDefaultCtx();
    tasks.start(true);
    tasks.data();
  },

  /**
   * Deploys the static output to a 'docs' folder and commits to Git.
   * Intended for GitHub Pages deployment.
   */
  github: () => {
    resetDefaultCtx();
    tasks.static();
    console.log("...deploying to github docs");

    const docsDir = path.join(packageRoot, "docs");
    rm(docsDir);
    mkdir(docsDir);

    const staticFiles = fs.readdirSync(ctx.staticOutDir);
    staticFiles.forEach((file) => {
      cp(path.join(ctx.staticOutDir, file), docsDir);
    });

    fs.closeSync(fs.openSync(path.join(docsDir, ".nojekyll"), "w"));

    runInherit("git add -A");
    runInherit('git commit -m "update github pages"');
    runInherit("git push");
  },
};

/**
 * Stages caller-supplied input files into workDir/staged-input so logos and
 * translations resolve next to config, matching the historical input/ layout.
 *
 * @param {ReturnType<typeof resolveBuildOptions>} options
 * @param {string} workDir
 * @returns {string} Path to the staged input directory.
 */
function stageExplicitInputs(options, workDir) {
  const stagedInput = path.join(workDir, "staged-input");
  mkdir(stagedInput);

  fs.copyFileSync(
    options.opinionsPath,
    path.join(stagedInput, "opinions.csv"),
  );
  fs.copyFileSync(options.summaryPath, path.join(stagedInput, "summary.json"));

  if (options.configPath) {
    fs.copyFileSync(
      options.configPath,
      path.join(stagedInput, "config.json"),
    );
  }

  if (options.predictedPath) {
    fs.copyFileSync(
      options.predictedPath,
      path.join(stagedInput, "predicted.json"),
    );
  }

  if (fs.existsSync(options.inputDir)) {
    for (const name of fs.readdirSync(options.inputDir)) {
      if (name.startsWith("logo.")) {
        cp(path.join(options.inputDir, name), stagedInput);
      }
    }
    const translations = path.join(options.inputDir, "translations.json");
    if (fs.existsSync(translations)) {
      cp(translations, stagedInput);
    }
    if (options.configPath && fs.existsSync(options.configPath)) {
      const config = JSON.parse(fs.readFileSync(options.configPath, "utf-8"));
      if (config.translations) {
        const t1 = path.join(options.inputDir, config.translations);
        const t2 = path.join(
          options.inputDir,
          `${config.translations}.json`,
        );
        if (fs.existsSync(t1)) cp(t1, stagedInput);
        else if (fs.existsSync(t2)) cp(t2, stagedInput);
      }
    }
  }

  return stagedInput;
}

/**
 * Runs a report build from argv.
 *
 * @param {string[]} [argv=process.argv]
 * @param {string} [cwd=process.cwd()]
 * @returns {Promise<{output?: string, outputDir?: string}>}
 */
export async function runBuild(argv = process.argv, cwd = process.cwd()) {
  const options = resolveBuildOptions(argv, cwd);
  const command = options.command;

  if (command === "preview") {
    tasks.preview();
    return {};
  }
  if (command === "dev") {
    tasks.dev();
    return {};
  }
  if (command === "github") {
    tasks.github();
    return {};
  }

  if (!["static", "inline", "build"].includes(command)) {
    if (tasks[command]) {
      tasks[command]();
      return {};
    }
    throw new Error(`Unknown command: ${command}`);
  }

  const mode = options.effectiveMode;
  const useExplicit =
    options.hasExplicitInputPaths || options.hasExplicitOutputPaths;

  if (!useExplicit) {
    resetDefaultCtx();
    if (mode === "inline") {
      tasks.inline();
      return { output: ctx.inlineOutFile };
    }
    tasks.static();
    return { outputDir: ctx.staticOutDir };
  }

  // Explicit paths: prepare workDir, stage inputs, then run the same npx steps.
  const workDir = path.join(packageRoot, "temp");
  ctx = {
    inputDir: path.join(workDir, "staged-input"),
    workDir,
    staticOutDir:
      mode === "static"
        ? options.outputDir
        : path.join(packageRoot, "output", "static"),
    inlineOutFile:
      mode === "inline"
        ? options.output
        : path.join(packageRoot, "output", "inline", "index.html"),
    opinionsCsv: path.join(workDir, "staged-input", "opinions.csv"),
    summaryPath: null,
    configPath: null,
    predictedPath: null,
    useLegacyDataJs: false,
  };

  if (mode === "inline") {
    console.log("\n** BUILDING REPORT (INLINE) **\n");
  } else {
    console.log("\n** BUILDING REPORT (STATIC) **\n");
  }

  tasks.start();
  console.log("...staging inputs");
  const stagedInput = stageExplicitInputs(options, workDir);
  ctx.inputDir = stagedInput;
  ctx.opinionsCsv = path.join(stagedInput, "opinions.csv");
  ctx.summaryPath = path.join(stagedInput, "summary.json");
  ctx.configPath = fs.existsSync(path.join(stagedInput, "config.json"))
    ? path.join(stagedInput, "config.json")
    : null;
  ctx.predictedPath = fs.existsSync(path.join(stagedInput, "predicted.json"))
    ? path.join(stagedInput, "predicted.json")
    : null;

  tasks.data();
  if (mode === "inline") {
    tasks.htmlInline();
    tasks.inlineAssets();
    tasks.end();
    console.log("\n** BUILD COMPLETE! **\n");
    return { output: ctx.inlineOutFile };
  }

  tasks.htmlStatic();
  tasks.assets();
  tasks.end();
  console.log("\n** BUILD COMPLETE! **\n");
  return { outputDir: ctx.staticOutDir };
}

// -----------------------------------------------------------------------------
// CLI Argument Parsing
// -----------------------------------------------------------------------------

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (isMain) {
  runBuild(process.argv, process.cwd()).catch((error) => {
    console.error(error.message || error);
    process.exit(1);
  });
}
