#!/usr/bin/env node
/**
 * Thin CLI wrapper around runBuild. Prefer `node build.js ...` when working
 * inside this directory; the bin entry is for convenience when installed.
 */
import { runBuild } from "../build.js";

runBuild(process.argv, process.cwd())
  .then((result) => {
    if (result.output) {
      console.log(`Report written to: ${result.output}`);
    } else if (result.outputDir) {
      console.log(`Report written to: ${result.outputDir}`);
    }
  })
  .catch((error) => {
    console.error(error.message || error);
    process.exit(1);
  });
