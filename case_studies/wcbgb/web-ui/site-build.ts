const { Command } = require("commander");
const { copyFileSync, writeFileSync } = require("fs");
const path = require("path");
const { exec, ExecException } = require("child_process");

async function main(): Promise<void> {
  // Parse command line arguments.
  const program = new Command();
  program.option("-t, --topics <file>", "The topics file location.");
  program.option("-s, --summary <file>", "The summary file location.");
  program.option("-c, --comments <file>", "The comments file location.");
  program.option("-r, --reportTitle <title>", "The title of the report.");
  program.parse(process.argv);
  const options = program.opts();

  if(!options["topics"]) {
    throw Error("Topics file path not specified");
  }
  if(!options["summary"]) {
    throw Error("Summary file path not specified");
  }
  if(!options["comments"]) {
    throw Error("Comments file path not specified");
  }
  if(!options["reportTitle"]) {
    throw Error("Report title not specified");
  }

  const reportMetadata = {
    title: options["reportTitle"],
  };

  // path to "data" folder
  const baseOutputPath = path.join(__dirname, "./data");

  copyFileSync(options["topics"], path.join(baseOutputPath, "/topic-stats.json"));
  copyFileSync(options["summary"], path.join(baseOutputPath, "/summary.json"));
  copyFileSync(options["comments"], path.join(baseOutputPath, "/comments.json"));
  writeFileSync(path.join(baseOutputPath, "/metadata.json"), JSON.stringify(reportMetadata, null, 2));

  await exec("npm run build", (error: typeof ExecException | null, stdout: string, stderr: string) => {
    if(error) {
      console.error(`Build failed: ${error.message}`);
      return;
    }
    if(stderr) {
      console.error(`Build errors/warnings: ${stderr}`);
    }
    console.log(`Build output: ${stdout}`);
    console.log("Build complete");
  });
}

main();
