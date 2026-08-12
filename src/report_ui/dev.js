import express from "express";
import mustache from "mustache";
import fs from "fs";
import { execSync } from "child_process";
import browserSync from "browser-sync";

const bs = browserSync.create();

const app = express();
const PORT = 3000;

app.use(express.static("src"));

function ensureData() {
  if (!fs.existsSync("./temp/data-static.json")) {
    console.log("temp/data-static.json not found, running build.js dev...");
    execSync("node build.js dev", { stdio: "inherit" });
  }
}

app.get("/", (req, res) => {
  try {
    ensureData();
    const template = fs.readFileSync("./src/index.mustache", "utf8");
    const data = JSON.parse(fs.readFileSync("./temp/data-static.json", "utf8"));

    const html = mustache.render(template, data);
    res.send(html);
  } catch (e) {
    console.error(e);
    res.status(500).send(`<h1>Error</h1><pre>${e.message}</pre>`);
  }
});

app.listen(PORT, () => {
  console.log("Server started. Initializing BrowserSync...");

  bs.init({
    proxy: `http://localhost:${PORT}`,
    files: [
      "src/**/*",
      "temp/**/*",
      {
        match: ["input/**/*"],
        fn: function (event, file) {
          console.log(`Input changed: ${file}. Re-running data processing...`);
          try {
            execSync("node build.js dev", { stdio: "inherit" });
          } catch (err) {
            console.error("Error rebuilding data:", err);
          }
        },
      },
    ],
    port: 3000,
    open: true,
    notify: false,
    serveStatic: ["temp", "input"],
  });
});
