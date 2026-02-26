import express from "express";
import mustache from "mustache";
import fs from "fs";
import browserSync from "browser-sync";

const bs = browserSync.create();

const app = express();
const PORT = 3000;

app.use(express.static("src"));

app.get("/", (req, res) => {
  try {
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
    files: ["src/**/*", "temp/**/*"],
    port: 3000,
    open: true,
    notify: false,
    serveStatic: ["temp", "input"],
  });
});
