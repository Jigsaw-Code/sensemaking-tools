import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/sensemaker-chart.js"),
      name: "SensemakerChart",
      fileName: (format) => `sensemaker-chart.${format}.js`,
      formats: ["es", "cjs", "umd"],
    },
    rollupOptions: {
      output: {
        globals: {
          d3: "d3",
        },
        external: ["d3"],
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === "style.css") return "sensemaker-viz-components.css";
          return assetInfo.name;
        },
        exports: "named",
      },
    },
  },
});
