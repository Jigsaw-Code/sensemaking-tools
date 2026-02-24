/** @type { import('@storybook/web-components-vite').StorybookConfig } */
const config = {
  stories: ["../stories/**/*.mdx", "../stories/**/*.stories.@(js|jsx|mjs|ts|tsx)"],
  addons: ["@storybook/addon-essentials"],
  framework: {
    name: "@storybook/web-components-vite",
    options: {},
  },
  staticDirs: ["../stories/data", "../stories/assets"],
  viteFinal: async (config) => {
    // Set base URL for GitHub Pages
    config.base =
      process.env.NODE_ENV === "production" ? "/sensemaking-tools/visualization-docs/" : "/";
    return config;
  },
};
export default config;
