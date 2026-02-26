# Sensemaker Visualization Components

A collection of reusable visualization components for displaying Sensemaker data, built with D3.js and Web Components.

### Installation
```bash
npm install
```

### Development
To run the Storybook development environment:
```bash
npm run storybook
```
This will start the Storybook server at `http://localhost:6006`. You can view and interact with all components in isolation.

> **Note:** The Storybook stories use sample data from the `stories/data` directory. To view all stories correctly, ensure you have both a `comments.json` and a `summary.json` file present in `stories/data/`.

### Building the Package
To build the package for production:
```bash
npm run build
```
The compiled files will be output to the `dist/` directory.

### Building Storybook Docs
To build the static Storybook documentation site:
```bash
npm run build-storybook
```
The static site will be output to the `storybook-static/` directory. You can deploy this directory to any static site host.

### Publishing to npm

To publish a new version of the package to npm, follow these steps:

1. Make sure you are logged in to npm:
    ```bash
    npm login
    ```
2. Update the version number using one of the following commands (this will automatically update `package.json` and create a Git commit and tag):
    - For a **patch** update (bug fixes, backwards compatible):
      ```bash
      npm version patch
      ```
    - For a **minor** update (new features, backwards compatible):
      ```bash
      npm version minor
      ```
    - For a **major** update (breaking changes):
      ```bash
      npm version major
      ```
3. Build the package:
    ```bash
    npm run build
    ```
4. Publish to npm:
    ```bash
    npm publish
    ```

> **Note:** The `npm version` command will automatically update the version in `package.json`. Always ensure your changes are committed before publishing.

## Data Source and License

The data used in this demo was gathered using the [Polis software](https://compdemocracy.org/Polis/) and is sub-licensed under CC BY 4.0 with Attribution to The Computational Democracy Project. The data and more information about how the data was collected can be found at the following link:

[https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green](https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green)
