import fs from "fs";
import path from "path";

const CWD = process.cwd();

// Configuration
const config = {
	// CSS file to copy
	cssFile: "src/styles/app.css",
	
	// Output CSS file path
	outputFile: "qualtrics/style.css",
	
	// Variables file to load
	variablesFile: "src/data/variables.json"
};

const processCssFile = () => {
	console.log(`Extracting CSS from ${config.cssFile}...\n`);
	
	try {
		const cssFilePath = path.join(CWD, config.cssFile);
		
		if (!fs.existsSync(cssFilePath)) {
			throw new Error(`CSS file not found: ${config.cssFile}`);
		}
		
		// Read the CSS file
		let cssContent = fs.readFileSync(cssFilePath, "utf-8");
		
		// Remove the @import './font.css'; line
		cssContent = cssContent.replace(/@import\s+['"]\.\/font\.css['"];?\s*\n?/g, "");
		// Remove the @import './global.css'; line
		cssContent = cssContent.replace(/@import\s+['"]\.\/global\.css['"];?\s*\n?/g, "");
		// Load variables from config file
		const variablesPath = path.join(CWD, config.variablesFile);
		let cssVariables = "";
		
		if (fs.existsSync(variablesPath)) {
			const variablesContent = fs.readFileSync(variablesPath, "utf-8");
			const variables = JSON.parse(variablesContent);
			
			// Generate CSS custom properties from variables
			cssVariables = ":root {\n";
			
			// Add colors as CSS variables
			if (variables.colors && Array.isArray(variables.colors)) {
				variables.colors.forEach((color, index) => {
					cssVariables += `  --color-${index}: #${color};\n`;
				});
			}
			
			// Add light colors as CSS variables
			if (variables.colorsLight && Array.isArray(variables.colorsLight)) {
				variables.colorsLight.forEach((color, index) => {
					cssVariables += `  --color-light-${index}: #${color};\n`;
				});
			}
			
			// Add other variables as needed
			if (variables.googleSheetsLink) {
				cssVariables += `  --google-sheets-link: "${variables.googleSheetsLink}";\n`;
			}
			
			cssVariables += "}\n\n";
			
			// Replace placeholders in CSS (e.g., {{colors.0}} or {{googleSheetsLink}})
			cssContent = cssContent.replace(/\{\{([^}]+)\}\}/g, (match, key) => {
				const keys = key.split(".");
				let value = variables;
				for (const k of keys) {
					if (value && typeof value === "object" && k in value) {
						value = value[k];
					} else {
						return match; // Return original if not found
					}
				}
				return typeof value === "string" ? value : JSON.stringify(value);
			});
		}
		
		// Prepend CSS variables to the content
		cssContent = cssVariables + cssContent;
		
		// Determine output path
		const outputPath = path.join(CWD, config.outputFile);
		const outputDir = path.dirname(outputPath);
		
		// Create output directory if it doesn't exist
		if (!fs.existsSync(outputDir)) {
			fs.mkdirSync(outputDir, { recursive: true });
		}
		
		// Write the file
		fs.writeFileSync(outputPath, cssContent, "utf-8");
		console.log(`✓ Extracted CSS to: ${config.outputFile}`);
		console.log(`  Output path: ${path.relative(CWD, outputPath)}\n`);
		
	} catch (error) {
		console.error(`Error: ${error.message}`);
		process.exit(1);
	}
};

// Export the function so it can be called from other scripts
export { processCssFile as extractCss };

// Execute if run directly
processCssFile();
console.log("Done!");

