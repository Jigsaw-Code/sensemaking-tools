import fs from "fs";
import path from "path";

const CWD = process.cwd();

// Configuration
const config = {
	// Svelte file to extract JavaScript from
	svelteFile: "src/components/Index.svelte",
	
	// Output JavaScript file path
	outputFile: "qualtrics/index.js",
	
	// Lines to add at the beginning of the JS file
	headerLines: [
		"// This file was generated automatically from Index.svelte",
	],
	
	// Lines to add at the end of the JS file
	footerLines: []
};

const extractScriptFromSvelte = (filePath) => {
	const fullPath = path.join(CWD, filePath);
	
	if (!fs.existsSync(fullPath)) {
		throw new Error(`File not found: ${filePath}`);
	}
	
	const content = fs.readFileSync(fullPath, "utf-8");
	
	// Extract content between <script> and </script> tags
	const scriptMatch = content.match(/<script>([\s\S]*?)<\/script>/);
	
	if (!scriptMatch) {
		throw new Error(`No <script> tag found in ${filePath}`);
	}
	
	// Get the script content (group 1 is the content between the tags)
	let scriptContent = scriptMatch[1];
	
	// Remove leading/trailing whitespace
	scriptContent = scriptContent.trim();
	
	return scriptContent;
};

const uncommentLines = (content) => {
	// Split into lines, uncomment lines that start with //, then rejoin
	const lines = content.split("\n");
	const uncommentedLines = lines.map(line => {
		// Match lines that start with optional whitespace followed by //
		const match = line.match(/^(\s*)\/\/\s?(.*)$/);
		if (match) {
			// Uncomment: preserve leading whitespace, remove // and any space after it
			return match[1] + match[2];
		}
		// Return line as-is if it doesn't start with //
		return line;
	});
	return uncommentedLines.join("\n");
};

const removeMarkedLines = (content) => {
	// Remove lines that contain // @remove (case-insensitive)
	const lines = content.split("\n");
	const filteredLines = lines.filter(line => {
		// Check if line contains @remove marker (case-insensitive)
		return !/@remove/i.test(line);
	});
	return filteredLines.join("\n");
};

const embedJsonImports = (content) => {
	// Match import statements: import VariableName from "path/to/file.json";
	const importRegex = /import\s+(\w+)\s+from\s+["']([^"']+\.json)["'];?/g;
	
	let processedContent = content;
	const matches = [...content.matchAll(importRegex)];
	
	for (const match of matches) {
		const variableName = match[1];
		const importPath = match[2];
		
		// Resolve the actual file path
		// Handle $data alias (maps to src/data)
		let jsonFilePath = importPath;
		if (importPath.startsWith("$data/")) {
			jsonFilePath = importPath.replace("$data/", "src/data/");
		} else if (importPath.startsWith("$")) {
			// Handle other $ aliases if needed
			jsonFilePath = importPath.replace("$", "src/");
		}
		
		const fullJsonPath = path.join(CWD, jsonFilePath);
		
		if (!fs.existsSync(fullJsonPath)) {
			console.warn(`Warning: JSON file not found: ${fullJsonPath}`);
			continue;
		}
		
		// Read and parse the JSON file
		const jsonContent = fs.readFileSync(fullJsonPath, "utf-8");
		// Remove any comment lines from JSON (like the ones we added earlier)
		const cleanedJson = jsonContent
			.split("\n")
			.filter(line => !line.trim().startsWith("//"))
			.join("\n");
		
		try {
			const jsonData = JSON.parse(cleanedJson);
			const jsonString = JSON.stringify(jsonData, null, "\t");
			
			// Replace the import statement with a const declaration
			const replacement = `const ${variableName} = ${jsonString};`;
			processedContent = processedContent.replace(match[0], replacement);
			
			console.log(`✓ Embedded JSON: ${variableName} from ${importPath}`);
		} catch (error) {
			console.warn(`Warning: Failed to parse JSON file ${fullJsonPath}: ${error.message}`);
		}
	}
	
	return processedContent;
};

const processSvelteFile = () => {
	console.log(`Extracting JavaScript from ${config.svelteFile}...\n`);
	
	try {
		// Extract the script content
		let scriptContent = extractScriptFromSvelte(config.svelteFile);
		
		// Remove lines marked with // @remove
		scriptContent = removeMarkedLines(scriptContent);
		
		// Embed JSON imports as variables
		scriptContent = embedJsonImports(scriptContent);
		
		// Uncomment any commented-out lines
		scriptContent = uncommentLines(scriptContent);
		
		// Build the output content
		let outputContent = "";
		
		// Add header lines
		if (config.headerLines.length > 0) {
			outputContent += config.headerLines.join("\n") + "\n\n";
		}
		
		// Add the script content
		outputContent += scriptContent;
		
		// Add footer lines
		if (config.footerLines.length > 0) {
			if (!scriptContent.endsWith("\n")) {
				outputContent += "\n";
			}
			outputContent += "\n" + config.footerLines.join("\n") + "\n";
		}
		
		// Determine output path
		const outputPath = path.join(CWD, config.outputFile);
		const outputDir = path.dirname(outputPath);
		
		// Create output directory if it doesn't exist
		if (!fs.existsSync(outputDir)) {
			fs.mkdirSync(outputDir, { recursive: true });
		}
		
		// Write the file
		fs.writeFileSync(outputPath, outputContent, "utf-8");
		console.log(`✓ Extracted JavaScript to: ${config.outputFile}`);
		console.log(`  Output path: ${path.relative(CWD, outputPath)}\n`);
		
	} catch (error) {
		console.error(`Error: ${error.message}`);
		process.exit(1);
	}
};

// Process the Svelte file
processSvelteFile();

// Also extract HTML
console.log("\n");
import("./extract-html.js").then(({ extractHtml }) => {
	extractHtml();
	
	// Also extract CSS
	console.log("\n");
	return import("./extract-css.js");
}).then(({ extractCss }) => {
	extractCss();
}).catch((error) => {
	console.error(`Error running tasks: ${error.message}`);
	process.exit(1);
});

