import fs from "fs";
import path from "path";

const CWD = process.cwd();

// Configuration
const config = {
	// Svelte file to extract HTML from
	svelteFile: "src/components/Index.svelte",
	
	// Output HTML file path
	outputFile: "qualtrics/index.html"
};

const extractHtmlFromSvelte = (filePath) => {
	const fullPath = path.join(CWD, filePath);
	
	if (!fs.existsSync(fullPath)) {
		throw new Error(`File not found: ${filePath}`);
	}
	
	const content = fs.readFileSync(fullPath, "utf-8");
	
	// Extract content after </script> tag
	// Match everything after the closing </script> tag
	const scriptEndMatch = content.indexOf("</script>");
	
	if (scriptEndMatch === -1) {
		throw new Error(`No </script> tag found in ${filePath}`);
	}
	
	// Get everything after </script>
	let htmlContent = content.substring(scriptEndMatch + "</script>".length);
	
	// Remove leading/trailing whitespace
	htmlContent = htmlContent.trim();
	
	return htmlContent;
};

const processSvelteFile = () => {
	console.log(`Extracting HTML from ${config.svelteFile}...\n`);
	
	try {
		// Extract the HTML content
		let htmlContent = extractHtmlFromSvelte(config.svelteFile);
		
		// Remove Svelte-specific tags that won't work in plain HTML
		// Remove <svelte:boundary> tags (including multiline with attributes) but keep their content
		htmlContent = htmlContent.replace(/<svelte:boundary[\s\S]*?>/g, "");
		htmlContent = htmlContent.replace(/<\/svelte:boundary>/g, "");
		
		// Remove empty lines
		htmlContent = htmlContent
			.split("\n")
			.filter(line => line.trim() !== "")
			.join("\n");
		
		// Determine output path
		const outputPath = path.join(CWD, config.outputFile);
		const outputDir = path.dirname(outputPath);
		
		// Create output directory if it doesn't exist
		if (!fs.existsSync(outputDir)) {
			fs.mkdirSync(outputDir, { recursive: true });
		}
		
		// Write the file
		fs.writeFileSync(outputPath, htmlContent, "utf-8");
		console.log(`✓ Extracted HTML to: ${config.outputFile}`);
		console.log(`  Output path: ${path.relative(CWD, outputPath)}\n`);
		
	} catch (error) {
		console.error(`Error: ${error.message}`);
		process.exit(1);
	}
};

// Export the function so it can be called from other scripts
export { processSvelteFile as extractHtml };

// Execute if run directly (check if this file is the main module)
const isMainModule = import.meta.url === `file://${path.resolve(process.argv[1])}` || 
                     process.argv[1] && process.argv[1].endsWith('extract-html.js');

if (isMainModule) {
	processSvelteFile();
	console.log("Done!");
}

