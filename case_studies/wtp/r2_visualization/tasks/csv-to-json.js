import fs from "fs";
import path from "path";

const CWD = process.cwd();

// CSV files to convert
const csvFiles = [
	{
		csvFile: "src/data/opinions.csv",
		jsonFile: "src/data/opinions.json"
	},
	{
		csvFile: "src/data/quotes.csv",
		jsonFile: "src/data/quotes.json"
	}
];

// Parse CSV line handling quoted fields
const parseCSVLine = (line) => {
	const fields = [];
	let currentField = "";
	let insideQuotes = false;
	
	for (let i = 0; i < line.length; i++) {
		const char = line[i];
		const nextChar = line[i + 1];
		
		if (char === '"') {
			if (insideQuotes && nextChar === '"') {
				// Escaped quote (double quote)
				currentField += '"';
				i++; // Skip next quote
			} else {
				// Toggle quote state
				insideQuotes = !insideQuotes;
			}
		} else if (char === ',' && !insideQuotes) {
			// Field separator
			fields.push(currentField);
			currentField = "";
		} else {
			currentField += char;
		}
	}
	
	// Push last field
	fields.push(currentField);
	
	return fields;
};

// Convert CSV to JSON
const csvToJson = (csvContent) => {
	const lines = csvContent.split(/\r?\n/).filter(line => line.trim() !== "");
	
	if (lines.length === 0) {
		return [];
	}
	
	// Parse header
	const headers = parseCSVLine(lines[0]);
	
	// Parse data rows
	const data = lines.slice(1).map(line => {
		const values = parseCSVLine(line);
		const row = {};
		
		headers.forEach((header, index) => {
			let value = values[index] || "";
			
			// Try to parse numbers
			if (value !== "" && !isNaN(value) && value.trim() !== "") {
				const numValue = Number(value);
				if (!isNaN(numValue)) {
					value = numValue;
				}
			}
			
			row[header] = value;
		});
		
		return row;
	});
	
	return data;
};

// Process each CSV file
try {
	for (const { csvFile, jsonFile } of csvFiles) {
		console.log(`Converting ${csvFile} to ${jsonFile}...`);
		
		// Read CSV file
		const csvContent = fs.readFileSync(path.join(CWD, csvFile), "utf-8");
		
		// Convert to JSON
		const jsonData = csvToJson(csvContent);
		
		// Write JSON file with pretty formatting
		const jsonContent = JSON.stringify(jsonData, null, "\t");
		fs.writeFileSync(path.join(CWD, jsonFile), jsonContent, "utf-8");
		
		console.log(`✓ Converted ${csvFile} to ${jsonFile}`);
		console.log(`  Rows: ${jsonData.length}`);
		console.log();
	}
	
	console.log("✓ All conversions complete!");
	
} catch (error) {
	console.error(`Error: ${error.message}`);
	process.exit(1);
}

