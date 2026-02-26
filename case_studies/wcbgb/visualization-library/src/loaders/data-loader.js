import * as d3 from "d3";

export class DataLoader {
  static async load(source, transform = true) {
    if (typeof source === "string") {
      try {
        if (source.startsWith("http")) {
          return await this._loadRemote(source, transform);
        } else {
          return await this._loadLocal(source, transform);
        }
      } catch (error) {
        console.error(`Error loading data from string source '${source}':`, error);
        return [];
      }
    } else if (source && (typeof source === "object" || Array.isArray(source))) {
      try {
        // Directly process the provided data object/array
        return this._transformData(JSON.parse(JSON.stringify(source)), transform); // Pass a clone to avoid unintended side-effects
      } catch (error) {
        console.error("Error processing direct data object:", error);
        return [];
      }
    } else {
      console.warn("DataLoader.load: Source is not a string or a valid data object/array.", source);
      return [];
    }
  }

  static _transformData(data, transform = true) {
    if (!transform) {
      // For non-transformed data (e.g., summary files)
      if (data && data.contents && Array.isArray(data.contents)) {
        return data.contents; // Return .contents if it's the typical summary structure
      }
      return data; // Otherwise, return data as-is
    }

    // For transformed data (e.g., main visualization data)
    let recordsToProcess = data;
    if (data && typeof data === "object" && !Array.isArray(data)) {
      recordsToProcess = Object.values(data);
    }

    if (!Array.isArray(recordsToProcess)) {
      console.warn(
        "_transformData: Expected an array for transformation, but received:",
        recordsToProcess
      );
      return [];
    }

    return (
      recordsToProcess
        // .filter((item) => item && !item.isFilteredOut) // Check if item itself is truthy
        .map((item) => {
          const transformedItem = { ...item };

          if (!item.subtopics && item.topics && typeof item.topics === "string") {
            const topicPairs = item.topics.split(";").map((s) => s.trim());
            const topics = [];
            const subtopics = [];

            topicPairs.forEach((topicPair) => {
              const [topic, subtopic] = topicPair.split(":").map((s) => s.trim());
              if (topic) {
                topics.push(topic);
                subtopics.push(subtopic || "");
              }
            });

            transformedItem.topics = topics;
            transformedItem.subtopics = subtopics;
          }

          // Ensure topics and subtopics are arrays
          transformedItem.topics = Array.isArray(transformedItem.topics)
            ? transformedItem.topics
            : [transformedItem.topics || ""];
          transformedItem.subtopics = Array.isArray(transformedItem.subtopics)
            ? transformedItem.subtopics
            : [transformedItem.subtopics || ""];

          return transformedItem;
        })
    );
  }

  static async _loadRemote(source, transform = true) {
    const data = await d3.json(source);
    return this._transformData(data, transform);
  }

  static async _loadLocal(source, transform = true) {
    const response = await fetch(source);
    const data = await response.json();
    return this._transformData(data, transform);
  }
}
