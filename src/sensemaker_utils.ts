// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Simple utils.

/**
 * Combines the data and instructions into a prompt to send to Vertex.
 * @param instructions: what the model should do.
 * @param data: the data that the model should consider.
 * @param additionalInstructions additional context to include in the prompt.
 * @returns the instructions and the data as a text
 */
export function getPrompt(instructions: string, data: string[], additionalInstructions?: string) {
  return `Instructions:
${instructions}
${additionalInstructions ? "\nAdditional context:\n" + additionalInstructions + "\n" : ""}
Comments:
${data.join("\n")}`; // separate comments with newlines
}
