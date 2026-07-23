# AI Agent Instructions

You are an expert Python developer working on this repository. Read and strictly adhere to these rules before writing or modifying any code.

## Project Overview

This repository shares tools developed by Jigsaw as a proof of concept to help make sense of large-scale online conversations. It demonstrates how Large Language Models (LLMs) like Gemini can be leveraged for such tasks. The code provided here offers a transparent look into Jigsaw's methods for categorization, summarization, and identifying points of agreement and disagreement in free response public opinion research. Our goal in sharing this is to inspire others by providing a potential starting point and useful elements for those tackling similar challenges.

More details can be found in the README.md file, along with instructions for running each step of the Sensemaking pipeline.

## Repository Boundaries & Rules
- **DO NOT touch the `case_studies/` directory.** Do not read from it, use it for context, or modify any files inside it. This is intended only to document past work, it does not need modification and should not be imported into other files.
- **Source Code Only:** All active development happens exclusively within the `src/` directory.
- **Google Style Guide:** All Python code must strictly follow standard [Google Python Style conventions](https://google.github.io/styleguide/pyguide.html). All functions, classes, and modules must include Google-style docstrings.

## Architecture & Tech Stack

### 1. LLM / Gemini Integration
We use a custom wrapper for all Gemini model interactions.
- **NEVER** import `google-generativeai`, `vertexai`, `litellm`, or other direct API libraries in the feature code.
- **ALWAYS** use the internal wrapper located at `src/models/genai_model.py`.
- *(Agent Note: If you are asked to implement a new LLM feature, first review `src/models/genai_model.py` to understand its expected inputs, outputs, and error handling).*

### 2. Data Handling (CSV)
- We use `pandas` exclusively for reading, writing, and manipulating CSV data.
- Do not use Python's built-in `csv` module unless explicitly requested.

## Executable Commands
Use these exact commands when verifying your work. Run them from the project root.

*   **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
*   **Run all tests:**
    ```bash
    pytest
    ```
*   **Run a specific test file:**
    ```bash
    pytest path/to/test_file.py
    ```
*   **Run a specific test function:**
    ```bash
    pytest path/to/test_file.py::test_function_name
    ```

## Agent Workflow
1. When asked to create a new file, place it in the appropriate subdirectory within `src/`.
2. Before presenting code, write unit tests for it using `pytest`.
3. If your code requires new dependencies, ask the user for permission before adding them to `requirements.txt`.
4. When asked to run, execute, or test the sensemaking pipeline, use the `/run-sensemaking` skill to guide the process step-by-step.