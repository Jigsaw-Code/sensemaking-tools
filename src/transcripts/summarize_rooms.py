"""Module to summarize room transcripts using Gemini."""

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any

from src.models import genai_model


PROMPT_TEMPLATE = """You are analyzing a room's transcript. The transcript begins with a header that lists the room's topic and members, followed by the transcript itself.
Your task is to convert the transcript into a per-user first-person statement that approximates the views they expressed in their room.
For participants who did not provide a view or say anything meaningful, you must output exactly "Also present" as their statement.

Return a JSON object where the keys are the participant names (extracted from the members list or transcript) and the values are their first-person statements.

Transcript:
{transcript}
"""


def parse_transcript(file_path: str) -> list[dict[str, str]]:
    """Parses a transcript file into room sections.

    Args:
        file_path: Path to the transcript file.

    Returns:
        A list of dictionaries, each containing a 'header' and 'content' for a room.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split based on "===== Room:"
    parts = re.split(r"(===== Room:.*?=====)", content)

    rooms = []
    # parts[0] is everything before the first room (could be empty or preamble)
    for i in range(1, len(parts), 2):
        header = parts[i]
        text = parts[i+1] if i+1 < len(parts) else ""
        room_content = header + "\n" + text.strip()
        rooms.append({"header": header, "content": room_content})
    return rooms


def response_parser(resp: dict[str, Any], job: dict[str, Any]) -> dict[str, str]:
    """Parses the response from the GenaiModel.

    Args:
        resp: The response dictionary from the model.
        job: The job dictionary.

    Returns:
        The parsed dictionary of user statements.

    Raises:
        ValueError: If the API returns an error or JSON parsing fails.
    """
    if "error" in resp and resp["error"]:
        raise ValueError(f"API Error: {resp['error']}")
    text = resp.get("text", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}. Raw text: {repr(text)}")
        raise ValueError(f"Failed to parse JSON: {e}")


async def summarize_rooms(transcript_path: str, output_path: str, model_name: str = "gemini-3.5-flash"):
    """Reads transcripts, queries Gemini for summaries, and writes output.

    Args:
        transcript_path: Path to the input transcript.
        output_path: Path to write the JSON results.
        model_name: Name of the Gemini model to use.
    """
    rooms = parse_transcript(transcript_path)
    logging.info("Found %d rooms.", len(rooms))

    model = genai_model.GenaiModel(model_name=model_name)

    prompts = []
    for room in rooms:
        prompt_text = PROMPT_TEMPLATE.format(transcript=room["content"])
        prompts.append({
            "prompt": prompt_text,
            "room_header": room["header"],
            "response_mime_type": "application/json",
        })

    logging.info("Sending transcripts to Gemini...")
    llm_response, _, _, _ = await model.process_prompts_concurrently(
        prompts=prompts,
        response_parser=response_parser,
        max_concurrent_calls=5,
    )

    results = []
    for _, row in llm_response.iterrows():
        results.append({
            "room_header": row["room_header"],
            "summaries": row["result"]
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("Saved results to %s", output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Summarize room transcripts")
    parser.add_argument(
        "--transcript",
        type=str,
        default="experimental/transcription/transcript.txt",
        help="Path to transcript file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experimental/transcription/room_summaries.json",
        help="Path to output file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Name of the Gemini model to use"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(summarize_rooms(args.transcript, args.output, args.model))


if __name__ == "__main__":
    main()
