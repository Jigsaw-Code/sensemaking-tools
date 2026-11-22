"""Tests for summarize_rooms.py."""

import json
import pytest

from src.transcripts import summarize_rooms


def test_parse_transcript(tmp_path):
    transcript_content = (
        "Some initial text\n"
        "===== Room: Topic A =====\n"
        "User1: Hello\n"
        "User2: Hi\n"
        "===== Room: Topic B =====\n"
        "User3: Test"
    )
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text(transcript_content)

    rooms = summarize_rooms.parse_transcript(str(transcript_file))
    assert len(rooms) == 2
    assert rooms[0]["header"] == "===== Room: Topic A ====="
    assert "User1: Hello" in rooms[0]["content"]
    assert rooms[1]["header"] == "===== Room: Topic B ====="
    assert "User3: Test" in rooms[1]["content"]


def test_response_parser_success():
    resp = {"text": '{"User1": "I said hello"}'}
    job = {}
    result = summarize_rooms.response_parser(resp, job)
    assert result == {"User1": "I said hello"}


def test_response_parser_error():
    resp = {"error": "API Error"}
    job = {}
    with pytest.raises(ValueError, match="API Error"):
        summarize_rooms.response_parser(resp, job)


def test_response_parser_invalid_json():
    resp = {"text": "This is not JSON"}
    job = {}
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        summarize_rooms.response_parser(resp, job)
