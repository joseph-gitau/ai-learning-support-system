"""Pytest suite for core app logic.

These tests avoid live API calls by mocking the Gemini client.
"""

from __future__ import annotations

import pytest

import app


def sample_questions() -> list[dict]:
    return [
        {
            "question": "What is a variable?",
            "options": [
                "A named storage location",
                "A loop type",
                "A file format",
                "A hardware component",
            ],
            "answer": "A named storage location",
            "explanation": "Variables store values using names.",
        },
        {
            "question": "Which keyword defines a function in Python?",
            "options": ["func", "define", "def", "lambda"],
            "answer": "def",
            "explanation": "The def keyword starts a function definition.",
        },
        {
            "question": "What does JSON stand for?",
            "options": [
                "Java Source Open Network",
                "JavaScript Object Notation",
                "Joined System Object Naming",
                "Just Simple Object Notation",
            ],
            "answer": "JavaScript Object Notation",
            "explanation": "JSON means JavaScript Object Notation.",
        },
    ]


def test_score_answers_counts_correctly() -> None:
    questions = sample_questions()
    selected = [
        "A named storage location",  # correct
        "lambda",  # incorrect
        "JavaScript Object Notation",  # correct
    ]

    result = app.score_answers(questions, selected)

    assert result["score"] == 2
    assert result["total"] == 3
    assert result["results"][0]["is_correct"] is True
    assert result["results"][1]["is_correct"] is False


def test_parse_quiz_json_accepts_markdown_wrapped_json() -> None:
    raw = """```json
[
  {
    "question": "Q1",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "explanation": "Because A"
  },
  {
    "question": "Q2",
    "options": ["A", "B", "C", "D"],
    "answer": "B",
    "explanation": "Because B"
  },
  {
    "question": "Q3",
    "options": ["A", "B", "C", "D"],
    "answer": "C",
    "explanation": "Because C"
  }
]
```"""

    parsed = app.parse_quiz_json(raw)

    assert len(parsed) == 3
    assert parsed[1]["answer"] == "B"


def test_parse_quiz_json_raises_for_invalid_schema() -> None:
    raw = "[{\"question\": \"Only one item\"}]"

    with pytest.raises(ValueError):
        app.parse_quiz_json(raw)


def test_generate_quiz_questions_with_mocked_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response_text = """
[
  {"question": "Q1", "options": ["A", "B", "C", "D"], "answer": "A", "explanation": "E1"},
  {"question": "Q2", "options": ["A", "B", "C", "D"], "answer": "B", "explanation": "E2"},
  {"question": "Q3", "options": ["A", "B", "C", "D"], "answer": "C", "explanation": "E3"}
]
"""

    class FakeResponse:
        text = fake_response_text

    class FakeModel:
        def __init__(self, _model_name: str):
            pass

        def generate_content(self, _prompt: str, generation_config: dict, request_options: dict):
            # Basic checks to ensure expected call pattern.
            assert "temperature" in generation_config
            assert "timeout" in request_options
            return FakeResponse()

    monkeypatch.setattr(app.genai, "configure", lambda api_key: None)
    monkeypatch.setattr(app.genai, "GenerativeModel", FakeModel)

    questions = app.generate_quiz_questions("Some notes", "fake-api-key")

    assert len(questions) == 3
    assert questions[2]["answer"] == "C"
