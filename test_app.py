"""Pytest suite
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


def test_build_quiz_prompt_includes_difficulty_and_style() -> None:
    prompt = app.build_quiz_prompt(
        "Binary trees and traversals",
        difficulty="Hard",
        style="Exam-style",
    )

    assert "Difficulty level: Hard" in prompt
    assert "Question style: Exam-style" in prompt
    assert "Binary trees and traversals" in prompt


def test_extract_json_array_handles_extra_text() -> None:
    raw = "Some preface text\n[{\"k\":1}]\nSome trailing text"
    extracted = app.extract_json_array(raw)
    assert extracted == "[{\"k\":1}]"


def test_parse_quiz_json_raises_when_answer_not_in_options() -> None:
    raw = """
[
  {"question": "Q1", "options": ["A", "B", "C", "D"], "answer": "Z", "explanation": "E1"},
  {"question": "Q2", "options": ["A", "B", "C", "D"], "answer": "B", "explanation": "E2"},
  {"question": "Q3", "options": ["A", "B", "C", "D"], "answer": "C", "explanation": "E3"}
]
"""
    with pytest.raises(ValueError, match="answer must match one of its options"):
        app.parse_quiz_json(raw)


def test_score_answers_all_wrong() -> None:
    questions = sample_questions()
    selected = ["A loop type", "func", "Joined System Object Naming"]
    result = app.score_answers(questions, selected)

    assert result["score"] == 0
    assert result["total"] == 3
    assert all(not item["is_correct"] for item in result["results"])


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

    captured: dict[str, str] = {}

    class FakeModel:
        def __init__(self, _model_name: str):
            pass

        def generate_content(self, _prompt: str, generation_config: dict, request_options: dict):
            # Basic checks to ensure expected call pattern.
            captured["prompt"] = _prompt
            assert "temperature" in generation_config
            assert "timeout" in request_options
            return FakeResponse()

    monkeypatch.setattr(app.genai, "configure", lambda api_key: None)
    monkeypatch.setattr(app.genai, "GenerativeModel", FakeModel)

    questions = app.generate_quiz_questions(
        "Some notes",
        "fake-api-key",
        difficulty="Easy",
        style="Application",
    )

    assert len(questions) == 3
    assert questions[2]["answer"] == "C"
    assert "Difficulty level: Easy" in captured["prompt"]
    assert "Question style: Application" in captured["prompt"]
