"""Quiz generation and scoring business logic."""

from __future__ import annotations

import json
from typing import Any

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash"


def build_quiz_prompt(
    notes: str,
    difficulty: str = "Mixed",
    style: str = "Conceptual",
) -> str:
    """Create a strict prompt that requests exactly 3 MCQs as JSON array."""
    return f"""
You are a quiz generator for university students.

Task:
- Read the study notes provided by the user.
- Return EXACTLY 3 multiple-choice questions.
- Each question must include 4 options.
- Return ONLY valid JSON (no markdown, no extra text).
- Difficulty level: {difficulty}
- Question style: {style}

Output format (JSON array):
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "Correct Option",
    "explanation": "Why the answer is correct"
  }}
]

Validation requirements:
- The top-level structure must be a JSON array of length 3.
- Each object must include keys: question, options, answer, explanation.
- options must contain exactly 4 strings.
- answer must exactly match one of the 4 options.

Study notes:
{notes}
""".strip()


def extract_json_array(raw_text: str) -> str:
    """Extract a JSON array from model output."""
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]

    return text


def parse_quiz_json(raw_text: str) -> list[dict[str, Any]]:
    """Parse and validate Gemini JSON response."""
    candidate = extract_json_array(raw_text)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Could not parse quiz JSON from AI response.") from exc

    if not isinstance(data, list) or len(data) != 3:
        raise ValueError("AI response must be a JSON array with exactly 3 questions.")

    required_keys = {"question", "options", "answer", "explanation"}
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {i} is not a valid JSON object.")

        if not required_keys.issubset(item.keys()):
            raise ValueError(
                f"Question {i} is missing one or more required fields: "
                "question, options, answer, explanation."
            )

        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Question {i} has an invalid question text.")

        options = item["options"]
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Question {i} must have exactly 4 options.")

        if not all(isinstance(opt, str) and opt.strip() for opt in options):
            raise ValueError(f"Question {i} has one or more invalid options.")

        if item["answer"] not in options:
            raise ValueError(f"Question {i} answer must match one of its options.")

        if not isinstance(item["explanation"], str) or not item["explanation"].strip():
            raise ValueError(f"Question {i} has an invalid explanation.")

    return data


def generate_quiz_questions(
    notes: str,
    api_key: str,
    model_name: str = MODEL_NAME,
    difficulty: str = "Mixed",
    style: str = "Conceptual",
) -> list[dict[str, Any]]:
    """Call Gemini API and return validated quiz questions."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        build_quiz_prompt(notes, difficulty=difficulty, style=style),
        generation_config={"temperature": 0.2},
        request_options={"timeout": 30},
    )

    if not getattr(response, "text", None):
        raise ValueError("AI returned an empty response. Please try again.")

    return parse_quiz_json(response.text)


def score_answers(
    questions: list[dict[str, Any]],
    selected_answers: list[str | None],
) -> dict[str, Any]:
    """Score user answers and return detailed result structure."""
    results: list[dict[str, Any]] = []
    correct_count = 0

    for question, chosen in zip(questions, selected_answers):
        correct_answer = question["answer"]
        is_correct = chosen == correct_answer

        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": question["question"],
                "chosen": chosen,
                "correct": correct_answer,
                "is_correct": is_correct,
                "explanation": question["explanation"],
            }
        )

    return {
        "score": correct_count,
        "total": len(questions),
        "results": results,
    }


def build_study_plan_prompt(
    weak_topics: list[str],
    goal: str,
    hours_per_week: int,
) -> str:
    """Build prompt for personalized study coaching plan."""
    topics = "\n".join(f"- {topic}" for topic in weak_topics) if weak_topics else "- No specific weak topics"
    return f"""
You are an academic study coach for a BSc Computer Science student.

Create a practical 2-week study plan in markdown format.

Constraints:
- Weekly availability: {hours_per_week} hours
- Student goal: {goal or 'Improve core understanding and quiz performance'}
- Prioritize these weak topics:
{topics}

Output format:
1. Short diagnostic summary
2. 2-week schedule (day-by-day tasks)
3. Active-recall quiz routine
4. Revision checklist
5. Success criteria with measurable targets

Tone: concise, encouraging, and actionable.
""".strip()


def generate_study_plan(
    weak_topics: list[str],
    goal: str,
    hours_per_week: int,
    api_key: str,
    model_name: str = MODEL_NAME,
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        build_study_plan_prompt(
            weak_topics=weak_topics,
            goal=goal,
            hours_per_week=hours_per_week,
        ),
        generation_config={"temperature": 0.4},
        request_options={"timeout": 40},
    )

    if not getattr(response, "text", None):
        raise ValueError("AI returned an empty study plan response.")

    return response.text.strip()
