"""AI-Based Student Learning Support System

Streamlit application that takes student notes, calls Gemini, and generates
3 multiple-choice questions in JSON format. It then renders an interactive
quiz, tracks state with Streamlit session state, and scores answers.
"""

from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

MODEL_NAME = "gemini-2.5-flash"


def build_quiz_prompt(notes: str) -> str:
    """Create a strict prompt that requests exactly 3 MCQs as JSON array."""
    return f"""
You are a quiz generator for university students.

Task:
- Read the study notes provided by the user.
- Return EXACTLY 3 multiple-choice questions.
- Each question must include 4 options.
- Return ONLY valid JSON (no markdown, no extra text).

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
    """Extract a JSON array from model output.

    Handles cases where the model may wrap JSON in markdown code fences.
    """
    text = raw_text.strip()

    # If response is wrapped in markdown code fences, unwrap it safely.
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence if present.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to isolate the first JSON array if extra text appears.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]

    return text


def parse_quiz_json(raw_text: str) -> list[dict[str, Any]]:
    """Parse and validate Gemini JSON response.

    Raises:
        ValueError: If JSON is invalid or does not match expected schema.
    """
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


def generate_quiz_questions(notes: str, api_key: str, model_name: str = MODEL_NAME) -> list[dict[str, Any]]:
    """Call Gemini API and return validated quiz questions."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = build_quiz_prompt(notes)

    response = model.generate_content(
        prompt,
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


def initialize_session_state() -> None:
    """Initialize state keys once to prevent key errors."""
    defaults = {
        "questions": None,
        "quiz_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    """Run the Streamlit app."""
    load_dotenv()

    st.set_page_config(page_title="AI Learning Support System", layout="wide")
    st.title("📘 AI-Based Student Learning Support System")
    st.caption("Paste your notes, generate a 3-question quiz, and test your understanding.")

    initialize_session_state()

    # Sidebar configuration for API key.
    st.sidebar.header("Configuration")
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Enter your key here or define GEMINI_API_KEY in a .env file.",
    ).strip()

    st.subheader("1) Paste your study notes")
    notes = st.text_area(
        "Study notes",
        height=260,
        placeholder="Paste your lecture notes, summaries, or textbook excerpts here...",
    )

    if st.button("Generate Quiz", type="primary"):
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar or .env file.")
        elif not notes.strip():
            st.error("Please paste some study notes before generating a quiz.")
        else:
            with st.spinner("Generating quiz from your notes..."):
                try:
                    st.session_state["questions"] = generate_quiz_questions(notes, api_key)
                    st.session_state["quiz_result"] = None
                    st.success("Quiz generated successfully. Answer the questions below.")
                except ValueError as exc:
                    st.error(f"Could not generate a valid quiz: {exc}")
                except Exception as exc:  # Broad catch for network/API issues.
                    st.error(
                        "Something went wrong while contacting Gemini. "
                        f"Please try again. Details: {exc}"
                    )

    questions = st.session_state.get("questions")
    if questions:
        st.subheader("2) Answer the quiz")

        with st.form("quiz_form"):
            selected_answers: list[str | None] = []

            for idx, item in enumerate(questions, start=1):
                st.markdown(f"**Q{idx}. {item['question']}**")
                selected = st.radio(
                    f"Select your answer for Q{idx}",
                    options=item["options"],
                    index=None,
                    key=f"q_{idx}",
                )
                selected_answers.append(selected)

            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            if any(ans is None for ans in selected_answers):
                st.error("Please answer all questions before submitting.")
            else:
                st.session_state["quiz_result"] = score_answers(questions, selected_answers)

    quiz_result = st.session_state.get("quiz_result")
    if quiz_result:
        st.subheader("3) Feedback & scoring")
        st.info(f"You scored {quiz_result['score']}/{quiz_result['total']}!")

        for i, item in enumerate(quiz_result["results"], start=1):
            if item["is_correct"]:
                st.success(
                    f"Q{i}: Correct ✅\n\n"
                    f"Your answer: {item['chosen']}\n\n"
                    f"Explanation: {item['explanation']}"
                )
            else:
                st.warning(
                    f"Q{i}: Incorrect ⚠️\n\n"
                    f"Your answer: {item['chosen']}\n\n"
                    f"Correct answer: {item['correct']}\n\n"
                    f"Explanation: {item['explanation']}"
                )


if __name__ == "__main__":
    main()
