"""AI-Based Student Learning Support System (Phase 2).

This Streamlit app includes:
- Sidebar multi-page navigation (Dashboard, Generate Quiz, Quiz History)
- Lightweight user login (username-based)
- SQLite persistence for quizzes/results
- User-level performance dashboard and score trend
- Quiz history with retake support (no API re-call needed)
"""

from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import (
    fetch_quiz_by_id,
    fetch_quiz_history,
    fetch_score_history,
    fetch_user_metrics,
    get_or_create_user,
    init_db,
    save_quiz_attempt,
)

MODEL_NAME = "gemini-2.5-flash"
MOTIVATIONAL_QUOTES = [
    "Small progress every day compounds into big mastery.",
    "Learn deeply, not quickly — depth wins exams and projects.",
    "Mistakes are feedback loops, not failures.",
    "Consistency beats intensity in long-term learning.",
]

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
    """Extract a JSON array from model output.

    Handles cases where the model may wrap JSON in markdown code fences.
    """
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

def initialize_session_state() -> None:
    """Initialize session keys once to prevent key errors."""
    defaults = {
        "username": None,
        "user_id": None,
        "page": "Dashboard",
        "questions": None,
        "source_text": "",
        "active_quiz_id": None,
        "quiz_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_quiz_widget_state(question_count: int = 10) -> None:
    """Clear radio widget keys so retakes/new quizzes start cleanly."""
    for idx in range(1, question_count + 1):
        key = f"q_{idx}"
        if key in st.session_state:
            del st.session_state[key]

def render_user_login() -> None:
    """Render lightweight username login in sidebar and persist active user."""
    st.sidebar.header("User")

    if st.session_state["username"]:
        st.sidebar.success(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Switch User"):
            st.session_state["username"] = None
            st.session_state["user_id"] = None
            st.session_state["questions"] = None
            st.session_state["quiz_result"] = None
            st.session_state["active_quiz_id"] = None
            clear_quiz_widget_state()
            st.rerun()
        return

    username = st.sidebar.text_input(
        "Username",
        placeholder="e.g., student_001",
        help="Simple mock login for local use. Data is tracked per username.",
    ).strip()

    if st.sidebar.button("Login"):
        if not username:
            st.sidebar.error("Please enter a username.")
        else:
            user_id = get_or_create_user(username)
            st.session_state["username"] = username
            st.session_state["user_id"] = user_id
            st.sidebar.success("Login successful.")
            st.rerun()

def render_navigation() -> str:
    """Render sidebar navigation and return selected page."""
    pages = ["Dashboard", "Generate Quiz", "Quiz History"]
    selected = st.sidebar.radio(
        "Navigate",
        options=pages,
        index=pages.index(st.session_state.get("page", "Dashboard")),
    )
    st.session_state["page"] = selected
    return selected

def render_dashboard(user_id: int) -> None:
    """Display user performance metrics and trend chart."""
    st.subheader("Dashboard")
    st.caption("Track your learning performance over time.")

    metrics = fetch_user_metrics(user_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Quizzes Taken", int(metrics["total_quizzes_taken"]))
    c2.metric("Average Score", f"{metrics['average_score_percent']:.1f}%")
    c3.metric("Total Questions Answered", int(metrics["total_questions_answered"]))

    st.markdown("### Score Trend")
    history = fetch_score_history(user_id)
    if not history:
        st.info("No quiz attempts yet. Generate and submit a quiz to see trend data.")
        return

    chart_df = pd.DataFrame(history)
    chart_df["attempt_date"] = pd.to_datetime(chart_df["attempt_date"])
    chart_df = chart_df.sort_values("attempt_date").set_index("attempt_date")
    st.line_chart(chart_df[["score_percent"]])

    with st.expander("View trend data table"):
        st.dataframe(
            chart_df.reset_index().rename(
                columns={
                    "attempt_date": "Date",
                    "score": "Score",
                    "total_questions": "Total Questions",
                    "score_percent": "Score %",
                }
            ),
            use_container_width=True,
        )

    st.markdown("### Topic Mastery Snapshot")
    topic_history = fetch_quiz_history(user_id)
    if topic_history:
        topic_df = pd.DataFrame(topic_history)
        topic_df["topic_label"] = topic_df["topic_preview"].fillna("Untitled topic").str.slice(0, 40)
        topic_mastery = (
            topic_df.groupby("topic_label", as_index=False)["score_percent"]
            .mean()
            .sort_values("score_percent", ascending=False)
            .head(8)
            .set_index("topic_label")
        )
        st.bar_chart(topic_mastery)
        with st.expander("View topic mastery table"):
            st.dataframe(
                topic_mastery.reset_index().rename(
                    columns={
                        "topic_label": "Topic",
                        "score_percent": "Average Score %",
                    }
                ),
                use_container_width=True,
            )
    else:
        st.info("Topic mastery will appear after your first completed quiz.")

    latest = history[-1]["score_percent"]
    best = max(row["score_percent"] for row in history)
    streak = 0
    for row in reversed(history):
        if row["score_percent"] >= 70:
            streak += 1
        else:
            break

    st.markdown("### Learning Momentum")
    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Score", f"{latest:.1f}%")
    m2.metric("Best Score", f"{best:.1f}%")
    m3.metric("Success Streak (≥70%)", streak)

    badges: list[str] = []
    if metrics["total_quizzes_taken"] >= 1:
        badges.append("🚀 Starter")
    if metrics["total_quizzes_taken"] >= 5:
        badges.append("📚 Consistent Learner")
    if best >= 90:
        badges.append("🏆 High Achiever")
    if streak >= 3:
        badges.append("🔥 On a Roll")

    st.markdown("### Badges")
    st.write(" ".join(badges) if badges else "No badges yet — take a quiz to begin.")

    quote = MOTIVATIONAL_QUOTES[user_id % len(MOTIVATIONAL_QUOTES)]
    st.info(f"💡 Study Coach: {quote}")

def render_quiz_form() -> None:
    """Render active quiz form and persist result after submission."""
    questions = st.session_state.get("questions")
    if not questions:
        st.info("No active quiz yet. Generate one first or retake from history.")
        return

    st.subheader("Answer the quiz")
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
            return

        result = score_answers(questions, selected_answers)
        st.session_state["quiz_result"] = result

        save_quiz_attempt(
            user_id=st.session_state["user_id"],
            source_text=st.session_state.get("source_text", ""),
            questions=questions,
            score=result["score"],
            total_questions=result["total"],
            selected_answers=selected_answers,
        )

def render_quiz_feedback() -> None:
    """Show score summary and per-question explanations."""
    quiz_result = st.session_state.get("quiz_result")
    if not quiz_result:
        return

    st.subheader("Feedback & Scoring")
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

    wrong_items = [item for item in quiz_result["results"] if not item["is_correct"]]
    with st.expander("Revision Cards"):
        if not wrong_items:
            st.success("Excellent work. No weak spots detected in this attempt.")
        else:
            st.write("Focus on these concepts before your next attempt:")
            for item in wrong_items:
                st.markdown(f"- **{item['question']}** → {item['correct']}")

    score_percent = (quiz_result["score"] / quiz_result["total"]) * 100 if quiz_result["total"] else 0
    if score_percent >= 80:
        st.balloons()
        st.success("Great performance. Try a harder difficulty next.")
    elif score_percent >= 50:
        st.info("Good attempt. Revisit revision cards and retake to improve.")
    else:
        st.warning("Keep going. Start with Easy mode and build confidence.")

def render_generate_quiz(api_key: str) -> None:
    """Render quiz generation page and active quiz interaction."""
    st.subheader("Generate Quiz")
    st.caption("Paste study notes to generate a 3-question quiz with Gemini.")

    notes = st.text_area(
        "Study notes",
        value=st.session_state.get("source_text", ""),
        height=240,
        placeholder="Paste your lecture notes, summaries, or textbook excerpts here...",
    )

    with st.expander("Advanced settings"):
        difficulty = st.selectbox(
            "Difficulty",
            options=["Easy", "Mixed", "Hard"],
            index=1,
        )
        style = st.selectbox(
            "Question style",
            options=["Conceptual", "Application", "Exam-style"],
            index=0,
        )
        st.write(f"Model: {MODEL_NAME}")
        st.write("Generation timeout: 30 seconds")
        st.write("Temperature: 0.2")

    if st.button("Generate Quiz", type="primary"):
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar or .env file.")
        elif not notes.strip():
            st.error("Please paste some study notes before generating a quiz.")
        else:
            with st.spinner("Generating quiz from your notes..."):
                try:
                    questions = generate_quiz_questions(
                        notes,
                        api_key,
                        difficulty=difficulty,
                        style=style,
                    )
                    st.session_state["questions"] = questions
                    st.session_state["source_text"] = notes
                    st.session_state["active_quiz_id"] = None
                    st.session_state["quiz_result"] = None
                    clear_quiz_widget_state(question_count=len(questions))
                    st.success("Quiz generated successfully. Answer the questions below.")
                except ValueError as exc:
                    st.error(f"Could not generate a valid quiz: {exc}")
                except Exception as exc:
                    st.error(
                        "Something went wrong while contacting Gemini. "
                        f"Please try again. Details: {exc}"
                    )

    render_quiz_form()
    render_quiz_feedback()

def render_quiz_history(user_id: int) -> None:
    """Display previous quizzes and allow retake without API call."""
    st.subheader("Quiz History")
    st.caption("Review previous attempts and retake any saved quiz.")

    history = fetch_quiz_history(user_id)
    if not history:
        st.info("No quiz history yet.")
        return

    for item in history:
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 2, 1])
            c1.markdown(f"**Topic preview:** {item['topic_preview']}")
            c1.caption(f"Created: {item['created_at']}")
            c2.metric("Last Score", f"{item['score']}/{item['total_questions']}")
            c2.caption(f"{item['score_percent']:.1f}%")

            if c3.button("Retake", key=f"retake_{item['quiz_id']}"):
                quiz = fetch_quiz_by_id(user_id=user_id, quiz_id=item["quiz_id"])
                if not quiz:
                    st.error("Could not load this quiz from the database.")
                else:
                    st.session_state["questions"] = quiz["questions"]
                    st.session_state["source_text"] = quiz["source_text"]
                    st.session_state["active_quiz_id"] = quiz["quiz_id"]
                    st.session_state["quiz_result"] = None
                    clear_quiz_widget_state(question_count=len(quiz["questions"]))
                    st.session_state["page"] = "Generate Quiz"
                    st.success("Quiz loaded. Redirecting to Generate Quiz...")
                    st.rerun()

def main() -> None:
    """Run the Streamlit app."""
    load_dotenv()
    init_db()

    st.set_page_config(page_title="AI Learning Support System", layout="wide")
    st.title("📘 AI-Based Student Learning Support System")
    st.caption("Design • Build • Test lifecycle with personalized learning analytics.")

    initialize_session_state()
    render_user_login()

    if not st.session_state["user_id"]:
        st.info("Please login from the sidebar to continue.")
        st.stop()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    with st.sidebar.expander("System status"):
        st.write(f"Model: {MODEL_NAME}")
        st.write("Auth mode: .env only")
        st.write("Gemini key loaded" if api_key else "Gemini key missing (.env)")

    page = render_navigation()
    if page == "Dashboard":
        render_dashboard(st.session_state["user_id"])
    elif page == "Generate Quiz":
        render_generate_quiz(api_key)
    elif page == "Quiz History":
        render_quiz_history(st.session_state["user_id"])

if __name__ == "__main__":
    main()
