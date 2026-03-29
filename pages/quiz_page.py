from __future__ import annotations

import streamlit as st

from components.ui import render_page_header
from components.session_state import clear_quiz_widget_state
from database import fetch_user_metrics, save_quiz_attempt
from services.quiz_engine import generate_quiz_questions, score_answers


def _render_quiz_form() -> None:
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


def _render_quiz_feedback() -> None:
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


def render_generate_quiz_page(api_key: str, model_name: str) -> None:
    render_page_header("Generate Quiz", "Turn your notes into an interactive AI practice session.")

    left, right = st.columns([2, 1])

    metrics = fetch_user_metrics(st.session_state["user_id"])
    avg_score = float(metrics["average_score_percent"])
    if avg_score >= 80:
        recommended_difficulty = "Hard"
    elif avg_score >= 50:
        recommended_difficulty = "Mixed"
    else:
        recommended_difficulty = "Easy"

    st.info(f"Adaptive recommendation: **{recommended_difficulty}** difficulty based on your average score {avg_score:.1f}%.")

    with left:
        notes = st.text_area(
            "Study notes",
            value=st.session_state.get("source_text", ""),
            height=260,
            placeholder="Paste your lecture notes, summaries, or textbook excerpts here...",
        )

    with right:
        with st.container(border=True):
            st.markdown("#### Quiz Settings")
            difficulty = st.selectbox(
                "Difficulty",
                options=["Easy", "Mixed", "Hard"],
                index=["Easy", "Mixed", "Hard"].index(recommended_difficulty),
            )
            style = st.selectbox(
                "Question style",
                options=["Conceptual", "Application", "Exam-style"],
                index=0,
            )
            st.caption(f"Model: {model_name}")
            st.caption("Timeout: 30s • Temperature: 0.2")

    if st.button("Generate Quiz", type="primary"):
        if not api_key:
            st.error("Please set GEMINI_API_KEY in your .env file.")
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

    _render_quiz_form()
    _render_quiz_feedback()
