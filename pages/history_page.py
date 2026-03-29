"""Quiz history page rendering."""

from __future__ import annotations

import streamlit as st

from components.modern_ui import render_page_header
from components.session_state import clear_quiz_widget_state
from database import fetch_quiz_by_id, fetch_quiz_history


def render_quiz_history_page(user_id: int) -> None:
    """Display previous quizzes and allow retake without API call."""
    render_page_header("Quiz History", "Review attempts and retake quizzes instantly.")

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
