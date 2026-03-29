"""Session-state helper utilities."""

from __future__ import annotations

import streamlit as st


def initialize_session_state() -> None:
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
    for idx in range(1, question_count + 1):
        key = f"q_{idx}"
        if key in st.session_state:
            del st.session_state[key]


def reset_user_state() -> None:
    st.session_state["username"] = None
    st.session_state["user_id"] = None
    st.session_state["questions"] = None
    st.session_state["quiz_result"] = None
    st.session_state["active_quiz_id"] = None
    clear_quiz_widget_state()
