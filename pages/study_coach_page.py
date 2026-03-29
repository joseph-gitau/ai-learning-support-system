from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from components.ui import render_page_header
from database import fetch_quiz_history, fetch_study_plan_history, save_study_plan
from services.quiz_engine import generate_study_plan


def render_study_coach_page(user_id: int, api_key: str) -> None:
    render_page_header("Study Coach", "Get a focused improvement plan from your quiz performance.")

    history = fetch_quiz_history(user_id)
    if not history:
        st.info("Take at least one quiz first so Study Coach can analyze your weak areas.")
        return

    history_df = pd.DataFrame(history)
    weak_df = history_df[history_df["score_percent"] < 70].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Weak Attempts", int(len(weak_df)))
    c2.metric("Avg Score", f"{history_df['score_percent'].mean():.1f}%")
    c3.metric("Needs Review", int(weak_df["topic_preview"].nunique()) if not weak_df.empty else 0)

    st.markdown("### Weak Topic Signals")
    if weak_df.empty:
        st.success("Great work — no weak-topic patterns detected. Try Hard difficulty for growth.")
        return

    weak_df["topic_label"] = weak_df["topic_preview"].fillna("Untitled topic").str.slice(0, 45)
    weak_summary = (
        weak_df.groupby("topic_label", as_index=False)["score_percent"]
        .mean()
        .sort_values("score_percent", ascending=True)
        .head(8)
        .set_index("topic_label")
    )
    st.bar_chart(weak_summary)

    with st.expander("View weak-topic table"):
        st.dataframe(
            weak_summary.reset_index().rename(
                columns={"topic_label": "Topic", "score_percent": "Average Score %"}
            ),
            use_container_width=True,
        )

    st.markdown("### Build My Study Plan")
    goal = st.text_input(
        "Learning goal",
        placeholder="e.g., prepare for end-semester exam in data structures",
    )
    hours_per_week = st.slider("Hours available per week", min_value=2, max_value=25, value=8)

    if st.button("Generate AI Study Plan", type="primary"):
        if not api_key:
            st.error("Please set GEMINI_API_KEY in your .env file to generate the study plan.")
            return

        weak_topics = weak_summary.reset_index()["topic_label"].tolist()
        with st.spinner("Generating personalized study plan..."):
            try:
                plan = generate_study_plan(
                    weak_topics=weak_topics,
                    goal=goal,
                    hours_per_week=hours_per_week,
                    api_key=api_key,
                )
                st.session_state["study_plan"] = plan
                st.session_state["study_plan_meta"] = {
                    "goal": goal,
                    "hours_per_week": hours_per_week,
                    "weak_topics": weak_topics,
                }

                plan_id = save_study_plan(
                    user_id=user_id,
                    goal=goal,
                    hours_per_week=hours_per_week,
                    weak_topics=weak_topics,
                    plan_text=plan,
                )
                st.success(f"Study plan generated and saved (Plan #{plan_id}).")
            except Exception as exc:
                st.error(f"Could not generate study plan: {exc}")

    if st.session_state.get("study_plan"):
        st.markdown("### Your AI Study Plan")
        st.markdown(st.session_state["study_plan"])

        st.download_button(
            "Download Plan (.md)",
            data=st.session_state["study_plan"],
            file_name="study_plan.md",
            mime="text/markdown",
            use_container_width=True,
        )

        plan_meta = st.session_state.get("study_plan_meta", {})
        weak_topics = plan_meta.get("weak_topics", weak_summary.reset_index()["topic_label"].tolist())
        st.markdown("### Spaced-Repetition Reminders")

        revision_offsets = [1, 3, 7, 14]
        reminder_rows: list[dict[str, str]] = []
        for topic in weak_topics:
            for offset in revision_offsets:
                reminder_rows.append(
                    {
                        "Topic": topic,
                        "Review On": (date.today() + timedelta(days=offset)).isoformat(),
                        "Session": f"D+{offset}",
                    }
                )

        reminder_df = pd.DataFrame(reminder_rows)
        st.dataframe(reminder_df, use_container_width=True)

    st.markdown("### Saved Study Plans")
    plan_history = fetch_study_plan_history(user_id)
    if not plan_history:
        st.caption("No saved study plans yet.")
        return

    for plan in plan_history:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**Plan #{plan['plan_id']}**")
            c1.caption(
                f"Created: {plan['created_at']} • Goal: {plan['goal'] or 'N/A'} • Hours/week: {plan['hours_per_week']}"
            )
            if c2.button("View", key=f"view_plan_{plan['plan_id']}"):
                st.session_state["study_plan"] = plan["plan_text"]
                st.session_state["study_plan_meta"] = {
                    "goal": plan["goal"],
                    "hours_per_week": plan["hours_per_week"],
                    "weak_topics": plan["weak_topics"],
                }
                st.rerun()
