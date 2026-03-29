from __future__ import annotations

import pandas as pd
import streamlit as st

from components.modern_ui import render_badges, render_metric_cards, render_page_header
from database import fetch_quiz_history, fetch_score_history, fetch_user_metrics


def render_dashboard_page(user_id: int, motivational_quotes: list[str]) -> None:
    render_page_header("Dashboard", "Track your learning performance with visual analytics.")

    metrics = fetch_user_metrics(user_id)
    render_metric_cards(
        total_quizzes=int(metrics["total_quizzes_taken"]),
        avg_score=float(metrics["average_score_percent"]),
        total_questions=int(metrics["total_questions_answered"]),
    )

    st.markdown("Average mastery progress")
    st.progress(min(max(metrics["average_score_percent"] / 100, 0.0), 1.0))

    overview_tab, topic_tab, momentum_tab = st.tabs(["Score Trend", "Topic Mastery", "Momentum"])

    with overview_tab:
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

    history = fetch_score_history(user_id)
    if not history:
        st.info("No quiz attempts yet. Generate and submit a quiz to see trend data.")
        return

    with topic_tab:
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

    with momentum_tab:
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
        render_badges(badges)

        quote = motivational_quotes[user_id % len(motivational_quotes)]
        st.info(f"💡 Study Coach: {quote}")
