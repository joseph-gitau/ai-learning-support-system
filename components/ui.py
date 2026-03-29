from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_elements import elements, mui

def apply_styles(css_path: str) -> None:
    css_file = Path(css_path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def render_top_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="top-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_top_header_bar(username: str | None, api_ready: bool) -> None:
    with elements("top_header_bar"):
        with mui.Paper(
            elevation=0,
            sx={
                "mb": 1,
                "p": 1.2,
                "borderRadius": 3,
                "border": "1px solid #E5E9FF",
                "background": "#FFFFFF",
            },
        ):
            with mui.Stack(direction="row", justifyContent="space-between", alignItems="center"):
                with mui.Stack(direction="row", spacing=1.2, alignItems="center"):
                    mui.icon.AutoAwesome(color="primary")
                    mui.Typography("AI Learning Support", variant="h6", sx={"fontWeight": 700})
                with mui.Stack(direction="row", spacing=1, alignItems="center"):
                    mui.Chip(
                        icon=mui.icon.Person(),
                        label=f"User: {username or 'Guest'}",
                        variant="outlined",
                        color="primary",
                    )
                    mui.Chip(
                        icon=mui.icon.Key(),
                        label="Gemini Ready" if api_ready else "Gemini Missing",
                        color="success" if api_ready else "warning",
                        variant="filled",
                    )

def render_metric_cards(total_quizzes: int, avg_score: float, total_questions: int) -> None:
    with elements("kpi_cards"):
        with mui.Grid(container=True, spacing=2):
            for icon_name, title, value in [
                ("Quiz", "Total Quizzes", str(total_quizzes)),
                ("TrendingUp", "Average Score", f"{avg_score:.1f}%"),
                ("AutoStories", "Total Questions", str(total_questions)),
            ]:
                with mui.Grid(item=True, xs=12, md=4):
                    with mui.Paper(
                        elevation=0,
                        sx={
                            "p": 1.5,
                            "borderRadius": 3,
                            "border": "1px solid #E7EBFF",
                            "height": "100%",
                            "background": "#FFF",
                        },
                    ):
                        with mui.Stack(direction="row", spacing=1.2, alignItems="center"):
                            getattr(mui.icon, icon_name)(color="primary")
                            with mui.Box():
                                mui.Typography(title, variant="body2", sx={"color": "#697094"})
                                mui.Typography(value, variant="h5", sx={"fontWeight": 700})

def render_badges(badges: list[str]) -> None:
    with elements("badge_row"):
        with mui.Stack(direction="row", spacing=1, flexWrap="wrap", useFlexGap=True):
            if not badges:
                mui.Chip(label="No badges yet", variant="outlined")
                return
            for badge in badges:
                mui.Chip(label=badge, color="secondary", variant="outlined")

def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f"### {title}")
    st.caption(subtitle)

def render_top_nav(pages: list[str], current_page: str) -> str:
    selected_page = current_page
    columns = st.columns(len(pages))
    for idx, page in enumerate(pages):
        button_type = "primary" if page == current_page else "secondary"
        if columns[idx].button(page, key=f"top_nav_{page}", use_container_width=True, type=button_type):
            selected_page = page
    return selected_page
