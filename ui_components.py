"""Reusable visual components for Streamlit pages."""

from __future__ import annotations

import streamlit as st


def apply_custom_css() -> None:
    """Inject lightweight custom CSS for a richer visual prototype."""
    st.markdown(
        """
        <style>
        .hero-card {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: linear-gradient(90deg, #5B6CFF 0%, #7D8BFF 100%);
            color: white;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 20px rgba(91,108,255,0.2);
        }
        .hero-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.95;
        }
        .soft-card {
            background: #FFFFFF;
            border: 1px solid #E6EAFD;
            border-radius: 12px;
            padding: 0.8rem;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: #EEF1FF;
            color: #3543B8;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_branding() -> None:
    """Render sidebar branding block."""
    st.sidebar.markdown("### 🎓 AI Learning Studio")
    st.sidebar.caption("Design • Build • Test")


def render_hero(title: str, subtitle: str) -> None:
    """Render gradient hero banner."""
    st.markdown(
        (
            '<div class="hero-card">'
            f'<div class="hero-title">{title}</div>'
            f'<div class="hero-subtitle">{subtitle}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_row(labels: list[str]) -> None:
    """Render simple badge/pill row."""
    if not labels:
        st.caption("No badges unlocked yet.")
        return

    html = "".join(f'<span class="pill">{label}</span>' for label in labels)
    st.markdown(html, unsafe_allow_html=True)


def render_system_status(model_name: str, api_key_loaded: bool) -> None:
    """Render compact system status widget in sidebar."""
    with st.sidebar.expander("System status"):
        st.write(f"Model: {model_name}")
        st.write("Auth mode: .env only")
        st.write("Gemini key loaded" if api_key_loaded else "Gemini key missing (.env)")


def render_page_header(title: str, caption: str) -> None:
    """Render consistent per-page heading."""
    st.markdown(f"## {title}")
    st.caption(caption)
