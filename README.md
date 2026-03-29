# AI-Based Student Learning Support System

A System prototype that helps students convert study notes into AI-generated quizzes, track learning performance, and receive personalized study coaching.

## What this project does

This system uses **Streamlit** for UI, **Google Gemini (gemini-2.5-flash)** for AI generation, and **SQLite** for local persistence.

Core capabilities:

- **Quiz generation from notes** (3 MCQs, JSON-validated)
- **Auto-grading with explanations**
- **Per-user tracking** (lightweight username login)
- **Dashboard analytics**
  - total quizzes
  - average score
  - trend over time
  - topic mastery
- **Quiz history + retake**
- **Study Coach page**
  - weak-topic detection
  - AI-generated 2-week study plan
  - saved study plan history
  - downloadable plan (`.md`)
  - spaced-repetition reminders

---

## Tech stack

- Python 3.x
- Streamlit
- google-generativeai
- streamlit-elements (Material UI style components)
- SQLite
- python-dotenv
- pandas
- pytest

---

## Project structure

- [app.py](app.py) – app entrypoint and top-level routing
- [database.py](database.py) – SQLite schema + data helper functions
- [services/quiz_engine.py](services/quiz_engine.py) – AI prompt, parsing, scoring, study-plan generation
- [pages/dashboard_page.py](pages/dashboard_page.py) – dashboard page
- [pages/quiz_page.py](pages/quiz_page.py) – quiz generation/attempt page
- [pages/history_page.py](pages/history_page.py) – quiz history + retake page
- [pages/study_coach_page.py](pages/study_coach_page.py) – personalized study planning page
- [components/modern_ui.py](components/modern_ui.py) – reusable visual components
- [components/session_state.py](components/session_state.py) – session-state helpers
- [assets/styles.css](assets/styles.css) – custom visual styling
- [.streamlit/config.toml](.streamlit/config.toml) – Streamlit theme/settings
- [test_app.py](test_app.py) – unit tests

---

## Setup (Linux bash)

### 1) Clone and open project

```bash
cd ai-learning-support-system
```

### 2) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5) Run the app

```bash
streamlit run app.py
```

---

## Running tests

```bash
pytest -q
```