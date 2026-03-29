"""SQLite utility helpers for the AI Learning Support System.

This module is intentionally modular so functions can be unit-tested independently.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "learning_support.db"


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize required database tables if they do not exist."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quizzes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    source_text TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    quiz_id INTEGER NOT NULL,
                    score INTEGER NOT NULL,
                    total_questions INTEGER NOT NULL,
                    selected_answers_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
                )
                """
            )


def get_or_create_user(username: str) -> int:
    """Return user ID for username, creating the user if needed."""
    cleaned = username.strip().lower()
    if not cleaned:
        raise ValueError("Username cannot be empty.")

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (username) VALUES (?)",
                (cleaned,),
            )
            row = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (cleaned,),
            ).fetchone()

    if row is None:
        raise RuntimeError("Could not create or fetch user.")
    return int(row["id"])


def insert_quiz(user_id: int, source_text: str, questions: list[dict[str, Any]]) -> int:
    """Insert a generated quiz and return its ID."""
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO quizzes (user_id, source_text, questions_json)
                VALUES (?, ?, ?)
                """,
                (user_id, source_text, json.dumps(questions)),
            )
            return int(cursor.lastrowid)


def insert_result(
    user_id: int,
    quiz_id: int,
    score: int,
    total_questions: int,
    selected_answers: list[str | None],
) -> int:
    """Insert a quiz result and return the result ID."""
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO results (user_id, quiz_id, score, total_questions, selected_answers_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, quiz_id, score, total_questions, json.dumps(selected_answers)),
            )
            return int(cursor.lastrowid)


def save_quiz_attempt(
    user_id: int,
    source_text: str,
    questions: list[dict[str, Any]],
    score: int,
    total_questions: int,
    selected_answers: list[str | None],
) -> tuple[int, int]:
    """Persist both quiz and result rows for a completed attempt."""
    quiz_id = insert_quiz(user_id=user_id, source_text=source_text, questions=questions)
    result_id = insert_result(
        user_id=user_id,
        quiz_id=quiz_id,
        score=score,
        total_questions=total_questions,
        selected_answers=selected_answers,
    )
    return quiz_id, result_id


def fetch_user_metrics(user_id: int) -> dict[str, float]:
    """Return aggregate metrics for dashboard cards."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_quizzes_taken,
                COALESCE(AVG(CASE WHEN total_questions > 0 THEN (score * 100.0) / total_questions END), 0) AS average_score_percent,
                COALESCE(SUM(total_questions), 0) AS total_questions_answered
            FROM results
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    return {
        "total_quizzes_taken": float(row["total_quizzes_taken"]),
        "average_score_percent": float(row["average_score_percent"]),
        "total_questions_answered": float(row["total_questions_answered"]),
    }


def fetch_score_history(user_id: int) -> list[dict[str, Any]]:
    """Return attempt-by-attempt score history for charting."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT
                created_at AS attempt_date,
                score,
                total_questions,
                CASE
                    WHEN total_questions > 0 THEN (score * 100.0) / total_questions
                    ELSE 0
                END AS score_percent
            FROM results
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_quiz_history(user_id: int) -> list[dict[str, Any]]:
    """Return recent quiz history with summary fields for list/table UI."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT
                q.id AS quiz_id,
                q.source_text,
                q.created_at,
                r.score,
                r.total_questions,
                CASE
                    WHEN r.total_questions > 0 THEN (r.score * 100.0) / r.total_questions
                    ELSE 0
                END AS score_percent
            FROM quizzes q
            JOIN results r ON r.quiz_id = q.id
            WHERE q.user_id = ?
            ORDER BY q.created_at DESC
            """,
            (user_id,),
        ).fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        source = row["source_text"] or ""
        preview = source.strip().replace("\n", " ")[:100]
        history.append(
            {
                "quiz_id": int(row["quiz_id"]),
                "topic_preview": preview + ("..." if len(source.strip()) > 100 else ""),
                "created_at": row["created_at"],
                "score": int(row["score"]),
                "total_questions": int(row["total_questions"]),
                "score_percent": float(row["score_percent"]),
            }
        )
    return history


def fetch_quiz_by_id(user_id: int, quiz_id: int) -> dict[str, Any] | None:
    """Fetch one quiz and parse stored questions JSON for retake."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT id, source_text, questions_json
            FROM quizzes
            WHERE id = ? AND user_id = ?
            """,
            (quiz_id, user_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "quiz_id": int(row["id"]),
        "source_text": row["source_text"],
        "questions": json.loads(row["questions_json"]),
    }
