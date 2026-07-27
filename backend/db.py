"""
db.py — SQLite storage for chat history.

Each message pair (question + answer) is saved so the conversation survives page
reloads. History is scoped by a per-browser `session_id`, so different visitors do not
see each other's chats.

Note: on ephemeral hosts (Hugging Face Spaces, Render free tier) the database file lives
on temporary disk and resets when the container restarts. That is fine for a demo; for
permanent history you would use a mounted disk or a managed database.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chats.db"


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    """Create the chats table if it does not exist. Called once at startup."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL
            )
        """)


def save_chat(session_id, question, answer):
    """Store one question/answer pair."""
    with _conn() as con:
        con.execute(
            "INSERT INTO chats (session_id, question, answer) VALUES (?, ?, ?)",
            (session_id, question, answer),
        )


def recent_chats(session_id, limit=100):
    """Return this session's messages, oldest first (so the UI can replay them)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT question, answer, created_at FROM chats "
            "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_chats(session_id):
    """Delete all messages for one session."""
    with _conn() as con:
        con.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
