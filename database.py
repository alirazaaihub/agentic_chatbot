"""
database.py — Production-grade SQLite persistence
Stores: users, conversations, messages, memories
"""

import sqlite3
import uuid
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Optional

DB_PATH = "chatbot.db"

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                conv_id     TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                msg_id      TEXT PRIMARY KEY,
                conv_id     TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT NOT NULL,
                source      TEXT DEFAULT 'llm',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(conv_id)
            );

            CREATE TABLE IF NOT EXISTS memories (
                mem_id      TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                content     TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv  ON messages(conv_id);
            CREATE INDEX IF NOT EXISTS idx_messages_user  ON messages(user_id);
            CREATE INDEX IF NOT EXISTS idx_convs_user     ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user  ON memories(user_id);
        """)

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def ensure_user(user_id: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES(?,?)",
            (user_id, datetime.utcnow().isoformat())
        )

# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(user_id: str, title: str = "New Chat") -> str:
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations(conv_id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conv_id, user_id, title, now, now)
        )
    return conv_id

def get_user_conversations(user_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]

def update_conv_title(conv_id: str, title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE conv_id=?",
            (title, datetime.utcnow().isoformat(), conv_id)
        )

def touch_conversation(conv_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE conv_id=?",
            (datetime.utcnow().isoformat(), conv_id)
        )

def delete_conversation(conv_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages     WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE conv_id=?", (conv_id,))

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def save_message(conv_id: str, user_id: str, role: str,
                 content: str, source: str = "llm") -> str:
    msg_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages(msg_id,conv_id,user_id,role,content,source,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (msg_id, conv_id, user_id, role, content, source,
             datetime.utcnow().isoformat())
        )
    touch_conversation(conv_id)
    return msg_id

def get_conversation_messages(conv_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conv_id=? ORDER BY created_at ASC",
            (conv_id,)
        ).fetchall()
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------

def save_memory_db(user_id: str, content: str):
    """Idempotent — duplicate content silently ignored via UNIQUE constraint."""
    mem_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO memories(mem_id,user_id,content,created_at) VALUES(?,?,?,?)",
            (mem_id, user_id, content, datetime.utcnow().isoformat())
        )

def get_user_memories(user_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]

File name > database.py
