"""SQLite-backed, per-user conversation history."""

import json
import sqlite3
import time
import uuid

from api.auth import AUTH_DB_PATH


def initialize_chat_store() -> None:
    """Create conversation and message tables in the local auth store."""
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
                ON conversations(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                trace_json TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )


def create_conversation(user_id: int, title: str) -> dict:
    """Create a conversation for a user and return its public fields."""
    now = int(time.time())
    conversation_id = uuid.uuid4().hex
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, user_id, title, now, now),
        )
    return {"id": conversation_id, "title": title, "updated_at": now}


def list_conversations(user_id: int, limit: int = 100) -> list[dict]:
    """List a user's conversations, newest activity first."""
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation_with_messages(user_id: int, conversation_id: str) -> dict | None:
    """Return a conversation and its messages only when owned by the user."""
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        conversation = connection.execute(
            """
            SELECT id, title
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
        if conversation is None:
            return None
        messages = connection.execute(
            """
            SELECT role, content, trace_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "messages": [
            {
                "role": message["role"],
                "content": message["content"],
                "reasoning_trace": json.loads(message["trace_json"])
                if message["trace_json"]
                else [],
            }
            for message in messages
        ],
    }


def rename_conversation(user_id: int, conversation_id: str, title: str) -> dict | None:
    """Rename a user's conversation and return its public fields."""
    now = int(time.time())
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?,
                updated_at = CASE WHEN updated_at >= ? THEN updated_at + 1 ELSE ? END
            WHERE id = ? AND user_id = ?
            """,
            (title, now, now, conversation_id, user_id),
        )
        if cursor.rowcount == 0:
            return None
        row = connection.execute(
            "SELECT id, title, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
    return dict(zip(("id", "title", "updated_at"), row))


def delete_conversation(user_id: int, conversation_id: str) -> bool:
    """Delete a conversation only when it belongs to the user."""
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
    return cursor.rowcount > 0


def append_message(
    user_id: int,
    conversation_id: str,
    role: str,
    content: str,
    trace: list[dict] | None = None,
) -> int | None:
    """Append a message to an owned conversation and update its activity."""
    now = int(time.time())
    trace_json = json.dumps(trace, ensure_ascii=False) if trace is not None else None
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if exists is None:
            return None
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, trace_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, trace_json, now),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CASE WHEN updated_at >= ? THEN updated_at + 1 ELSE ? END
            WHERE id = ? AND user_id = ?
            """,
            (now, now, conversation_id, user_id),
        )
    return cursor.lastrowid
