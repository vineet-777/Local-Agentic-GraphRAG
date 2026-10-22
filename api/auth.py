"""Local, SQLite-backed user authentication and bearer sessions."""

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import DB_DIR

AUTH_DB_PATH = DB_DIR / "auth.sqlite3"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
_password_hasher = PasswordHasher()
_bearer_scheme = HTTPBearer(auto_error=False)


class UserAlreadyExistsError(Exception):
    """Raised when a signup email is already registered."""


def initialize_auth_store() -> None:
    """Create the local user and session tables if they do not exist."""
    Path(AUTH_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
            """
        )


def _public_user(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "created_at": row["created_at"],
    }


def create_user(email: str, name: str, password: str) -> dict:
    normalized_email = email.strip().lower()
    password_hash = _password_hasher.hash(password)
    try:
        with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                (normalized_email, name.strip(), password_hash),
            )
            row = connection.execute(
                "SELECT id, email, name, created_at FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return _public_user(row)
    except sqlite3.IntegrityError as error:
        raise UserAlreadyExistsError from error


def authenticate_user(email: str, password: str) -> dict | None:
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip(),),
        ).fetchone()

    if row is None:
        return None
    try:
        if not _password_hasher.verify(row["password_hash"], password):
            return None
    except (VerifyMismatchError, InvalidHashError):
        return None

    return _public_user(row)


def create_session(user_id: int) -> tuple[str, int]:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    expires_at = now + SESSION_TTL_SECONDS
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token_hash, user_id, expires_at, now),
        )
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    return token, SESSION_TTL_SECONDS


def get_user_for_token(token: str) -> dict | None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT users.id, users.email, users.name, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """,
            (token_hash, int(time.time())),
        ).fetchone()
    return _public_user(row) if row else None


def revoke_session(token: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with sqlite3.connect(AUTH_DB_PATH, timeout=10) as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Sign in is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_for_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Session is invalid or expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user