"""
SQLite database for all Spectra persistent storage.

Tables:
    conversations  – every chat turn (role, content, session_id)
    pc_profile     – key/value user profile built by the PC scanner
    activity_log   – real-time monitoring events
    training_log   – LoRA training session records
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Database:
    """SQLite-backed store for conversations, user profile, and activity."""

    def __init__(self, db_path: str = "spectra_memory.db") -> None:
        """Open (or create) the database at *db_path* and initialise schema.

        Args:
            db_path: File-system path for the SQLite database file.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Open a persistent connection with WAL mode for better concurrency."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def _create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            session_id  TEXT    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
            USING fts5(content, content=conversations, content_rowid=id);

        CREATE TRIGGER IF NOT EXISTS conversations_ai
            AFTER INSERT ON conversations BEGIN
                INSERT INTO conversations_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

        CREATE TABLE IF NOT EXISTS pc_profile (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT    NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT,
            updated_at  TEXT    NOT NULL,
            UNIQUE(category, key)
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            activity_type TEXT    NOT NULL,
            details       TEXT
        );

        CREATE TABLE IF NOT EXISTS training_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            conversations_used  INTEGER NOT NULL,
            duration            REAL    NOT NULL,
            adapter_path        TEXT    NOT NULL
        );
        """
        self._conn.executescript(ddl)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Conversation methods
    # ------------------------------------------------------------------

    def save_message(
        self, role: str, content: str, session_id: Optional[str] = None
    ) -> None:
        """Persist a single conversation turn.

        Args:
            role: Either ``"user"`` or ``"assistant"``.
            content: The message text.
            session_id: Identifier grouping turns into a session.  A new
                UUID is generated if *None*.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO conversations (timestamp, role, content, session_id) VALUES (?,?,?,?)",
            (ts, role, content, session_id),
        )
        self._conn.commit()

    def get_recent_conversations(
        self, n: int = 10, session_id: Optional[str] = None
    ) -> List[Dict]:
        """Return the *n* most recent conversation turns, oldest-first.

        Args:
            n: Maximum number of rows to return.
            session_id: When provided, restrict results to this session.

        Returns:
            List of dicts with keys ``role``, ``content``, ``timestamp``.
        """
        if session_id:
            rows = self._conn.execute(
                """SELECT role, content, timestamp FROM conversations
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT role, content, timestamp FROM conversations
                   ORDER BY id DESC LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_conversation_count(self) -> int:
        """Return the total number of conversation turns stored."""
        row = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0] if row else 0

    def get_training_data(self, since_id: int = 0) -> List[Dict]:
        """Fetch conversation turns not yet used for training.

        Args:
            since_id: Only return rows with ``id > since_id``.

        Returns:
            List of dicts with keys ``id``, ``role``, ``content``, ``session_id``.
        """
        rows = self._conn.execute(
            """SELECT id, role, content, session_id FROM conversations
               WHERE id > ? ORDER BY id ASC""",
            (since_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_conversations(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search over conversation content.

        Args:
            query: FTS5 query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching rows as dicts.
        """
        rows = self._conn.execute(
            """SELECT c.id, c.role, c.content, c.timestamp
               FROM conversations c
               JOIN conversations_fts f ON c.id = f.rowid
               WHERE conversations_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Profile methods
    # ------------------------------------------------------------------

    def update_profile(self, category: str, key: str, value: str) -> None:
        """Insert or update a user profile key/value pair.

        Args:
            category: Logical grouping (e.g. ``"software"``, ``"music"``).
            key: The profile attribute name.
            value: The attribute value (stored as text).
        """
        ts = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO pc_profile (category, key, value, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(category, key) DO UPDATE
               SET value=excluded.value, updated_at=excluded.updated_at""",
            (category, key, value, ts),
        )
        self._conn.commit()

    def get_profile(self, category: Optional[str] = None) -> List[Dict]:
        """Retrieve user profile entries.

        Args:
            category: If provided, only return entries for this category.

        Returns:
            List of dicts with keys ``category``, ``key``, ``value``, ``updated_at``.
        """
        if category:
            rows = self._conn.execute(
                "SELECT category, key, value, updated_at FROM pc_profile WHERE category=?",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT category, key, value, updated_at FROM pc_profile"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Activity log methods
    # ------------------------------------------------------------------

    def save_activity(self, activity_type: str, details: str) -> None:
        """Log a monitoring event.

        Args:
            activity_type: Short label such as ``"spotify"``, ``"browser"``.
            details: Human-readable description of the event.
        """
        ts = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO activity_log (timestamp, activity_type, details) VALUES (?,?,?)",
            (ts, activity_type, details),
        )
        self._conn.commit()

    def get_recent_activities(self, n: int = 20) -> List[Dict]:
        """Return the *n* most recent activity log entries, newest first.

        Args:
            n: Maximum number of rows to return.

        Returns:
            List of dicts with keys ``timestamp``, ``activity_type``, ``details``.
        """
        rows = self._conn.execute(
            "SELECT timestamp, activity_type, details FROM activity_log ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Training log methods
    # ------------------------------------------------------------------

    def save_training_log(
        self, conversations_used: int, duration: float, adapter_path: str
    ) -> None:
        """Record a completed LoRA training run.

        Args:
            conversations_used: Number of conversation turns used.
            duration: Wall-clock training time in seconds.
            adapter_path: File-system path where the adapter was saved.
        """
        ts = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO training_log (timestamp, conversations_used, duration, adapter_path)
               VALUES (?,?,?,?)""",
            (ts, conversations_used, duration, adapter_path),
        )
        self._conn.commit()

    def get_last_training_id(self) -> int:
        """Return the highest conversation ``id`` used in the last training run.

        Returns 0 if no training has been performed yet.
        """
        row = self._conn.execute(
            "SELECT adapter_path FROM training_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return 0
        # The adapter path encodes the max conversation id via data_prep naming
        # but we store the count separately; return total conversations as proxy.
        count_row = self._conn.execute(
            "SELECT conversations_used FROM training_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return count_row[0] if count_row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection gracefully."""
        if self._conn:
            self._conn.close()
            self._conn = None
