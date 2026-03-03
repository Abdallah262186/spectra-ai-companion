"""SQLite-backed persistent memory for Spectra.

Tables
------
conversations : Every chat turn (user and assistant messages).
pc_profile    : Key-value user profile built by the PC scanner.
activity_log  : Timestamped monitoring events.
training_log  : Record of completed LoRA training sessions.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Database:
    """Manages all persistent storage via SQLite.

    Args:
        path: File-system path to the SQLite database file.
    """

    def __init__(self, path: str = "spectra_memory.db") -> None:
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the database and all required tables if they do not exist."""
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Database initialised at %s", self.path)

    def _create_tables(self) -> None:
        """Create schema tables."""
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                session_id  TEXT    NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(content, content=conversations, content_rowid=id);

            CREATE TABLE IF NOT EXISTS pc_profile (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT    NOT NULL,
                key         TEXT    NOT NULL,
                value       TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                UNIQUE(category, key)
            );

            CREATE TABLE IF NOT EXISTS activity_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                activity_type TEXT    NOT NULL,
                details       TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS training_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT    NOT NULL,
                conversations_used    INTEGER NOT NULL DEFAULT 0,
                last_conversation_id  INTEGER NOT NULL DEFAULT 0,
                duration              REAL    NOT NULL DEFAULT 0,
                adapter_path          TEXT    NOT NULL DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def _connection(self) -> sqlite3.Connection:
        """Return an active connection, reconnecting if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def save_message(self, role: str, content: str, session_id: str = "") -> None:
        """Persist a single conversation turn.

        Args:
            role: ``"user"`` or ``"assistant"``.
            content: The message text.
            session_id: Optional session identifier for grouping turns.
        """
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        conn = self._connection()
        cursor = conn.execute(
            "INSERT INTO conversations (timestamp, role, content, session_id) VALUES (?,?,?,?)",
            (ts, role, content, session_id),
        )
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO conversations_fts (rowid, content) VALUES (?,?)",
            (row_id, content),
        )
        conn.commit()

    def get_recent_conversations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the *n* most recent conversation rows (newest last).

        Args:
            n: Maximum number of rows to return.

        Returns:
            List of row dictionaries with keys: id, timestamp, role, content, session_id.
        """
        conn = self._connection()
        rows = conn.execute(
            "SELECT id, timestamp, role, content, session_id "
            "FROM conversations ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_training_data(self, since_id: int = 0) -> List[Dict[str, Any]]:
        """Return conversation rows not yet used for training.

        Args:
            since_id: Only return rows with ``id > since_id``.

        Returns:
            List of row dictionaries ordered by id ascending.
        """
        conn = self._connection()
        rows = conn.execute(
            "SELECT id, timestamp, role, content, session_id "
            "FROM conversations WHERE id > ? ORDER BY id ASC",
            (since_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search over conversation content.

        Args:
            query: FTS5 query string.
            limit: Maximum number of results.

        Returns:
            Matching row dictionaries.
        """
        conn = self._connection()
        rows = conn.execute(
            "SELECT c.id, c.timestamp, c.role, c.content, c.session_id "
            "FROM conversations c "
            "JOIN conversations_fts fts ON fts.rowid = c.id "
            "WHERE conversations_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # PC Profile
    # ------------------------------------------------------------------

    def update_profile(self, category: str, key: str, value: str) -> None:
        """Insert or replace a profile entry.

        Args:
            category: Broad grouping (e.g. ``"software"``, ``"music"``).
            key: Specific attribute name.
            value: String value.
        """
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        conn = self._connection()
        conn.execute(
            "INSERT INTO pc_profile (category, key, value, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(category, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (category, key, value, ts),
        )
        conn.commit()

    def get_profile(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return profile entries, optionally filtered by category.

        Args:
            category: If provided, only entries in this category are returned.

        Returns:
            List of row dictionaries.
        """
        conn = self._connection()
        if category:
            rows = conn.execute(
                "SELECT id, category, key, value, updated_at FROM pc_profile WHERE category=?",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category, key, value, updated_at FROM pc_profile"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    def save_activity(self, activity_type: str, details: str = "") -> None:
        """Log a monitoring event.

        Args:
            activity_type: Short label (e.g. ``"spotify"``, ``"download"``).
            details: Human-readable description of the event.
        """
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        conn = self._connection()
        conn.execute(
            "INSERT INTO activity_log (timestamp, activity_type, details) VALUES (?,?,?)",
            (ts, activity_type, details),
        )
        conn.commit()

    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent activity log entries (newest last).

        Args:
            limit: Maximum number of rows.

        Returns:
            List of row dictionaries.
        """
        conn = self._connection()
        rows = conn.execute(
            "SELECT id, timestamp, activity_type, details "
            "FROM activity_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Training log
    # ------------------------------------------------------------------

    def log_training(
        self,
        conversations_used: int,
        duration: float,
        adapter_path: str,
        last_conversation_id: int = 0,
    ) -> None:
        """Record a completed training session.

        Args:
            conversations_used: How many conversation rows were used.
            duration: Training duration in seconds.
            adapter_path: Where the LoRA adapter was saved.
            last_conversation_id: The highest conversation ``id`` included in
                this training run.  Used by the next run to fetch only new data.
        """
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        conn = self._connection()
        conn.execute(
            "INSERT INTO training_log "
            "(timestamp, conversations_used, last_conversation_id, duration, adapter_path) "
            "VALUES (?,?,?,?,?)",
            (ts, conversations_used, last_conversation_id, duration, adapter_path),
        )
        conn.commit()

    def get_last_trained_id(self) -> int:
        """Return the highest conversation id included in the last training run.

        Returns:
            The ``last_conversation_id`` value from the most recent training
            log entry, or 0 if no training has been run yet.
        """
        conn = self._connection()
        row = conn.execute(
            "SELECT last_conversation_id FROM training_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row and row["last_conversation_id"] is not None:
            return int(row["last_conversation_id"])
        return 0
