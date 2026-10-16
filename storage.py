"""SQLite persistence for the comment assistant MVP."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

DRAFT_STATUSES = frozenset({"pending", "approved", "rejected", "published", "failed"})
DISTILLATION_STATUSES = frozenset({"pending", "approved", "rejected"})
SOURCE_KINDS = frozenset({"manual", "web"})
_MAX_PAGE_SIZE = 100

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    bvid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('manual', 'web')),
    title TEXT NOT NULL,
    url TEXT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((video_id IS NOT NULL) <> (source_id IS NOT NULL)),
    UNIQUE (video_id, position),
    UNIQUE (source_id, position)
);

CREATE TABLE IF NOT EXISTS distillations (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    limitations_json TEXT NOT NULL DEFAULT '[]',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected')),
    approved_source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL UNIQUE,
    parent_external_id TEXT,
    root_external_id TEXT,
    author TEXT NOT NULL,
    content TEXT NOT NULL,
    commented_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reply_drafts (
    id INTEGER PRIMARY KEY,
    comment_id INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'published', 'failed')),
    error TEXT,
    risk_score INTEGER NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    details TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON knowledge_chunks(video_id, position);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON knowledge_chunks(source_id, position);
CREATE INDEX IF NOT EXISTS idx_distillations_status ON distillations(status, id DESC);
CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_drafts_status ON reply_drafts(status, id DESC);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(id DESC);
"""


def default_database_path() -> Path:
    """Return the database location, honoring the deployment data directory."""
    data_dir = os.environ.get("BILI_DATA_DIR")
    base = Path(data_dir).expanduser() if data_dir else Path(__file__).resolve().parent / "data"
    return base / "assistant.sqlite3"


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _page(page: int, page_size: int) -> tuple[int, int]:
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size must be a positive integer")
    size = min(page_size, _MAX_PAGE_SIZE)
    return size, (page - 1) * size


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


class Storage:
    """Small repository facade; each operation owns its SQLite connection."""

    def __init__(self, database: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(database).expanduser() if database is not None else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(_SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sources)")}
            if "video_id" not in columns:
                connection.execute(
                    "ALTER TABLE sources ADD COLUMN video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE"
                )
            comment_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(comments)")
            }
            if "root_external_id" not in comment_columns:
                connection.execute("ALTER TABLE comments ADD COLUMN root_external_id TEXT")
            draft_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(reply_drafts)")
            }
            if "risk_score" not in draft_columns:
                connection.execute(
                    "ALTER TABLE reply_drafts ADD COLUMN risk_score INTEGER NOT NULL DEFAULT 0"
                )

    def upsert_video(self, bvid: str, title: str, content: str = "") -> dict[str, Any]:
        bvid, title = _required(bvid, "bvid"), _required(title, "title")
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO videos (bvid, title, content) VALUES (?, ?, ?)
                   ON CONFLICT(bvid) DO UPDATE SET title=excluded.title,
                   content=excluded.content, updated_at=CURRENT_TIMESTAMP""",
                (bvid, title, content),
            )
            return dict(connection.execute("SELECT * FROM videos WHERE bvid = ?", (bvid,)).fetchone())

    def get_video(self, bvid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            return _row(connection.execute("SELECT * FROM videos WHERE bvid = ?", (bvid,)).fetchone())

    def list_videos(self, page: int = 1, page_size: int = 50) -> list[dict[str, Any]]:
        limit, offset = _page(page, page_size)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM videos ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [dict(item) for item in rows]

    def create_source(
        self,
        kind: str,
        title: str,
        content: str,
        url: str | None = None,
        *,
        video_id: int | None = None,
    ) -> dict[str, Any]:
        if kind not in SOURCE_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(SOURCE_KINDS))}")
        title = _required(title, "title")
        if kind == "web":
            url = _required(url or "", "url")
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO sources (video_id, kind, title, url, content) VALUES (?, ?, ?, ?, ?)",
                (video_id, kind, title, url, content),
            )
            return dict(connection.execute("SELECT * FROM sources WHERE id = ?", (cursor.lastrowid,)).fetchone())

    def delete_sources(self, source_ids: Sequence[int]) -> int:
        clean_ids = list(dict.fromkeys(int(item) for item in source_ids))
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._transaction() as connection:
            cursor = connection.execute(
                f"DELETE FROM sources WHERE id IN ({placeholders})", clean_ids
            )
            return int(cursor.rowcount)

    def replace_source(
        self,
        source_id: int,
        *,
        title: str,
        content: str,
        url: str | None = None,
        chunks: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        title = _required(title, "title")
        clean_chunks = self._clean_chunks(chunks) if chunks is not None else None
        with self._transaction() as connection:
            source = connection.execute("SELECT kind FROM sources WHERE id = ?", (source_id,)).fetchone()
            if source is None:
                raise KeyError(f"source {source_id} does not exist")
            if source["kind"] == "web":
                url = _required(url or "", "url")
            connection.execute(
                "UPDATE sources SET title=?, url=?, content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (title, url, content, source_id),
            )
            if clean_chunks is not None:
                self._replace_chunks(connection, "source_id", source_id, clean_chunks)
            return dict(connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            return _row(connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())

    def list_sources(
        self, *, video_id: int | None = None, page: int = 1, page_size: int = 50
    ) -> list[dict[str, Any]]:
        limit, offset = _page(page, page_size)
        where, params = ("WHERE video_id = ?", [video_id]) if video_id is not None else ("", [])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM sources {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [dict(item) for item in rows]

    @staticmethod
    def _clean_chunks(chunks: Iterable[str]) -> list[str]:
        result = []
        for chunk in chunks:
            if not isinstance(chunk, str) or not chunk.strip():
                raise ValueError("chunks must contain non-empty strings")
            result.append(chunk.strip())
        return result

    @staticmethod
    def _replace_chunks(
        connection: sqlite3.Connection, owner_column: str, owner_id: int, chunks: Sequence[str]
    ) -> None:
        connection.execute(f"DELETE FROM knowledge_chunks WHERE {owner_column} = ?", (owner_id,))
        connection.executemany(
            f"INSERT INTO knowledge_chunks ({owner_column}, position, content) VALUES (?, ?, ?)",
            ((owner_id, position, content) for position, content in enumerate(chunks)),
        )

    def replace_video_chunks(self, video_id: int, chunks: Iterable[str]) -> list[dict[str, Any]]:
        clean = self._clean_chunks(chunks)
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is None:
                raise KeyError(f"video {video_id} does not exist")
            self._replace_chunks(connection, "video_id", video_id, clean)
        return self.list_chunks(video_id=video_id)

    def replace_source_chunks(self, source_id: int, chunks: Iterable[str]) -> list[dict[str, Any]]:
        clean = self._clean_chunks(chunks)
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM sources WHERE id = ?", (source_id,)).fetchone() is None:
                raise KeyError(f"source {source_id} does not exist")
            self._replace_chunks(connection, "source_id", source_id, clean)
        return self.list_chunks(source_id=source_id)

    def list_chunks(
        self, *, video_id: int | None = None, source_id: int | None = None
    ) -> list[dict[str, Any]]:
        if (video_id is None) == (source_id is None):
            raise ValueError("provide exactly one of video_id or source_id")
        column, value = ("video_id", video_id) if video_id is not None else ("source_id", source_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM knowledge_chunks WHERE {column} = ? ORDER BY position", (value,)
            ).fetchall()
            return [dict(item) for item in rows]

    def search_chunks(
        self,
        query: str,
        *,
        video_id: int | None = None,
        source_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)))
        if not terms:
            return []
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, _MAX_PAGE_SIZE)
        patterns = [f"%{term.replace('%', '\\%').replace('_', '\\_')}%" for term in terms]
        score = " + ".join("CASE WHEN lower(k.content) LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END" for _ in terms)
        where = ["(" + " OR ".join("lower(k.content) LIKE ? ESCAPE '\\'" for _ in terms) + ")"]
        params: list[Any] = patterns + patterns
        if video_id is not None:
            where.append("(k.video_id = ? OR s.video_id = ?)")
            params.extend([video_id, video_id])
        if source_id is not None:
            where.append("k.source_id = ?")
            params.append(source_id)
        params.append(limit)
        sql = f"""SELECT k.*, s.title AS source_title, s.url AS source_url,
                  ({score}) AS score FROM knowledge_chunks k
                  LEFT JOIN sources s ON s.id = k.source_id
                  WHERE {' AND '.join(where)} ORDER BY score DESC, k.id ASC LIMIT ?"""
        with closing(self._connect()) as connection:
            return [dict(item) for item in connection.execute(sql, params).fetchall()]

    def create_distillation(
        self,
        video_id: int,
        query: str,
        *,
        topic: str,
        summary: str,
        facts: Sequence[str],
        limitations: Sequence[str],
        conflicts: Sequence[str],
        source_ids: Sequence[int],
        confidence: float,
    ) -> dict[str, Any]:
        query = _required(query, "query")
        topic = _required(topic, "topic")
        summary = _required(summary, "summary")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a number between 0 and 1")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be a number between 0 and 1")
        clean_facts = self._clean_string_list(facts, "facts", required=True)
        clean_limitations = self._clean_string_list(limitations, "limitations")
        clean_conflicts = self._clean_string_list(conflicts, "conflicts")
        clean_source_ids = list(dict.fromkeys(int(item) for item in source_ids))
        if not clean_source_ids:
            raise ValueError("source_ids must not be empty")
        with self._transaction() as connection:
            placeholders = ",".join("?" for _ in clean_source_ids)
            found = connection.execute(
                f"SELECT COUNT(*) FROM sources WHERE video_id=? AND id IN ({placeholders})",
                [video_id, *clean_source_ids],
            ).fetchone()[0]
            if found != len(clean_source_ids):
                raise ValueError("all sources must belong to the selected video")
            cursor = connection.execute(
                """INSERT INTO distillations
                   (video_id, query, topic, summary, facts_json, limitations_json,
                    conflicts_json, source_ids_json, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video_id, query, topic, summary,
                    json.dumps(clean_facts, ensure_ascii=False),
                    json.dumps(clean_limitations, ensure_ascii=False),
                    json.dumps(clean_conflicts, ensure_ascii=False),
                    json.dumps(clean_source_ids), confidence,
                ),
            )
            distillation_id = int(cursor.lastrowid)
            self._audit(connection, "distillation.created", "distillation", distillation_id, "pending")
            return self._distillation_row(
                connection.execute("SELECT * FROM distillations WHERE id=?", (distillation_id,)).fetchone()
            )

    @staticmethod
    def _clean_string_list(values: Sequence[str], name: str, *, required: bool = False) -> list[str]:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} must be a list")
        result = [str(item).strip() for item in values if str(item).strip()]
        if required and not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _distillation_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("facts", "limitations", "conflicts", "source_ids"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        return item

    def get_distillation(self, distillation_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT d.*, v.bvid, v.title AS video_title
                   FROM distillations d JOIN videos v ON v.id=d.video_id WHERE d.id=?""",
                (distillation_id,),
            ).fetchone()
            return self._distillation_row(row) if row else None

    def list_distillations(
        self,
        *,
        video_id: int | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 30,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in DISTILLATION_STATUSES:
            raise ValueError("invalid distillation status")
        limit, offset = _page(page, page_size)
        where, params = [], []
        if video_id is not None:
            where.append("d.video_id=?")
            params.append(video_id)
        if status is not None:
            where.append("d.status=?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT d.*, v.bvid, v.title AS video_title
                    FROM distillations d JOIN videos v ON v.id=d.video_id
                    {clause} ORDER BY d.id DESC LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
            return [self._distillation_row(row) for row in rows]

    def approve_distillation(
        self, distillation_id: int, content: str, chunks: Sequence[str]
    ) -> dict[str, Any]:
        content = _required(content, "content")
        clean_chunks = self._clean_chunks(chunks)
        if not clean_chunks:
            raise ValueError("chunks must not be empty")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM distillations WHERE id=?", (distillation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"distillation {distillation_id} does not exist")
            if row["status"] != "pending":
                raise ValueError("only pending distillations can be approved")
            cursor = connection.execute(
                """INSERT INTO sources (video_id, kind, title, content)
                   VALUES (?, 'manual', ?, ?)""",
                (row["video_id"], f"AI蒸馏：{row['topic']}", content),
            )
            source_id = int(cursor.lastrowid)
            self._replace_chunks(connection, "source_id", source_id, clean_chunks)
            connection.execute(
                """UPDATE distillations SET status='approved', approved_source_id=?,
                   updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (source_id, distillation_id),
            )
            self._audit(connection, "distillation.approved", "distillation", distillation_id, str(source_id))
            return self._distillation_row(
                connection.execute("SELECT * FROM distillations WHERE id=?", (distillation_id,)).fetchone()
            )

    def reject_distillation(self, distillation_id: int) -> dict[str, Any]:
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE distillations SET status='rejected', updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status='pending'""",
                (distillation_id,),
            )
            if cursor.rowcount == 0:
                if connection.execute("SELECT 1 FROM distillations WHERE id=?", (distillation_id,)).fetchone():
                    raise ValueError("only pending distillations can be rejected")
                raise KeyError(f"distillation {distillation_id} does not exist")
            self._audit(connection, "distillation.rejected", "distillation", distillation_id, "")
            return self._distillation_row(
                connection.execute("SELECT * FROM distillations WHERE id=?", (distillation_id,)).fetchone()
            )

    def add_comment(
        self,
        video_id: int,
        external_id: str,
        author: str,
        content: str,
        *,
        parent_external_id: str | None = None,
        root_external_id: str | None = None,
        commented_at: str | None = None,
    ) -> dict[str, Any]:
        external_id = _required(external_id, "external_id")
        author = _required(author, "author")
        content = _required(content, "content")
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO comments
                   (video_id, external_id, parent_external_id, root_external_id,
                    author, content, commented_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(external_id) DO UPDATE SET video_id=excluded.video_id,
                   parent_external_id=excluded.parent_external_id,
                   root_external_id=excluded.root_external_id, author=excluded.author,
                   content=excluded.content, commented_at=excluded.commented_at""",
                (
                    video_id, external_id, parent_external_id, root_external_id,
                    author, content, commented_at,
                ),
            )
            return dict(connection.execute("SELECT * FROM comments WHERE external_id=?", (external_id,)).fetchone())

    def list_comments(
        self, *, video_id: int | None = None, page: int = 1, page_size: int = 50
    ) -> list[dict[str, Any]]:
        limit, offset = _page(page, page_size)
        where, params = ("WHERE video_id = ?", [video_id]) if video_id is not None else ("", [])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM comments {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [dict(item) for item in rows]

    def get_draft_for_external_comment(self, external_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT d.* FROM reply_drafts d
                   JOIN comments c ON c.id = d.comment_id
                   WHERE c.external_id = ? ORDER BY d.id DESC LIMIT 1""",
                (external_id,),
            ).fetchone()
            return _row(row)

    def create_draft(
        self, comment_id: int, content: str, *, risk_score: int = 0
    ) -> dict[str, Any]:
        content = _required(content, "content")
        if isinstance(risk_score, bool) or not isinstance(risk_score, int) or not 0 <= risk_score <= 100:
            raise ValueError("risk_score must be an integer between 0 and 100")
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO reply_drafts (comment_id, content, risk_score) VALUES (?, ?, ?)",
                (comment_id, content, risk_score),
            )
            draft_id = int(cursor.lastrowid)
            self._audit(connection, "draft.created", "reply_draft", draft_id, "pending")
            return dict(connection.execute("SELECT * FROM reply_drafts WHERE id=?", (draft_id,)).fetchone())

    def list_drafts(
        self,
        *,
        status: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in DRAFT_STATUSES:
            raise ValueError("invalid draft status")
        limit, offset = _page(page, page_size)
        where, params = ("WHERE status = ?", [status]) if status else ("", [])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT d.*, c.external_id AS comment_external_id,
                    c.parent_external_id, c.root_external_id, c.author,
                    c.content AS comment_content, v.bvid, v.title AS video_title
                    FROM reply_drafts d
                    JOIN comments c ON c.id = d.comment_id
                    JOIN videos v ON v.id = c.video_id
                    {where.replace('status', 'd.status')}
                    ORDER BY d.id DESC LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
            return [dict(item) for item in rows]

    def get_draft(self, draft_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT d.*, c.external_id AS comment_external_id,
                   c.parent_external_id, c.root_external_id, c.author,
                   c.content AS comment_content, v.bvid, v.title AS video_title
                   FROM reply_drafts d
                   JOIN comments c ON c.id = d.comment_id
                   JOIN videos v ON v.id = c.video_id
                   WHERE d.id = ?""",
                (draft_id,),
            ).fetchone()
            return _row(row)

    def update_draft(self, draft_id: int, content: str) -> dict[str, Any]:
        content = _required(content, "content")
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE reply_drafts SET content=?, updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status IN ('pending', 'approved', 'failed')""",
                (content, draft_id),
            )
            if cursor.rowcount == 0:
                if connection.execute("SELECT 1 FROM reply_drafts WHERE id=?", (draft_id,)).fetchone():
                    raise ValueError("published or rejected drafts cannot be edited")
                raise KeyError(f"draft {draft_id} does not exist")
            self._audit(connection, "draft.updated", "reply_draft", draft_id, "")
            return dict(connection.execute("SELECT * FROM reply_drafts WHERE id=?", (draft_id,)).fetchone())

    def transition_draft(
        self, draft_id: int, status: str, *, error: str | None = None
    ) -> dict[str, Any]:
        if status not in DRAFT_STATUSES:
            raise ValueError("invalid draft status")
        allowed = {
            "pending": {"approved", "rejected"},
            "approved": {"rejected", "published", "failed"},
            "failed": {"approved", "rejected", "published"},
            "rejected": set(),
            "published": set(),
        }
        with self._transaction() as connection:
            current = connection.execute("SELECT status FROM reply_drafts WHERE id=?", (draft_id,)).fetchone()
            if current is None:
                raise KeyError(f"draft {draft_id} does not exist")
            old_status = current["status"]
            if status != old_status and status not in allowed[old_status]:
                raise ValueError(f"cannot transition draft from {old_status} to {status}")
            connection.execute(
                "UPDATE reply_drafts SET status=?, error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (status, error if status == "failed" else None, draft_id),
            )
            self._audit(connection, "draft.status_changed", "reply_draft", draft_id, f"{old_status}->{status}")
            return dict(connection.execute("SELECT * FROM reply_drafts WHERE id=?", (draft_id,)).fetchone())

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        event_type: str,
        entity_type: str,
        entity_id: int | str | None,
        details: str,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO audit_events (event_type, entity_type, entity_id, details) VALUES (?, ?, ?, ?)",
            (event_type, entity_type, None if entity_id is None else str(entity_id), details),
        )
        return int(cursor.lastrowid)

    def record_audit_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: int | str | None = None,
        details: str = "",
    ) -> dict[str, Any]:
        event_type = _required(event_type, "event_type")
        entity_type = _required(entity_type, "entity_type")
        with self._transaction() as connection:
            event_id = self._audit(connection, event_type, entity_type, entity_id, details)
            return dict(connection.execute("SELECT * FROM audit_events WHERE id=?", (event_id,)).fetchone())

    def list_audit_events(
        self, *, page: int = 1, page_size: int = 50
    ) -> list[dict[str, Any]]:
        limit, offset = _page(page, page_size)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [dict(item) for item in rows]

    def summary_counts(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "videos", "sources", "knowledge_chunks", "distillations",
                    "comments", "reply_drafts", "audit_events",
                )
            }
            counts["distillations_by_status"] = {
                status: 0 for status in sorted(DISTILLATION_STATUSES)
            }
            counts["distillations_by_status"].update(
                dict(connection.execute(
                    "SELECT status, COUNT(*) FROM distillations GROUP BY status"
                ).fetchall())
            )
            counts["drafts_by_status"] = {
                status: 0 for status in sorted(DRAFT_STATUSES)
            }
            counts["drafts_by_status"].update(
                dict(connection.execute("SELECT status, COUNT(*) FROM reply_drafts GROUP BY status").fetchall())
            )
            return counts

