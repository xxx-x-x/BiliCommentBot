import os
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from storage import Storage, default_database_path


class StorageTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp_dir.cleanup)
        self.storage = Storage(Path(self.temp_dir.name) / "test.sqlite3")
        self.video = self.storage.upsert_video("BV1test", "Test video", "Transcript")

    def test_default_path_honors_data_dir(self):
        expected = Path(self.temp_dir.name) / "assistant.sqlite3"
        with patch.dict(os.environ, {"BILI_DATA_DIR": self.temp_dir.name}):
            self.assertEqual(default_database_path(), expected)
            storage = Storage()
            self.assertTrue(storage.path.exists())

    def test_schema_constraints_and_video_upsert(self):
        updated = self.storage.upsert_video("BV1test", "New title", "New transcript")
        self.assertEqual(updated["id"], self.video["id"])
        self.assertEqual(self.storage.get_video("BV1test")["title"], "New title")
        with closing(sqlite3.connect(self.storage.path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO knowledge_chunks (video_id, source_id, position, content) VALUES (?, ?, 0, 'bad')",
                    (self.video["id"], 999),
                )

    def test_sources_and_atomic_chunk_replacement(self):
        manual = self.storage.create_source("manual", "Notes", "Body")
        web = self.storage.create_source(
            "web", "Docs", "Web body", "https://example.test", video_id=self.video["id"]
        )
        chunks = self.storage.replace_source_chunks(manual["id"], ["alpha one", "beta two"])
        self.assertEqual([item["position"] for item in chunks], [0, 1])

        replaced = self.storage.replace_source(
            manual["id"], title="Updated", content="Fresh", chunks=["gamma three"]
        )
        self.assertEqual(replaced["title"], "Updated")
        self.assertEqual([item["content"] for item in self.storage.list_chunks(source_id=manual["id"])], ["gamma three"])
        self.assertEqual(web["url"], "https://example.test")
        self.assertEqual(self.storage.list_sources(video_id=self.video["id"])[0]["id"], web["id"])

        with self.assertRaises(ValueError):
            self.storage.replace_source_chunks(manual["id"], ["valid", ""])
        self.assertEqual([item["content"] for item in self.storage.list_chunks(source_id=manual["id"])], ["gamma three"])
        with self.assertRaises(ValueError):
            self.storage.create_source("web", "Missing URL", "body")

    def test_lexical_retrieval_filters_and_scores(self):
        other = self.storage.upsert_video("BV2test", "Other")
        self.storage.replace_video_chunks(
            self.video["id"], ["sqlite durable storage", "sqlite only", "irrelevant"]
        )
        self.storage.replace_video_chunks(other["id"], ["sqlite durable elsewhere"])

        results = self.storage.search_chunks("sqlite durable", video_id=self.video["id"])
        self.assertEqual([item["score"] for item in results], [2, 1])
        self.assertTrue(all(item["video_id"] == self.video["id"] for item in results))
        self.assertEqual(self.storage.search_chunks("%%%"), [])

    def test_distillation_requires_approval_before_chunks_exist(self):
        source = self.storage.create_source(
            "web", "Docs", "Raw evidence", "https://example.test", video_id=self.video["id"]
        )
        distillation = self.storage.create_distillation(
            self.video["id"],
            "sqlite",
            topic="SQLite durability",
            summary="SQLite is durable when configured correctly.",
            facts=["Transactions are atomic"],
            limitations=["Configuration matters"],
            conflicts=[],
            source_ids=[source["id"]],
            confidence=0.9,
        )
        self.assertEqual(distillation["status"], "pending")
        self.assertEqual(self.storage.search_chunks("atomic", video_id=self.video["id"]), [])
        approved = self.storage.approve_distillation(
            distillation["id"], "主题：SQLite\n事实：事务具有原子性", ["事务具有原子性"]
        )
        self.assertEqual(approved["status"], "approved")
        self.assertIsNotNone(approved["approved_source_id"])
        self.assertEqual(len(self.storage.search_chunks("原子性", video_id=self.video["id"])), 1)
        with self.assertRaises(ValueError):
            self.storage.reject_distillation(distillation["id"])

    def test_comments_drafts_transitions_and_audit(self):
        comment = self.storage.add_comment(
            self.video["id"], "comment-1", "Alice", "Question", commented_at="2026-01-01T00:00:00Z"
        )
        changed = self.storage.add_comment(self.video["id"], "comment-1", "Alice", "Edited")
        self.assertEqual(changed["id"], comment["id"])

        draft = self.storage.create_draft(comment["id"], "Initial answer", risk_score=15)
        detail = self.storage.get_draft(draft["id"])
        self.assertEqual(detail["bvid"], "BV1test")
        self.assertEqual(detail["risk_score"], 15)
        draft = self.storage.update_draft(draft["id"], "Better answer")
        self.assertEqual(draft["content"], "Better answer")
        approved = self.storage.transition_draft(draft["id"], "approved")
        published = self.storage.transition_draft(approved["id"], "published")
        self.assertEqual(published["status"], "published")
        with self.assertRaises(ValueError):
            self.storage.update_draft(draft["id"], "Too late")
        with self.assertRaises(ValueError):
            self.storage.transition_draft(draft["id"], "pending")

        second = self.storage.create_draft(comment["id"], "Retry me")
        self.storage.transition_draft(second["id"], "approved")
        failed = self.storage.transition_draft(second["id"], "failed", error="network")
        self.assertEqual(failed["error"], "network")
        self.assertEqual(len(self.storage.list_drafts(status="failed")), 1)
        self.assertGreaterEqual(len(self.storage.list_audit_events()), 7)

    def test_pagination_is_validated_and_capped(self):
        for index in range(105):
            self.storage.upsert_video(f"BV{index:04d}", f"Video {index}")
        self.assertEqual(len(self.storage.list_videos(page_size=1_000)), 100)
        self.assertEqual(len(self.storage.list_videos(page=2, page_size=100)), 6)
        for kwargs in ({"page": 0}, {"page_size": 0}, {"page": True}):
            with self.assertRaises(ValueError):
                self.storage.list_videos(**kwargs)

    def test_summary_counts(self):
        source = self.storage.create_source("manual", "Notes", "Body")
        self.storage.replace_source_chunks(source["id"], ["one", "two"])
        comment = self.storage.add_comment(self.video["id"], "c-summary", "Bob", "Hello")
        self.storage.create_draft(comment["id"], "Hi")
        counts = self.storage.summary_counts()
        self.assertEqual(counts["videos"], 1)
        self.assertEqual(counts["sources"], 1)
        self.assertEqual(counts["knowledge_chunks"], 2)
        self.assertEqual(counts["comments"], 1)
        self.assertEqual(counts["drafts_by_status"]["pending"], 1)

    def test_connections_are_safe_across_threads(self):
        errors = []

        def worker(index):
            try:
                self.storage.add_comment(
                    self.video["id"], f"thread-{index}", f"User {index}", f"Comment {index}"
                )
            except Exception as exc:  # pragma: no cover - assertion reports unexpected failures
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.storage.list_comments(page_size=20)), 8)


if __name__ == "__main__":
    unittest.main()
