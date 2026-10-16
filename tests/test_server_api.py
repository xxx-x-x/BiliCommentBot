import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import server
from bot import BiliCommentBot, DEFAULT_CONFIG
from storage import Storage


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp_dir.cleanup)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["ai"]["api_key"] = "secret"
        self.bot = BiliCommentBot(config, logging.getLogger("api-test"))
        self.bot.storage = Storage(Path(self.temp_dir.name) / "api.sqlite3")
        self.old_bot = server._bot
        server._bot = self.bot
        self.addCleanup(setattr, server, "_bot", self.old_bot)
        server.app.config["TESTING"] = True
        self.client = server.app.test_client()

    def test_load_config_migrates_legacy_deepseek_section(self):
        config_path = Path(self.temp_dir.name) / "legacy.toml"
        config_path.write_text(
            '[deepseek]\napi_key = "legacy-key"\nbase_url = "https://legacy.example/v1"\nmodel = "gpt-legacy"\n',
            encoding="utf-8",
        )
        with patch.object(server, "CONFIG_FILE", str(config_path)):
            config = server.load_config()
        self.assertEqual(config["ai"]["api_key"], "legacy-key")
        self.assertEqual(config["ai"]["model"], "gpt-legacy")

    def test_video_document_and_manual_source_are_chunked(self):
        response = self.client.post("/api/knowledge/videos", json={
            "bvid": "BV1api", "title": "视频", "content": "视频文案。" * 200
        })
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.get_json()["chunks"], 1)

        response = self.client.post("/api/knowledge/sources", json={
            "bvid": "BV1api", "title": "补充资料", "content": "资料内容。" * 100
        })
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.get_json()["chunks"], 0)

        response = self.client.post("/api/knowledge/sources", json={
            "bvid": "BV1api", "content": "无标题资料"
        })
        self.assertEqual(response.status_code, 200)

    @patch("server.load_config")
    @patch("server.distill_sources")
    @patch("server.search_web")
    def test_search_results_wait_for_distillation_approval(self, search, distill, load_config):
        from search_providers import SearchResult

        load_config.return_value = copy.deepcopy(DEFAULT_CONFIG)
        self.bot.storage.upsert_video("BV1api", "视频")
        search.return_value = [SearchResult("资料", "https://example.com", "可检索内容")]
        distill.return_value = {
            "topic": "主题", "summary": "摘要", "facts": ["可核验事实"],
            "limitations": [], "conflicts": [], "source_ids": [1], "confidence": 0.8,
        }
        response = self.client.post("/api/knowledge/search", json={
            "bvid": "BV1api", "query": "主题"
        })
        self.assertEqual(response.status_code, 200)
        record = response.get_json()["distillation"]
        self.assertEqual(record["status"], "pending")
        video = self.bot.storage.get_video("BV1api")
        self.assertEqual(self.bot.storage.search_chunks("可核验事实", video_id=video["id"]), [])
        response = self.client.post(
            f"/api/knowledge/distillations/{record['id']}/approve", json={}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            len(self.bot.storage.search_chunks("可核验事实", video_id=video["id"])), 1
        )

    @patch("server.load_config")
    @patch("server.distill_sources", side_effect=Exception("unexpected"))
    @patch("server.search_web")
    def test_failed_distillation_does_not_leave_sources(self, search, distill, load_config):
        from search_providers import SearchResult
        from distillation import DistillationError

        distill.side_effect = DistillationError("模型格式错误")
        load_config.return_value = copy.deepcopy(DEFAULT_CONFIG)
        self.bot.storage.upsert_video("BV1rollback", "视频")
        search.return_value = [SearchResult("资料", "https://example.com", "内容")]
        response = self.client.post("/api/knowledge/search", json={
            "bvid": "BV1rollback", "query": "主题"
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.bot.storage.list_sources(), [])

    def test_pending_draft_can_be_approved_then_published(self):
        video = self.bot.storage.upsert_video("BV1api", "视频")
        comment = self.bot.storage.add_comment(video["id"], "c-api", "Alice", "问题")
        draft = self.bot.storage.create_draft(comment["id"], "初稿")

        response = self.client.post(
            f"/api/drafts/{draft['id']}/approve", json={"content": "编辑后的回复"}
        )
        self.assertEqual(response.status_code, 200)
        self.bot.reply_comment = Mock(return_value=True)
        response = self.client.post(f"/api/drafts/{draft['id']}/publish", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["draft"]["status"], "published")


if __name__ == "__main__":
    unittest.main()
