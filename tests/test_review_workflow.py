import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bot import BiliCommentBot, Comment, DEFAULT_CONFIG
from storage import Storage


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp_dir.cleanup)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["ai"]["api_key"] = "secret"
        self.config["reply"]["only_bvid"] = "BV1test"
        self.config["reply"]["max_process"] = 1
        self.config["reply"]["reply_delay"] = 0
        self.bot = BiliCommentBot(self.config, logging.getLogger("review-test"))
        self.bot.storage = Storage(Path(self.temp_dir.name) / "workflow.sqlite3")

    def test_generate_reply_sends_bearer_key_and_knowledge(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "回答"}}]}
        self.bot.session.post = Mock(return_value=response)

        reply = self.bot.generate_reply("问题", knowledge_context="[1] 资料\n可靠内容")

        self.assertEqual(reply, "回答")
        call = self.bot.session.post.call_args
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer secret")
        messages = call.kwargs["json"]["messages"]
        self.assertIn("可靠内容", messages[1]["content"])

    def test_manual_strategy_creates_pending_draft_without_publishing(self):
        self.config["publishing"]["strategy"] = "manual"
        self.bot.get_video_comments = Mock(return_value=[
            Comment("c1", "这个内容是什么？", "Alice", "u1", 1)
        ])
        self.bot.generate_reply = Mock(return_value="这是审核前草稿")
        self.bot.reply_comment = Mock(return_value=True)

        self.bot.process_comments()

        drafts = self.bot.storage.list_drafts(status="pending")
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["comment_external_id"], "c1")
        self.bot.reply_comment.assert_not_called()

    def test_automatic_strategy_publishes_the_draft(self):
        self.config["publishing"]["strategy"] = "automatic"
        self.bot.get_video_comments = Mock(return_value=[
            Comment("c2", "普通问题", "Bob", "u2", 1)
        ])
        self.bot.generate_reply = Mock(return_value="自动回复")
        self.bot.reply_comment = Mock(return_value=True)

        self.bot.process_comments()

        drafts = self.bot.storage.list_drafts(status="published")
        self.assertEqual(len(drafts), 1)
        self.bot.reply_comment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
