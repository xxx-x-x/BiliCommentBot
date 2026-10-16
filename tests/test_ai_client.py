import unittest
from unittest.mock import Mock

from ai_client import chat_completions_url, request_chat_completion, resolve_ai_config


class AIClientTests(unittest.TestCase):
    def test_builds_url_from_base_or_full_endpoint(self):
        self.assertEqual(
            chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )
        endpoint = "https://gateway.example.com/openai/chat/completions"
        self.assertEqual(chat_completions_url(endpoint), endpoint)

    def test_legacy_config_is_supported(self):
        self.assertEqual(
            resolve_ai_config({"deepseek": {"api_key": "legacy"}})["api_key"], "legacy"
        )
        self.assertEqual(
            resolve_ai_config({"ai": {"api_key": "new"}, "deepseek": {"api_key": "old"}})["api_key"],
            "new",
        )

    def test_json_mode_is_optional(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        session = Mock()
        session.post.return_value = response
        base = {
            "api_key": "key", "base_url": "https://api.example.com/v1",
            "model": "gpt-test", "max_tokens": 100, "temperature": 0.2,
        }
        self.assertEqual(
            request_chat_completion(base, [{"role": "user", "content": "hi"}], json_mode=True, session=session),
            "ok",
        )
        self.assertNotIn("response_format", session.post.call_args.kwargs["json"])
        base["json_mode"] = True
        request_chat_completion(base, [{"role": "user", "content": "hi"}], json_mode=True, session=session)
        self.assertEqual(
            session.post.call_args.kwargs["json"]["response_format"], {"type": "json_object"}
        )


if __name__ == "__main__":
    unittest.main()
