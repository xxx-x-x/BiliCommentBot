import unittest
from unittest.mock import Mock

from distillation import DistillationError, distill_sources, format_distillation, parse_distillation


class DistillationTests(unittest.TestCase):
    def test_parses_fenced_json_and_validates_sources(self):
        result = parse_distillation(
            '''```json
            {"topic":"主题","summary":"摘要","facts":["事实"],"limitations":[],
             "conflicts":["来源说法不同"],"source_ids":[1],"confidence":0.8}
            ```''',
            {1, 2},
        )
        self.assertEqual(result["source_ids"], [1])
        self.assertIn("可信度：80%", format_distillation(result))

    def test_rejects_hallucinated_source(self):
        with self.assertRaises(DistillationError):
            parse_distillation({
                "topic": "主题", "summary": "摘要", "facts": ["事实"],
                "limitations": [], "conflicts": [], "source_ids": [99], "confidence": 0.5,
            }, {1})

    def test_calls_model_with_evidence(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": (
            '{"topic":"主题","summary":"摘要","facts":["事实"],"limitations":[], '
            '"conflicts":[],"source_ids":[3],"confidence":0.9}'
        )}}]}
        post = Mock(return_value=response)
        result = distill_sources(
            "问题", [{"id": 3, "title": "资料", "url": "https://example.com", "content": "证据"}],
            {"api_key": "secret", "base_url": "https://api.example.com", "model": "model"},
            post=post,
        )
        self.assertEqual(result["topic"], "主题")
        call = post.call_args
        self.assertTrue(call.kwargs["headers"]["Authorization"].startswith("Bearer "))
        self.assertIn("证据", call.kwargs["json"]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
