import os
import unittest
from unittest.mock import Mock, patch

from search_providers import SearchProviderError, search_web


class SearchProviderTests(unittest.TestCase):
    def test_disabled_provider_returns_no_results(self):
        self.assertEqual(search_web("测试", {"provider": "disabled"}), [])

    def test_blank_query_is_rejected(self):
        with self.assertRaises(ValueError):
            search_web("  ", {"provider": "disabled"})

    @patch("search_providers.requests.post")
    def test_tavily_results_are_normalized(self, post):
        response = Mock()
        response.json.return_value = {
            "results": [{"title": "资料", "url": "https://example.com", "content": "摘要"}]
        }
        response.raise_for_status.return_value = None
        post.return_value = response

        results = search_web("主题", {"provider": "tavily", "api_key": "secret"})

        self.assertEqual(results[0].as_dict(), {
            "title": "资料",
            "url": "https://example.com",
            "content": "摘要",
        })

    def test_tavily_key_can_come_from_environment(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}), patch(
            "search_providers.requests.post"
        ) as post:
            response = Mock()
            response.json.return_value = {"results": []}
            response.raise_for_status.return_value = None
            post.return_value = response
            self.assertEqual(search_web("主题", {"provider": "tavily"}), [])

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(SearchProviderError):
            search_web("主题", {"provider": "unknown"})


if __name__ == "__main__":
    unittest.main()
