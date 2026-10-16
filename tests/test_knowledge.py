import unittest

from knowledge import build_knowledge_context, chunk_text


class KnowledgeTests(unittest.TestCase):
    def test_chunk_text_preserves_all_content(self):
        text = "第一段。" * 80 + "\n\n" + "第二段。" * 80
        chunks = chunk_text(text, chunk_size=240, overlap=0)
        self.assertGreater(len(chunks), 2)
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))
        self.assertTrue(all(len(chunk) <= 240 for chunk in chunks))

    def test_blank_text_has_no_chunks(self):
        self.assertEqual(chunk_text(" \n "), [])

    def test_context_includes_source_and_respects_budget(self):
        context = build_knowledge_context([
            {
                "source_title": "官方资料",
                "source_url": "https://example.com",
                "content": "资料内容" * 300,
            }
        ], max_chars=600)
        self.assertIn("官方资料", context)
        self.assertIn("https://example.com", context)
        self.assertLessEqual(len(context), 600)


if __name__ == "__main__":
    unittest.main()
