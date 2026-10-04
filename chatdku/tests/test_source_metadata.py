"""Small, service-free checks for the retrieval boundary metadata contract."""

import unittest

from chatdku.core.tools.source_metadata import normalize_source_metadata


class SourceMetadataTests(unittest.TestCase):
    def test_keyword_result_gains_common_file_name(self):
        original = {
            "filename": "handbook.pdf",
            "url": "https://example.edu/handbook",
            "page_number": 7,
        }
        normalized = normalize_source_metadata(original)

        self.assertEqual(normalized["file_name"], "handbook.pdf")
        self.assertEqual(normalized["filename"], "handbook.pdf")
        self.assertEqual(normalized["url"], original["url"])
        self.assertEqual(normalized["page_number"], 7)
        self.assertNotIn("file_name", original)

    def test_vector_result_keeps_canonical_name(self):
        original = {"file_name": "catalog.pdf", "url": "no url", "page_number": 2}
        self.assertEqual(normalize_source_metadata(original), original)

    def test_canonical_name_wins_if_both_are_present(self):
        original = {"file_name": "canonical.pdf", "filename": "legacy.pdf"}
        self.assertEqual(normalize_source_metadata(original)["file_name"], "canonical.pdf")

    def test_missing_name_does_not_invent_one(self):
        self.assertEqual(
            normalize_source_metadata({"url": "no url"}), {"url": "no url"}
        )


if __name__ == "__main__":
    unittest.main()
