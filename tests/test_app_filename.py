import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from entry_service import format_entry_markdown, list_markdown_entries, normalize_entry_for_storage
from src.file_naming import build_output_filename


class FilenameGenerationTests(unittest.TestCase):
    def test_build_output_filename_uses_date_and_slugified_title(self) -> None:
        filename = build_output_filename("Huge Worm", "06-08-2026")

        self.assertEqual(filename, "06-08-2026_huge-worm")

    def test_build_output_filename_requires_title_to_be_present(self) -> None:
        with self.assertRaises(ValueError):
            build_output_filename("", "06-08-2026")


class MarkdownFormattingTests(unittest.TestCase):
    def test_format_entry_markdown_uses_generated_title_field(self) -> None:
        entry = {
            "dream_title": "Glass City",
            "dream_description": "I was flying through a city of glass.",
            "dream_symbols": ["glass", "flight"],
            "dream_vibes": ["wonder", "calm"],
        }

        markdown = format_entry_markdown(entry, "06-08-2026")

        self.assertIn("# Glass City", markdown)

    def test_normalize_entry_for_storage_uses_runtime_date(self) -> None:
        entry = {
            "dream_title": "Short Horse Transformation",
            "dream_date": "23-12-2023",
            "dream_description": "I transformed into a short horse.",
            "dream_symbols": ["horse"],
            "dream_vibes": ["surprise"],
        }

        normalized = normalize_entry_for_storage(entry, "06-08-2026")

        self.assertEqual(normalized["dream_date"], "06-08-2026")

    def test_list_markdown_entries_returns_sorted_markdown_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "b.md").write_text("second", encoding="utf-8")
            (temp_path / "a.md").write_text("first", encoding="utf-8")
            (temp_path / "notes.txt").write_text("ignore", encoding="utf-8")

            entries = list_markdown_entries(temp_path)

            self.assertEqual([entry.name for entry in entries], ["a.md", "b.md"])


if __name__ == "__main__":
    unittest.main()
