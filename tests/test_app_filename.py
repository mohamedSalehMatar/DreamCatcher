import unittest

from src.file_naming import build_output_filename


class FilenameGenerationTests(unittest.TestCase):
    def test_build_output_filename_uses_date_and_slugified_title(self) -> None:
        filename = build_output_filename("Huge Worm", "06-08-2026")

        self.assertEqual(filename, "06-08-2026_huge-worm")

    def test_build_output_filename_requires_title_to_be_present(self) -> None:
        with self.assertRaises(ValueError):
            build_output_filename("", "06-08-2026")


if __name__ == "__main__":
    unittest.main()
