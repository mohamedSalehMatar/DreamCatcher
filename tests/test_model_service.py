import importlib
import unittest


class ParseModelOutputTests(unittest.TestCase):
    def test_parse_model_output_extracts_json_from_wrapped_response(self) -> None:
        module = importlib.import_module("src.model_service")

        raw_output = '''Sure — here is the dream entry:
        {"title": "Moon Train", "date": "06-08-2026", "description": "I dreamed of a red train crossing a moonlit ocean.", "symbols": ["train", "moon", "ocean"], "vibes": ["curious", "calm"]}
        '''

        parsed = module.parse_model_output(raw_output)

        self.assertEqual(parsed["title"], "Moon Train")
        self.assertEqual(parsed["description"], "I dreamed of a red train crossing a moonlit ocean.")
        self.assertEqual(parsed["symbols"], ["train", "moon", "ocean"])


if __name__ == "__main__":
    unittest.main()
