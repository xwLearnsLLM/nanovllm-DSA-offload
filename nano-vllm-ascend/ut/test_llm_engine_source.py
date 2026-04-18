from pathlib import Path
import unittest


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nanovllm"
    / "engine"
    / "llm_engine.py"
)
ENGINE_SOURCE = ENGINE_PATH.read_text(encoding="utf-8")


class TestLLMEngineSource(unittest.TestCase):
    def test_deepseek_tokenizer_enables_mistral_regex_fix(self):
        self.assertIn('kwargs["fix_mistral_regex"] = True', ENGINE_SOURCE)
        self.assertIn("fix_mistral_regex=True", ENGINE_SOURCE)

    def test_deepseek_string_prompt_uses_plain_encoding(self):
        self.assertIn("add_special_tokens=False", ENGINE_SOURCE)
        self.assertIn("def _encode_string_prompt", ENGINE_SOURCE)


if __name__ == "__main__":
    unittest.main()
