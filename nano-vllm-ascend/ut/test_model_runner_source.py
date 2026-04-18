from pathlib import Path
import unittest


MODEL_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "nanovllm"
    / "engine"
    / "model_runner.py"
)
MODEL_RUNNER_SOURCE = MODEL_RUNNER_PATH.read_text(encoding="utf-8")


class TestModelRunnerSource(unittest.TestCase):
    def test_execute_token_logging_is_opt_in(self):
        self.assertIn("NANOVLLM_LOG_EXECUTE_TOKENS", MODEL_RUNNER_SOURCE)
        self.assertIn("if _env_flag(", MODEL_RUNNER_SOURCE)


if __name__ == "__main__":
    unittest.main()
