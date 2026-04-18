import re
from pathlib import Path
import unittest

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nanovllm"
    / "models"
    / "deepseek_v32.py"
)
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
HELPER_NAMESPACE = {"torch": torch}


def _load_helper(func_name: str):
    pattern = re.compile(
        rf"^def {func_name}\(.*?(?=^def |^class |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(SOURCE)
    if match is None:
        raise RuntimeError(f"Unable to locate helper {func_name!r} in {MODULE_PATH}.")
    exec(match.group(0), HELPER_NAMESPACE)
    return HELPER_NAMESPACE[func_name]


_hadamard_transform = _load_helper("_hadamard_transform")
_rotate_activation = _load_helper("_rotate_activation")


class TestDeepseekV32Helpers(unittest.TestCase):
    def test_hadamard_transform_matches_reference(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        expected = torch.tensor([[10.0, -2.0, -4.0, 0.0]], dtype=torch.float32)
        actual = _hadamard_transform(x)
        self.assertTrue(torch.allclose(actual, expected))

    def test_rotate_activation_is_normalized(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        expected = torch.tensor([[5.0, -1.0, -2.0, 0.0]], dtype=torch.float32)
        actual = _rotate_activation(x)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual.norm(), x.norm(), atol=1e-6))

    def test_hadamard_requires_power_of_two(self):
        x = torch.ones(2, 3, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "power of 2"):
            _hadamard_transform(x)


if __name__ == "__main__":
    unittest.main()
