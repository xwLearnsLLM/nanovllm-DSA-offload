from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    max_steps: Optional[int] = None
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative"
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer or None")
