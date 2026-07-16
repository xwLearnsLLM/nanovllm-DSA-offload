from __future__ import annotations

import re
from collections.abc import Mapping
from operator import index
from typing import Any


_MIN_GLM_TOKENIZER_VERSION = (5, 5)
_PINNED_GLM_TOKENIZER_VERSION = "5.5.3"


def normalize_token_ids(value: Any) -> list[int]:
    """Convert tokenizer outputs for one prompt to an owned list of IDs."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Tokenizer output does not contain input_ids.")
        value = value["input_ids"]

    encoding_ids = getattr(value, "ids", None)
    if encoding_ids is not None:
        value = encoding_ids

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "Prompt token IDs must be one unbatched list, BatchEncoding, "
            "or Encoding; "
            f"got {type(value).__name__}."
        )

    normalized = []
    for token_id in value:
        try:
            normalized.append(index(token_id))
        except TypeError as error:
            raise TypeError(
                "Prompt token IDs must contain integers; "
                f"got {type(token_id).__name__}."
            ) from error
    if not normalized:
        raise ValueError("Prompt must contain at least one token.")
    return normalized


def require_glm_tokenizer_version(version: str) -> None:
    """Reject Transformers releases that cannot read GLM's v5 tokenizer."""

    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match is None:
        raise RuntimeError(
            f"Cannot parse installed Transformers version {version!r}."
        )
    major_minor = tuple(int(value) for value in match.groups())
    if major_minor < _MIN_GLM_TOKENIZER_VERSION:
        raise RuntimeError(
            "GLM-5.1 uses the Transformers v5 TokenizersBackend and the "
            "list-form extra_special_tokens metadata, which requires "
            "Transformers >= 5.5 in this project. Installed version: "
            f"{version}. Run: python3 -m pip install --upgrade "
            f"'transformers=={_PINNED_GLM_TOKENIZER_VERSION}'"
        )


def load_glm_tokenizer(model_path: str, *, trust_remote_code: bool):
    import transformers
    from transformers import AutoTokenizer

    require_glm_tokenizer_version(transformers.__version__)
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

