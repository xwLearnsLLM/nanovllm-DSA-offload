from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config
from nanovllm.engine.dsa_offload import (
    build_dsa_debug_selection,
    parse_dsa_debug_selection,
    validate_dsa_debug_selection,
)


def test_dsa_debug_selection_defaults_to_native():
    assert parse_dsa_debug_selection(None) == "native"
    assert parse_dsa_debug_selection("  ") == "native"
    assert build_dsa_debug_selection(
        torch.tensor([2176], dtype=torch.int32),
        "native",
    ) is None


@pytest.mark.parametrize(
    "mode",
    ("retained_skip_gs", "retained_gs"),
)
def test_retained_debug_selection_matches_compact_hbm_layout(mode):
    candidate_lens = torch.tensor([2176, 8192], dtype=torch.int32)

    selected = build_dsa_debug_selection(candidate_lens, mode)

    assert selected is not None
    assert selected.shape == (2, 1, 2048)
    assert selected.dtype == torch.int32
    expected_2176 = torch.cat(
        (
            torch.arange(0, 128, dtype=torch.int32),
            torch.arange(256, 2176, dtype=torch.int32),
        )
    )
    expected_8192 = torch.cat(
        (
            torch.arange(0, 128, dtype=torch.int32),
            torch.arange(6272, 8192, dtype=torch.int32),
        )
    )
    assert torch.equal(selected[0, 0], expected_2176)
    assert torch.equal(selected[1, 0], expected_8192)
    assert torch.unique(selected[0, 0]).numel() == 2048
    assert torch.unique(selected[1, 0]).numel() == 2048


def test_last2048_debug_selection_is_contiguous_suffix():
    candidate_lens = torch.tensor([2176, 8192], dtype=torch.int64)

    selected = build_dsa_debug_selection(candidate_lens, "last2048_gs")

    assert selected is not None
    assert selected.dtype == torch.int64
    assert torch.equal(
        selected[0, 0],
        torch.arange(128, 2176, dtype=torch.int64),
    )
    assert torch.equal(
        selected[1, 0],
        torch.arange(6144, 8192, dtype=torch.int64),
    )


@pytest.mark.parametrize("value", ["retained", "skip_gs", "last2048", "NATIVE"])
def test_dsa_debug_selection_rejects_unknown_modes(value):
    with pytest.raises(ValueError, match="NANOVLLM_DSA_DEBUG_SELECTION"):
        parse_dsa_debug_selection(value)


def test_non_native_debug_selection_requires_eager_and_block128():
    assert (
        validate_dsa_debug_selection(
            "retained_gs",
            enforce_eager=True,
            block_size=128,
        )
        == "retained_gs"
    )
    with pytest.raises(ValueError, match="eager-only"):
        validate_dsa_debug_selection(
            "retained_gs",
            enforce_eager=False,
            block_size=128,
        )
    with pytest.raises(ValueError, match="BLOCK_SIZE=128"):
        validate_dsa_debug_selection(
            "retained_gs",
            enforce_eager=True,
            block_size=64,
        )


def test_config_propagates_eager_debug_selection(monkeypatch):
    monkeypatch.setenv("NANOVLLM_DSA_DEBUG_SELECTION", "retained_gs")
    config = object.__new__(Config)
    config.enforce_eager = True
    config.kvcache_block_size = 128
    config.decode_graph_capture_sizes = (1,)
    config.hf_config = SimpleNamespace()

    config._configure_decode_graph()

    assert config.decode_graph_capture_sizes == ()
    assert config.hf_config.nanovllm_dsa_debug_selection == "retained_gs"


def test_config_rejects_non_native_selection_with_full_graph(monkeypatch):
    monkeypatch.setenv("NANOVLLM_DSA_DEBUG_SELECTION", "last2048_gs")
    config = object.__new__(Config)
    config.enforce_eager = False
    config.kvcache_block_size = 128
    config.hf_config = SimpleNamespace()

    with pytest.raises(ValueError, match="eager-only"):
        config._configure_decode_graph()


def test_dsa_debug_selection_rejects_invalid_candidate_lens():
    with pytest.raises(ValueError, match="one-dimensional"):
        build_dsa_debug_selection(
            torch.tensor([[2176]], dtype=torch.int32),
            "retained_gs",
        )
    with pytest.raises(TypeError, match="int32 or int64"):
        build_dsa_debug_selection(
            torch.tensor([2176.0], dtype=torch.float32),
            "retained_gs",
        )
    with pytest.raises(ValueError, match="exceed 2048"):
        build_dsa_debug_selection(
            torch.tensor([2048], dtype=torch.int32),
            "retained_gs",
        )
