from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config
from nanovllm.engine.dsa_offload import (
    DSANativeSelectionStats,
    build_dsa_debug_selection,
    dsa_debug_prints_native_stats,
    dsa_debug_rotary_mode,
    dsa_debug_uses_native_selection,
    parse_dsa_debug_selection,
    summarize_dsa_native_selection,
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
    ("mode", "rotary_mode"),
    (
        ("native_interleave_stats", "interleave"),
        ("native_half_stats", "half"),
    ),
)
def test_native_stats_modes_select_native_topk_and_override_rope(
    mode,
    rotary_mode,
):
    assert dsa_debug_uses_native_selection(mode)
    assert dsa_debug_prints_native_stats(mode)
    assert dsa_debug_rotary_mode(mode, "half") == rotary_mode
    assert build_dsa_debug_selection(
        torch.tensor([8192], dtype=torch.int32),
        mode,
    ) is None


def test_plain_native_keeps_configured_rope_without_stats():
    assert dsa_debug_uses_native_selection("native")
    assert not dsa_debug_prints_native_stats("native")
    assert dsa_debug_rotary_mode("native", "interleave") == "interleave"


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


@pytest.mark.parametrize(
    "mode",
    ("native_interleave_stats", "native_half_stats"),
)
def test_native_stats_diagnostics_require_eager_and_block128(mode):
    assert (
        validate_dsa_debug_selection(
            mode,
            enforce_eager=True,
            block_size=128,
        )
        == mode
    )
    with pytest.raises(ValueError, match="eager-only"):
        validate_dsa_debug_selection(
            mode,
            enforce_eager=False,
            block_size=128,
        )
    with pytest.raises(ValueError, match="BLOCK_SIZE=128"):
        validate_dsa_debug_selection(
            mode,
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


def test_summarize_native_selection_reports_retained_and_distribution():
    row0 = torch.cat(
        (
            torch.arange(0, 128, dtype=torch.int32),
            torch.arange(6272, 8192, dtype=torch.int32),
        )
    )
    row1 = torch.arange(6144, 8192, dtype=torch.int32)
    topk = torch.stack((row0, row1)).unsqueeze(1)

    summaries = summarize_dsa_native_selection(
        topk,
        torch.tensor([8192, 8192], dtype=torch.int32),
    )

    assert summaries == [
        DSANativeSelectionStats(
            row=0,
            candidate_len=8192,
            valid_count=2048,
            unique_count=2048,
            invalid_count=0,
            duplicate_count=0,
            min_index=0,
            max_index=8191,
            retained_overlap=2048,
            last2048_overlap=1920,
            tail128_count=128,
            quartile_counts=(128, 0, 0, 1920),
        ),
        DSANativeSelectionStats(
            row=1,
            candidate_len=8192,
            valid_count=2048,
            unique_count=2048,
            invalid_count=0,
            duplicate_count=0,
            min_index=6144,
            max_index=8191,
            retained_overlap=1920,
            last2048_overlap=2048,
            tail128_count=128,
            quartile_counts=(0, 0, 0, 2048),
        ),
    ]


def test_summarize_native_selection_counts_invalid_and_duplicates():
    topk = torch.tensor(
        [[[-1, 0, 0, 127, 128, 4095, 4096]]],
        dtype=torch.int32,
    )

    summary = summarize_dsa_native_selection(
        topk,
        torch.tensor([4096], dtype=torch.int64),
    )[0]

    assert summary.valid_count == 5
    assert summary.unique_count == 4
    assert summary.invalid_count == 2
    assert summary.duplicate_count == 1
    assert summary.retained_overlap == 3
    assert summary.last2048_overlap == 1
    assert summary.quartile_counts == (3, 0, 0, 1)
