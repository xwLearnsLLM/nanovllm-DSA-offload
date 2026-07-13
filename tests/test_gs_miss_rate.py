import pytest

from nanovllm.engine.dsa_offload import (
    compute_gs_miss_counts,
    parse_gs_miss_rate_layers,
)


def test_parse_gs_miss_rate_layers():
    assert parse_gs_miss_rate_layers(None, 61) == frozenset()
    assert parse_gs_miss_rate_layers("  ", 61) == frozenset()
    assert parse_gs_miss_rate_layers("0, 30,60,30", 61) == frozenset(
        {0, 30, 60}
    )


@pytest.mark.parametrize("value", ["0,,30", "layer0", "-1", "61"])
def test_parse_gs_miss_rate_layers_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="NANOVLLM_GS_MISS_RATE_ON_LAYERS"):
        parse_gs_miss_rate_layers(value, 61)


def test_compute_gs_miss_counts_uses_set_difference():
    assert compute_gs_miss_counts(
        [[1, 2, 2, 3, -1], [10, 11]],
        [[2, 4, -1], [10, 11, 12]],
    ) == [2, 0]


def test_compute_gs_miss_counts_requires_matching_rows():
    with pytest.raises(ValueError, match="row counts"):
        compute_gs_miss_counts([[1]], [])
