import sys
from types import SimpleNamespace

import torch

from nanovllm.layers.layernorm import RMSNorm


def test_rms_norm_cpu_fallback_matches_reference():
    norm = RMSNorm(4, eps=1e-5)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    residual = torch.tensor([[0.5, 1.0, -1.5, 2.0]])

    expected_plain = x * torch.rsqrt(
        x.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    expected_plain = expected_plain * norm.weight
    expected_residual = x + residual
    expected = expected_residual * torch.rsqrt(
        expected_residual.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    expected = expected * norm.weight

    torch.testing.assert_close(norm(x), expected_plain)
    actual, actual_residual = norm(x, residual)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, expected_residual)


def test_rms_norm_npu_path_uses_official_ops(monkeypatch):
    normalized = object()
    updated_residual = object()
    calls = []

    def npu_rms_norm(x, weight, eps):
        calls.append(("rms", x, weight, eps))
        return normalized, object()

    def npu_add_rms_norm(x, residual, weight, eps):
        calls.append(("add_rms", x, residual, weight, eps))
        return normalized, object(), updated_residual

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_rms_norm=npu_rms_norm,
            npu_add_rms_norm=npu_add_rms_norm,
        ),
    )

    norm = RMSNorm(4, eps=1e-5)
    x = SimpleNamespace(device=SimpleNamespace(type="npu"))
    residual = object()

    assert norm.rms_forward(x) is normalized
    assert norm.add_rms_forward(x, residual) == (
        normalized,
        updated_residual,
    )
    assert calls == [
        ("rms", x, norm.weight, norm.eps),
        ("add_rms", x, residual, norm.weight, norm.eps),
    ]
