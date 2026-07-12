import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_strided_batched_matmul_tla():
    """Compare the CATLASS strided batched matmul (TLA) wrapper against a reference computation.

    Golden logic (from examples/45_strided_batched_matmul_tla/strided_batched_matmul_tla.cpp):
    For each batch b: C[b] = A[b] @ B[b]
    where A[b] is [M, K] and B[b] is [K, N], with strided memory access
    between batches.
    """
    batch, m, n, k = 2, 256, 256, 256

    a = torch.randn(batch, m, k, dtype=torch.float16, device="npu")
    b = torch.randn(batch, k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.strided_batched_matmul_tla(
        a, b, "float16", transA=False, transB=False
    )

    expected = torch.matmul(a, b)

    assert result.shape == (batch, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
