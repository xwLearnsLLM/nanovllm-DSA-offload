#!/usr/bin/env python3
"""Validate the public schemas plus Meta/Fake behavior without NPU allocation."""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import nanovllm_dsa_a5  # noqa: F401


BLOCK = 128
TOPK = 2048


def same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left is right or left._cdata == right._cdata


def tensors(device: str = "meta") -> dict[str, torch.Tensor]:
    batch = 3
    source_capacity = 4096
    source_blocks = source_capacity // BLOCK
    hbm_blocks = 96
    query = torch.empty((batch, 32, 128), dtype=torch.bfloat16, device=device)
    index_key = torch.empty(
        (source_blocks, BLOCK, 1, 128), dtype=torch.bfloat16, device=device
    )
    weights = torch.empty((batch, 32), dtype=torch.bfloat16, device=device)
    req = torch.empty((batch,), dtype=torch.int32, device=device)
    pool = torch.empty((7, source_capacity), dtype=torch.int32, device=device)
    cache_tokens = torch.empty((batch,), dtype=torch.int32, device=device)
    lengths = torch.empty((batch,), dtype=torch.int32, device=device)
    source_table = torch.empty(
        (batch, source_blocks), dtype=torch.int32, device=device
    )
    source_ids = torch.empty((batch, 1, TOPK), dtype=torch.int32, device=device)
    slots = torch.empty_like(source_ids)
    counts = torch.empty((batch,), dtype=torch.int32, device=device)
    hbm_kpe = torch.empty((hbm_blocks, BLOCK, 64), dtype=torch.bfloat16, device=device)
    hbm_ckv = torch.empty((hbm_blocks, BLOCK, 512), dtype=torch.bfloat16, device=device)
    dram_kpe = torch.empty(
        (source_blocks, BLOCK, 64), dtype=torch.bfloat16, device=device
    )
    dram_ckv = torch.empty(
        (source_blocks, BLOCK, 512), dtype=torch.bfloat16, device=device
    )
    hbm_table = torch.empty((batch, 32), dtype=torch.int32, device=device)
    attention_query = torch.empty(
        (batch, 8, 512), dtype=torch.bfloat16, device=device
    )
    query_rope = torch.empty(
        (batch, 8, 64), dtype=torch.bfloat16, device=device
    )
    key_rope = hbm_kpe.view(hbm_blocks, BLOCK, 1, 64)
    actual_q = torch.empty((batch,), dtype=torch.int32, device=device)
    actual_kv = torch.empty((batch,), dtype=torch.int32, device=device)
    return {
        "query": query,
        "index_key": index_key,
        "weights": weights,
        "req": req,
        "pool": pool,
        "cache_tokens": cache_tokens,
        "lengths": lengths,
        "source_table": source_table,
        "source_ids": source_ids,
        "slots": slots,
        "counts": counts,
        "hbm_kpe": hbm_kpe,
        "hbm_ckv": hbm_ckv,
        "dram_kpe": dram_kpe,
        "dram_ckv": dram_ckv,
        "hbm_table": hbm_table,
        "attention_query": attention_query,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "actual_q": actual_q,
        "actual_kv": actual_kv,
    }


def run_api(values: dict[str, torch.Tensor]) -> None:
    lidu = torch.ops.nanovllm_dsa.lidu_decode_update.default(
        values["query"],
        values["index_key"],
        values["weights"],
        values["req"],
        values["pool"],
        values["cache_tokens"],
        values["lengths"],
        values["source_table"],
    )
    assert [tuple(item.shape) for item in lidu] == [
        (3, 1, TOPK),
        (3, 1, TOPK),
        (3,),
        (7, 4096),
    ]
    assert same_tensor(lidu[3], values["pool"])

    lidu_out = torch.ops.nanovllm_dsa.lidu_decode_update_out.default(
        values["query"],
        values["index_key"],
        values["weights"],
        values["req"],
        values["pool"],
        values["cache_tokens"],
        values["lengths"],
        values["source_table"],
        values["source_ids"],
        values["slots"],
        values["counts"],
    )
    for result, expected in zip(
        lidu_out,
        (values["source_ids"], values["slots"], values["counts"], values["pool"]),
    ):
        assert same_tensor(result, expected)

    scatter = torch.ops.nanovllm_dsa.scatter_copy.default(
        values["hbm_kpe"],
        values["hbm_ckv"],
        values["dram_kpe"],
        values["dram_ckv"],
        values["hbm_table"],
        values["source_table"],
        values["source_ids"].view(3, TOPK),
        values["slots"].view(3, TOPK),
        values["counts"],
    )
    assert same_tensor(scatter[0], values["hbm_kpe"])
    assert same_tensor(scatter[1], values["hbm_ckv"])

    attention = torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
        values["attention_query"],
        values["hbm_ckv"].view(96, BLOCK, 1, 512),
        values["hbm_ckv"].view(96, BLOCK, 1, 512),
        values["slots"],
        values["cache_tokens"],
        values["hbm_table"],
        values["actual_q"],
        values["actual_kv"],
        values["query_rope"],
        values["key_rope"],
        1.0,
    )
    assert tuple(attention.shape) == (3, 8, 512)
    assert attention.dtype == torch.bfloat16


def expect_sfa_head_limit(
    values: dict[str, torch.Tensor],
    query: torch.Tensor,
    query_rope: torch.Tensor,
) -> None:
    key = values["hbm_ckv"].view(96, BLOCK, 1, 512)
    try:
        torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
            query,
            key,
            key,
            values["slots"],
            values["cache_tokens"],
            values["hbm_table"],
            values["actual_q"],
            values["actual_kv"],
            query_rope,
            values["key_rope"],
            1.0,
        )
    except RuntimeError as error:
        if "1 <= N <= 64" not in str(error):
            raise
    else:
        raise AssertionError("SFA must reject q_head > 64")


def check_sfa_head_limit_meta() -> None:
    values = tensors()
    query = torch.empty((3, 65, 512), dtype=torch.bfloat16, device="meta")
    query_rope = torch.empty((3, 65, 64), dtype=torch.bfloat16, device="meta")
    expect_sfa_head_limit(values, query, query_rope)


def fake_values() -> dict[str, torch.Tensor]:
    mode = FakeTensorMode()
    real = tensors("cpu")
    bad_query = torch.empty((3, 65, 512), dtype=torch.bfloat16)
    bad_query_rope = torch.empty((3, 65, 64), dtype=torch.bfloat16)
    with mode:
        fake = {name: mode.from_tensor(value) for name, value in real.items()}
        run_api(fake)
        expect_sfa_head_limit(
            fake,
            mode.from_tensor(bad_query),
            mode.from_tensor(bad_query_rope),
        )
    return fake


def check_schemas() -> None:
    schemas = {
        "lidu": str(torch.ops.nanovllm_dsa.lidu_decode_update.default._schema),
        "lidu_out": str(torch.ops.nanovllm_dsa.lidu_decode_update_out.default._schema),
        "scatter": str(torch.ops.nanovllm_dsa.scatter_copy.default._schema),
    }
    if "Tensor(a!) cache_slots_pool" not in schemas["lidu"]:
        raise AssertionError("LIDU cache state is not declared mutable")
    if not all(alias in schemas["lidu_out"] for alias in ("Tensor(b!)", "Tensor(c!)", "Tensor(d!)")):
        raise AssertionError("LIDU out buffers are not declared mutable aliases")
    if "Tensor(a!) hbm_kpe" not in schemas["scatter"] or "Tensor(b!) hbm_ckv" not in schemas["scatter"]:
        raise AssertionError("SCATTER HBM caches are not declared mutable aliases")


def main() -> None:
    check_schemas()
    run_api(tensors())
    check_sfa_head_limit_meta()
    fake_values()
    print("A5_NANOVLLM_DSA_META_FAKE_UT_OK", flush=True)


if __name__ == "__main__":
    main()
