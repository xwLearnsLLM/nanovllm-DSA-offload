# DsaIndexUpdate

Standalone CANN custom op for nano-vllm DSA decode offload.

## Interface

`DsaIndexUpdate(score, hbm_cached_tokens_pool, candidate_lens, selected_lens, req_pool_entries, max_copy_tokens) -> promote_idx, demote_idx, copy_counts`

Inputs:

- `score`: `bf16`, shape `(bsz, max_candidate_len)`. The kernel may overwrite selected score positions with `-inf`; callers should treat it as scratch.
- `hbm_cached_tokens_pool`: `int32`, shape `(pool_capacity, max_sparse_tokens)`. Updated in place.
- `candidate_lens`: `int32`, shape `(bsz,)`.
- `selected_lens`: `int32`, shape `(bsz,)`.
- `req_pool_entries`: `int32`, shape `(bsz,)`.
- `max_copy_tokens`: int attr, current kernel supports `1 <= max_copy_tokens <= 128`.

Outputs:

- `promote_idx`: `int32`, shape `(bsz, output_capacity)`, where `output_capacity >= max_copy_tokens`.
- `demote_idx`: `int32`, shape `(bsz, output_capacity)`, same capacity as `promote_idx`.
- `copy_counts`: `int32`, shape `(bsz,)`.

For each request row `b`, the kernel computes:

- `copy_counts[b] = min(max_copy_tokens, selected_lens[b], max(candidate_lens[b] - selected_lens[b], 0))`.
- `promote_idx[b, :copy_counts[b]]`: highest-score uncached original token ids.
- `demote_idx[b, :copy_counts[b]]`: lowest-score local sparse slots in `hbm_cached_tokens_pool[req_pool_entries[b]]`.
- `hbm_cached_tokens_pool[req_pool_entries[b], demote_idx] = promote_idx`.

Rows with no uncached tokens, empty selected budget, or invalid pool entries produce `copy_counts[b] = 0`.

## Build

Use the repository-level script instead of invoking this directory directly:

```bash
bash scripts/build_dsa_index_update_op.sh
```

Accuracy and performance are checked by:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python ut_ops/probe_dsa_index_update.py --device npu:0
```
