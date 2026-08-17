# Vendored C8 QSFA

The QSFA compute/tiling sources in `op_kernel/c8_vendor` and
`op_host/a5_sparse_tail_attention_c8_mtp_stage1_tiling.*` come from:

- repository: `https://gitcode.com/cann/ops-transformer.git`
- branch: `9.1.0`
- commit: `eb08f13f280465403e4f1f9e42d98dc21a17754e`
- source operator: `attention/kv_quant_sparse_flash_attention`

The required `attention/common/op_kernel` and
`attention/sparse_flash_attention/op_kernel/arch35/common` dependency closure
is included with trailing whitespace normalized.
`LICENSE.cann-ops-transformer` contains the upstream CANN Open Software
License Agreement 2.0.
The upstream `util_regbase.h` expects the build-system-provided `util.h`;
the compatible copy already used by this repository's A5 SFA baseline is
included beside it so the vendored tree is self-contained.

The checked-in `ops.json` deliberately uses only BF16/FP16 plus INT8 for the
msopgen scaffold because released msopgen IR parsers do not accept the
`float8_e4m3fn` or `hifloat8` tokens.  The repository OpDef that overwrites the
generated stub still registers the real FP8/HIFLOAT8/INT8 combinations.  The
Torch adapter passes native FP8 tensors directly and, matching op-plugin 26.0,
uses an ACL `TensorWrapper` when a UINT8-backed HIFLOAT8 tensor is accompanied
by `kv_dtype=torch_npu.hifloat8`.

The public 26.0.0 PyTorch binding was cross-checked against
`Ascend/op-plugin` branch `26.0.0`, commit
`1e29069792b034752fef906f9b70164df1106c74`, file
`op_plugin/ops/opapi/KvQuantSparseFlashAttentionKernelNpuOpApi.cpp`.  That
file is only an `aclnnKvQuantSparseFlashAttention` single-output adapter; it
does not contain the device kernel or expose P/M/L.  The repository-local
Torch adapter therefore targets the renamed vendored operators and cannot
fall back to that system symbol.

Repository modifications are intentionally limited to the staged ABI:

- the state kernel writes unnormalized FP32 P, max M, and denominator L;
- empty rows write the exact identity `(P=0, M=-inf, L=0)`;
- Stage1 interprets the compact LI miss-prefix/hit-suffix tensor and
  synthesizes the causal resident tail indices inside the kernel;
- Stage1 caches each request's TND query range/resident metadata before its
  local-query loop, and dispatches each Vector S2 sub-tile to a branch-free
  hit or tail resolver (only the single boundary sub-tile stays mixed);
- Stage2 reads Stage1 P/M/L after its own P/M/L is ready in UB, performs the
  stable merge and normalization there, and directly writes attention output;
- QK, softmax update, and PV computation remain identical to the vendored
  QSFA body; only slot selection and the final state/output boundary change;
- both ops are renamed to avoid resolving the system QSFA implementation.

Two private performance probes instantiate the same source with compile-time
modes.  One restores the upstream full-slot/output path and optionally appends
P/M/L GM writes; the other compares that upstream path with compact-top-k plus
TND tail resolution at an equal token count.  Probe output values are not a
numerical contract and the probe symbols are not part of the public Python API.
