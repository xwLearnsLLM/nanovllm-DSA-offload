#include "register/op_def_registry.h"
#include "../op_kernel/a5_sparse_tail_attention_c8_tnd_probe_template_tiling_key.h"
#include "a5_sparse_tail_attention_c8_tnd_probe_tiling.h"

namespace optiling {
namespace c8_mtp {
ge::graphStatus TilingKvQuantSparseFlashAttention(gert::TilingContext *context);
ge::graphStatus TilingPrepareForKvQuantSparseFlashAttention(
    gert::TilingParseContext *context);
} // namespace c8_mtp

IMPL_OP_OPTILING(A5SparseTailAttentionC8TndProbe)
    .Tiling(c8_mtp::TilingKvQuantSparseFlashAttention)
    .TilingParse<c8_mtp::KvQuantSparseFlashAttentionCompileInfo>(
        c8_mtp::TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling
