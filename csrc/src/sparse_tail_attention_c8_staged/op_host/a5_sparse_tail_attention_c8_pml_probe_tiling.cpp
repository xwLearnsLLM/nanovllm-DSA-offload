#include "register/op_def_registry.h"
#include "../op_kernel/a5_sparse_tail_attention_c8_pml_probe_template_tiling_key.h"
#include "a5_sparse_tail_attention_c8_pml_probe_tiling.h"

namespace optiling {
ge::graphStatus TilingKvQuantSparseFlashAttention(gert::TilingContext *context);
ge::graphStatus TilingPrepareForKvQuantSparseFlashAttention(
    gert::TilingParseContext *context);

IMPL_OP_OPTILING(A5SparseTailAttentionC8PmlProbe)
    .Tiling(TilingKvQuantSparseFlashAttention)
    .TilingParse<KvQuantSparseFlashAttentionCompileInfo>(
        TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling
