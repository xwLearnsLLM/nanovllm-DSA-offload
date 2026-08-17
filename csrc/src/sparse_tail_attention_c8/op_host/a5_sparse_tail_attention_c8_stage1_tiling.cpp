#include "register/op_def_registry.h"
#include "../op_kernel/a5_qsfa/sparse_tail_attention_c8_stage1_template_tiling_key.h"
#include "kv_quant_sparse_flash_attention_tiling.h"

namespace optiling {
ge::graphStatus TilingKvQuantSparseFlashAttention(gert::TilingContext *context);
ge::graphStatus TilingPrepareForKvQuantSparseFlashAttention(gert::TilingParseContext *context);

IMPL_OP_OPTILING(A5SparseTailAttentionC8Stage1)
    .Tiling(TilingKvQuantSparseFlashAttention)
    .TilingParse<KvQuantSparseFlashAttentionCompileInfo>(
        TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling
