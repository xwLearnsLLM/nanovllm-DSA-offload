#define A5_C8_STAGE2_TILING_ONLY
#include "register/op_def_registry.h"
#include "../op_kernel/a5_sparse_tail_attention_c8_mtp_stage2_template_tiling_key.h"
#include "a5_sparse_tail_attention_c8_mtp_stage2_tiling.h"

namespace optiling {
namespace c8_mtp {
ge::graphStatus TilingKvQuantSparseFlashAttention(gert::TilingContext *context);
ge::graphStatus TilingPrepareForKvQuantSparseFlashAttention(
    gert::TilingParseContext *context);
} // namespace c8_mtp

IMPL_OP_OPTILING(A5SparseTailAttentionC8MtpStage2)
    .Tiling(c8_mtp::TilingKvQuantSparseFlashAttention)
    .TilingParse<c8_mtp::KvQuantSparseFlashAttentionCompileInfo>(
        c8_mtp::TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling
