#ifndef NANOVLLM_FUSED_COPY_SFA_TILING_H
#define NANOVLLM_FUSED_COPY_SFA_TILING_H

#include "../../sparse_tail_attention/op_host/nanovllm_sparse_tail_attention_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(NanovllmFusedCopySfaTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NanovllmSparseTailAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(NanovllmSparseTailAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(NanovllmSparseTailAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NanovllmSparseTailAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(NanovllmSparseTailAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF(uint32_t, copyCap);
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlockNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(
    NanovllmFusedCopySfa,
    NanovllmFusedCopySfaTilingData)

struct NanovllmFusedCopySfaCompileInfo {
};

}  // namespace optiling

#endif
