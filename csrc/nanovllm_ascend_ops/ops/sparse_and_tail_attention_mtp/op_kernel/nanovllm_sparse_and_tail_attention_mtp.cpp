/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "kernel_operator.h"
#include "nanovllm_sparse_and_tail_attention_mtp_template_tiling_key.h"
#include "../sparse_and_tail_attention/nanovllm_sparse_and_tail_attention_kernel_mla.h"

using namespace AscendC;

#define SFA_MTP_OP_IMPL(templateClass, tilingdataClass, ...)                    \
    do {                                                                        \
        templateClass<SFAType<__VA_ARGS__, SFA_STAGE_NORMAL, true>> op;         \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);   \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;        \
        op.Init(query, key, value, sparseIndices, cacheTokens, nullptr,         \
                actualSeqLengthsQuery, actualSeqLengthsKV, blocktable,          \
                queryRope, keyRope, attentionOut, nullptr, nullptr, nullptr,    \
                nullptr, nullptr, nullptr, user, tiling_data, tiling, &tPipe);  \
        op.Process();                                                           \
    } while (0)

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void nanovllm_sparse_and_tail_attention_mtp(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 &&
                  ORIG_DTYPE_KEY == DT_FLOAT16 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        SFA_MTP_OP_IMPL(
            NanovllmSparseAndTailAttentionMla,
            NanovllmSparseAndTailAttentionTilingDataMla,
            half, half, half, FLASH_DECODE,
            static_cast<SFA_LAYOUT>(LAYOUT_T),
            static_cast<SFA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE);
    } else {
        SFA_MTP_OP_IMPL(
            NanovllmSparseAndTailAttentionMla,
            NanovllmSparseAndTailAttentionTilingDataMla,
            bfloat16_t, bfloat16_t, bfloat16_t, FLASH_DECODE,
            static_cast<SFA_LAYOUT>(LAYOUT_T),
            static_cast<SFA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE);
    }
}
