#include "kernel_operator.h"
#include "sfa_impl/nanovllm_sparse_tail_attention_template_tiling_key.h"

// The fused tiling payload starts with the complete production SFA payload.
// The source-aware gather changes live in a private kernel fork so the
// standalone SFA baseline remains byte-identical to nano-vLLM.
using NanovllmSparseTailAttentionTilingDataMla =
    NanovllmFusedCopySfaTilingData;

#include "sfa_impl/nanovllm_sparse_tail_attention_kernel_mla.h"

using namespace AscendC;

template <typename T, int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T,
          int TEMPLATE_MODE>
__aicore__ inline void RunSourceAwareAttention(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *sourceTokenIds,
    __gm__ uint8_t *copyCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *attentionWorkspace,
    const NanovllmFusedCopySfaTilingData *fusedTiling,
    __gm__ uint8_t *tiling,
    TPipe *pipe)
{
    using FusedType = SFAType<
        T, T, T, FLASH_DECODE,
        static_cast<SFA_LAYOUT>(LAYOUT_T),
        static_cast<SFA_LAYOUT>(KV_LAYOUT_T),
        TEMPLATE_MODE, SFA_STAGE_NORMAL, true>;
    NanovllmSparseTailAttentionMla<FusedType> op;
    const auto *attentionTiling =
        reinterpret_cast<
            const NanovllmSparseTailAttentionTilingDataMla *>(
                fusedTiling);
    op.Init(
        query, key, value, sparseIndices, cacheTokens, nullptr,
        actualSeqLengthsQuery, actualSeqLengthsKv, hbmBlockTable,
        queryRope, hbmKeyRope, attentionOut,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        attentionWorkspace, attentionTiling, tiling, pipe);
    op.InitSourceAwareGather(
        dramKeyRope, dramKvCache, dramBlockTable, sourceTokenIds,
        copyCounts, fusedTiling->copyCap,
        fusedTiling->dramMaxBlockNum);
    op.Process();
}

template <int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void nanovllm_fused_copy_sfa(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *sourceTokenIds,
    __gm__ uint8_t *copyCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(
        NanovllmFusedCopySfaTilingData,
        fusedTilingIn, tiling);
    const auto *fusedTiling = &fusedTilingIn;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        RunSourceAwareAttention<
            half, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                sourceTokenIds, copyCounts, attentionOut,
                attentionWorkspace, fusedTiling, tiling, &pipe);
    } else {
        RunSourceAwareAttention<
            bfloat16_t, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T,
            TEMPLATE_MODE>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                sourceTokenIds, copyCounts, attentionOut,
                attentionWorkspace, fusedTiling, tiling, &pipe);
    }
}
