#include "kernel_operator.h"
#define C_TEMPLATE 0
#define V_TEMPLATE 1

// OPC generates only this operator's tiling class.  The fused payload has the
// complete production SFA payload as its prefix, so expose it under the type
// name expected by the shared SFA implementation (the same pattern used by
// the non-MTP fused_copy_sfa kernel).
using NanovllmSparseTailAttentionTilingDataMla =
    NanovllmFusedCopySfaMtpTilingData;

#include "../sparse_tail_attention/nanovllm_sparse_tail_attention_kernel_mla.h"
#include "fused_copy_sfa_mtp_scatter_stage.h"

using namespace AscendC;
using namespace FusedCopySfaMtpNs;

namespace {
constexpr uint32_t COPY_READY_FLAG = 15;

template <typename T>
__aicore__ inline void RunFusedMtp(
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
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *attentionWorkspace,
    const NanovllmFusedCopySfaMtpTilingData *fusedTiling,
    __gm__ uint8_t *tiling,
    TPipe *pipe)
{
    if ASCEND_IS_AIV {
        FusedMtpScatterStage<T> scatter(pipe, fusedTiling);
        scatter.Init(
            hbmKeyRope, key, dramKeyRope, dramKvCache,
            hbmBlockTable, dramBlockTable, missSourceIds,
            missDestinationSlots, missCounts);
        scatter.Process();

        // Make every local MTE3 write visible before waking its paired AIC.
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        if (!scatter.UsesOwnerSchedule()) {
            // General batch shapes distribute the union list over all AIVs.
            // The target B=24 path intentionally skips this global barrier:
            // each AIC/AIV owner group touches exactly one request.
            SyncAll();
        }
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(
            COPY_READY_FLAG);
    } else {
        CrossCoreWaitFlag(COPY_READY_FLAG);
    }

    using MtpType = SFAType<
        T, T, T, false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND,
        V_TEMPLATE, SFA_STAGE_NORMAL, true>;
    NanovllmSparseTailAttentionMla<MtpType> attention;
    const auto *attentionTiling = reinterpret_cast<
        const NanovllmSparseTailAttentionTilingDataMla *>(fusedTiling);
    attention.Init(
        query, key, value, sparseIndices, cacheTokens, nullptr,
        actualSeqLengthsQuery, actualSeqLengthsKv, hbmBlockTable,
        queryRope, hbmKeyRope, attentionOut,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        attentionWorkspace, attentionTiling, tiling, pipe);
    attention.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void nanovllm_fused_copy_sfa_mtp(
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
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(
        NanovllmFusedCopySfaMtpTilingData,
        fusedTilingIn, tiling);
    const auto *fusedTiling = &fusedTilingIn;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
            RunFusedMtp<half>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                missSourceIds, missDestinationSlots, missCounts,
                attentionOut, attentionWorkspace, fusedTiling,
                tiling, &pipe);
        } else {
            RunFusedMtp<bfloat16_t>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                missSourceIds, missDestinationSlots, missCounts,
                attentionOut, attentionWorkspace, fusedTiling,
                tiling, &pipe);
        }
    }
}
