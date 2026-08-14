/**
 * Quantized LightningIndexer phase embedded in fused_li_manage_c8.
 *
 * The implementation is derived from the Ascend 950 C8 Quant
 * LightningIndexer services used by vLLM-Ascend.  It writes the native
 * top-2048 token IDs directly into the caller-owned source_ids buffer; the
 * request-pool phase running later in the same MIX kernel consumes that row
 * in place.
 */

#ifndef A5_FUSED_LI_MANAGE_C8_QLI_H
#define A5_FUSED_LI_MANAGE_C8_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_fused_li_manage_c8_tiling.h"
#include "arch35/quant_lightning_indexer_common.h"
#include "arch35/quant_lightning_indexer_service_cube.h"
#include "arch35/quant_lightning_indexer_service_vector.h"

namespace a5_fused_li_manage_c8 {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

constexpr uint32_t QLI_BLOCK_SIZE = 128;
constexpr uint32_t QLI_HEAD_DIM = 128;
constexpr uint32_t QLI_TOPK = 2048;
constexpr uint32_t REQUEST_DONE_EVENT = 6;

using C8QliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class QuantLiPhase {
public:
    __aicore__ inline QuantLiPhase(
        TPipe *pipe, const A5FusedLiManageC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR weights,
        GM_ADDR queryDequantScale, GM_ADDR keyDequantScale,
        GM_ADDR cacheTokens, GM_ADDR candidateLens,
        GM_ADDR blockTable, GM_ADDR topkIndices,
        GM_ADDR userWorkspace)
    {
        if ASCEND_IS_AIV {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_;
        }

        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = QLI_HEAD_DIM;
        constInfo_.sparseCount = QLI_TOPK;
        constInfo_.kSeqSize = tiling_->maxCandidateLen;
        constInfo_.qSeqSize = 1;
        constInfo_.kCacheBlockSize = QLI_BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = LI_LAYOUT::TND;
        constInfo_.attenMaskFlag = false;
        constInfo_.cmpRatio = 1;
        constInfo_.batchSupperFlag = false;
        constInfo_.stride = tiling_->keyStride;
        constInfo_.scaleStride = tiling_->scaleStride;
        constInfo_.mBaseSize = 256;
        constInfo_.s1BaseSize =
            (constInfo_.mBaseSize + constInfo_.gSize - 1U) /
            constInfo_.gSize;
        constInfo_.s2BaseSize = QLI_BLOCK_SIZE;

        GlobalTensor<uint16_t> scoreWorkspace;
        scoreWorkspace.SetGlobalBuffer(
            (__gm__ uint16_t *)(userWorkspace +
                static_cast<uint64_t>(aiCoreIdx_) *
                    tiling_->scoreWorkspaceStride));

        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
            queryScaleGm_.SetGlobalBuffer(
                (__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer((__gm__ float *)keyDequantScale);
            topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
            vectorService_.InitParams(constInfo_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                topkIndicesGm_, blockTableGm_);
            vectorService_.InitVecWorkspaceTensor(scoreWorkspace);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)key);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        bool hasActiveRow = false;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            if (IsActiveRow(batch)) {
                hasActiveRow = true;
                break;
            }
        }
        if (!hasActiveRow) {
            return;
        }

        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            if (!IsActiveRow(batch)) {
                continue;
            }
            const uint32_t candidate = static_cast<uint32_t>(
                candidateLensGm_.GetValue(batch));
            const uint32_t loopCount = candidate / QLI_BLOCK_SIZE;
            for (uint32_t s2 = 0; s2 < loopCount; ++s2, ++globalLoop) {
                RunInfo run{};
                run.loop = globalLoop;
                run.bN2Idx = batch;
                run.bIdx = batch;
                run.n2Idx = 0;
                run.gS1Idx = 0;
                run.s2Idx = s2;
                run.actS1Size = 1;
                run.actS2Size = candidate;
                run.actS2SizeOrig = candidate;
                run.actMBaseSize = tiling_->indexHeads;
                run.actualSingleProcessSInnerSize = QLI_BLOCK_SIZE;
                run.actualSingleProcessSInnerSizeAlign = QLI_BLOCK_SIZE;
                run.tensorQueryOffset =
                    static_cast<uint64_t>(batch) * tiling_->indexHeads *
                    QLI_HEAD_DIM;
                run.tensorKeyOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE *
                    QLI_HEAD_DIM;
                run.tensorKeyScaleOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE;
                run.tensorWeightsOffset =
                    static_cast<uint64_t>(batch) * tiling_->indexHeads;
                run.indiceOutOffset =
                    static_cast<uint64_t>(batch) * QLI_TOPK;
                run.isFirstS2InnerLoop = s2 == 0;
                run.isLastS2InnerLoop = s2 + 1U == loopCount;
                run.isAllLoopEnd = false;
                run.isValid = true;

                if ASCEND_IS_AIC {
                    matmulService_.ComputeMm1(run);
                } else {
                    vectorService_.ProcessVec1(run);
                    if (run.isLastS2InnerLoop) {
                        vectorService_.ProcessTopK(run);
                    }
                }
            }

            // The Cube may reuse its score workspace only after the even AIV
            // has consumed the completed row in TopK.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                    REQUEST_DONE_EVENT);
            } else if ((subBlockIdx_ & 1U) == 0U) {
                CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                    REQUEST_DONE_EVENT);
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    __aicore__ inline bool IsActiveRow(uint32_t batch)
    {
        const int32_t budget = cacheTokensGm_.GetValue(batch);
        const int32_t candidate = candidateLensGm_.GetValue(batch);
        return budget != 0 &&
            candidate >= static_cast<int32_t>(QLI_TOPK) &&
            candidate <= static_cast<int32_t>(tiling_->maxCandidateLen) &&
            candidate % static_cast<int32_t>(QLI_BLOCK_SIZE) == 0;
    }

    TPipe *pipe_;
    const A5FusedLiManageC8TilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<C8QliType> matmulService_;
    QLIVector<C8QliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> topkIndicesGm_;
};
} // namespace a5_fused_li_manage_c8

#endif
