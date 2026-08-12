#ifndef NANOVLLM_FUSED_COPY_SFA_MTP_UNION_SCATTER_H
#define NANOVLLM_FUSED_COPY_SFA_MTP_UNION_SCATTER_H

#include "kernel_operator.h"

namespace FusedCopySfaMtpNs {
using namespace AscendC;

constexpr int64_t MTP_SCATTER_BLOCK_SIZE = 128;
constexpr int64_t MTP_SCATTER_BLOCK_SHIFT = 7;
constexpr int64_t MTP_SCATTER_BLOCK_MASK =
    MTP_SCATTER_BLOCK_SIZE - 1;
constexpr int64_t MTP_SCATTER_KPE_DIM = 64;
constexpr int64_t MTP_SCATTER_CKV_DIM = 512;
constexpr int64_t MTP_SCATTER_UNION_CAPACITY = 8192;
constexpr int64_t MTP_SCATTER_KPE_BYTES =
    MTP_SCATTER_KPE_DIM * sizeof(uint16_t);
constexpr int64_t MTP_SCATTER_CKV_BYTES =
    MTP_SCATTER_CKV_DIM * sizeof(uint16_t);

// Copies each unique union miss exactly once.  The compact concatenation of
// all requests' valid miss lists is divided into contiguous, balanced ranges
// across all AIVs.  The source-aware Attention path reads current misses
// directly from DRAM, so no copy->Attention barrier is required; this stage
// only updates persistent HBM.
template <typename T>
class FusedMtpUnionScatterStage {
public:
    __aicore__ inline FusedMtpUnionScatterStage(
        TPipe *pipe,
        const NanovllmFusedCopySfaMtpTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR hbmKpe,
        GM_ADDR hbmCkv,
        GM_ADDR dramKpe,
        GM_ADDR dramCkv,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR missSrcIds,
        GM_ADDR missDstSlots,
        GM_ADDR missCounts)
    {
        blockIdx_ = GetBlockIdx();
        logicalCoreCount_ =
            tiling_->singleCoreParams.usedCoreNum * 2U;
        batchSize_ = tiling_->baseParams.batchSize;
        kpeUbOffset_ = MTP_SCATTER_CKV_BYTES / sizeof(T);

        pipe_->InitBuffer(
            copyQueue_, 2,
            MTP_SCATTER_CKV_BYTES + MTP_SCATTER_KPE_BYTES);

        hbmKpeGm_.SetGlobalBuffer((__gm__ T *)hbmKpe);
        hbmCkvGm_.SetGlobalBuffer((__gm__ T *)hbmCkv);
        dramKpeGm_.SetGlobalBuffer((__gm__ T *)dramKpe);
        dramCkvGm_.SetGlobalBuffer((__gm__ T *)dramCkv);
        hbmBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t *)dramBlockTable);
        missSrcIdsGm_.SetGlobalBuffer((__gm__ int32_t *)missSrcIds);
        missDstSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)missDstSlots);
        missCountsGm_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
    }

    __aicore__ inline void Process()
    {
        if (logicalCoreCount_ == 0 || blockIdx_ >= logicalCoreCount_) {
            return;
        }
        int64_t totalMisses = 0;
        for (uint32_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            const int32_t missCount = missCountsGm_.GetValue(batchIdx);
            ASSERT_MSG(
                missCount >= 0 &&
                    missCount <= MTP_SCATTER_UNION_CAPACITY,
                "MTP union miss_count exceeds 8192.");
            totalMisses += missCount;
        }
        const int64_t workStart = totalMisses * blockIdx_ /
            logicalCoreCount_;
        const int64_t workEnd = totalMisses * (blockIdx_ + 1) /
            logicalCoreCount_;
        if (workStart >= workEnd) {
            return;
        }

        uint32_t batchIdx = 0;
        int64_t requestStart = 0;
        int32_t missCount = 0;
        while (batchIdx < batchSize_) {
            missCount = missCountsGm_.GetValue(batchIdx);
            if (workStart < requestStart + missCount) {
                break;
            }
            requestStart += missCount;
            ++batchIdx;
        }
        ASSERT_MSG(batchIdx < batchSize_, "Invalid compact MTP miss range.");
        if (batchIdx >= batchSize_) {
            return;
        }

        int32_t copyIdx = static_cast<int32_t>(workStart - requestStart);
        CopyAddress currentAddress;
        if (!ResolveAddress(batchIdx, copyIdx, currentAddress)) {
            return;
        }
        CopyIn(currentAddress);
        for (int64_t compactIdx = workStart + 1;
             compactIdx < workEnd;
             ++compactIdx) {
            ++copyIdx;
            while (copyIdx >= missCount) {
                ++batchIdx;
                ASSERT_MSG(
                    batchIdx < batchSize_,
                    "Compact MTP miss range exceeds batch metadata.");
                if (batchIdx >= batchSize_) {
                    CopyOut(currentAddress);
                    return;
                }
                missCount = missCountsGm_.GetValue(batchIdx);
                copyIdx = 0;
            }
            CopyAddress nextAddress;
            if (!ResolveAddress(batchIdx, copyIdx, nextAddress)) {
                CopyOut(currentAddress);
                return;
            }
            CopyIn(nextAddress);
            CopyOut(currentAddress);
            currentAddress = nextAddress;
        }
        CopyOut(currentAddress);

        // Publish this AIV's persistent writes before it enters Attention.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

private:
    struct CopyAddress {
        int64_t srcCkv = 0;
        int64_t dstCkv = 0;
        int64_t srcKpe = 0;
        int64_t dstKpe = 0;
    };

    __aicore__ inline bool ResolveAddress(
        uint32_t batchIdx, int32_t copyIdx, CopyAddress &address)
    {
        const int64_t pairOffset = static_cast<int64_t>(batchIdx) *
            MTP_SCATTER_UNION_CAPACITY + copyIdx;
        const int32_t srcToken = missSrcIdsGm_.GetValue(pairOffset);
        const int32_t dstSlot = missDstSlotsGm_.GetValue(pairOffset);
        ASSERT_MSG(
            srcToken >= 0 && dstSlot >= 0,
            "active MTP union source/destination must be non-negative.");
        if (srcToken < 0 || dstSlot < 0) {
            return false;
        }

        const int64_t srcBlockCol =
            static_cast<int64_t>(srcToken) >> MTP_SCATTER_BLOCK_SHIFT;
        const int64_t srcBlockOffset =
            static_cast<int64_t>(srcToken) & MTP_SCATTER_BLOCK_MASK;
        const int64_t dstBlockCol =
            static_cast<int64_t>(dstSlot) >> MTP_SCATTER_BLOCK_SHIFT;
        const int64_t dstBlockOffset =
            static_cast<int64_t>(dstSlot) & MTP_SCATTER_BLOCK_MASK;
        const int64_t hbmMaxBlocks =
            tiling_->baseParams.maxBlockNumPerBatch;
        ASSERT_MSG(
            srcBlockCol < tiling_->dramMaxBlockNum &&
                dstBlockCol < hbmMaxBlocks,
            "active MTP union entry exceeds its block table.");
        if (srcBlockCol >= tiling_->dramMaxBlockNum ||
            dstBlockCol >= hbmMaxBlocks) {
            return false;
        }

        const int32_t srcPhysicalBlock =
            dramBlockTableGm_.GetValue(
                batchIdx * tiling_->dramMaxBlockNum + srcBlockCol);
        const int32_t dstPhysicalBlock =
            hbmBlockTableGm_.GetValue(
                batchIdx * hbmMaxBlocks + dstBlockCol);
        ASSERT_MSG(
            srcPhysicalBlock >= 0 && dstPhysicalBlock >= 0,
            "MTP fused block-table entries must be non-negative.");
        if (srcPhysicalBlock < 0 || dstPhysicalBlock < 0) {
            return false;
        }

        address.srcCkv =
            (static_cast<int64_t>(srcPhysicalBlock) *
                 MTP_SCATTER_BLOCK_SIZE +
             srcBlockOffset) * MTP_SCATTER_CKV_DIM;
        address.dstCkv =
            (static_cast<int64_t>(dstPhysicalBlock) *
                 MTP_SCATTER_BLOCK_SIZE +
             dstBlockOffset) * MTP_SCATTER_CKV_DIM;
        address.srcKpe =
            (static_cast<int64_t>(srcPhysicalBlock) *
                 MTP_SCATTER_BLOCK_SIZE +
             srcBlockOffset) * MTP_SCATTER_KPE_DIM;
        address.dstKpe =
            (static_cast<int64_t>(dstPhysicalBlock) *
                 MTP_SCATTER_BLOCK_SIZE +
             dstBlockOffset) * MTP_SCATTER_KPE_DIM;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams ckvParams{
            1, static_cast<uint32_t>(MTP_SCATTER_CKV_BYTES), 0, 0, 0};
        DataCopyExtParams kpeParams{
            1, static_cast<uint32_t>(MTP_SCATTER_KPE_BYTES), 0, 0, 0};
        DataCopyPad(
            local, dramCkvGm_[address.srcCkv], ckvParams, padParams);
        DataCopyPad(
            local[kpeUbOffset_], dramKpeGm_[address.srcKpe],
            kpeParams, padParams);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        DataCopyExtParams ckvParams{
            1, static_cast<uint32_t>(MTP_SCATTER_CKV_BYTES), 0, 0, 0};
        DataCopyExtParams kpeParams{
            1, static_cast<uint32_t>(MTP_SCATTER_KPE_BYTES), 0, 0, 0};
        DataCopyPad(hbmCkvGm_[address.dstCkv], local, ckvParams);
        DataCopyPad(
            hbmKpeGm_[address.dstKpe], local[kpeUbOffset_], kpeParams);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe *pipe_;
    const NanovllmFusedCopySfaMtpTilingData *tiling_;
    int32_t blockIdx_ = -1;
    uint32_t logicalCoreCount_ = 0;
    uint32_t batchSize_ = 0;
    int32_t kpeUbOffset_ = 0;

    GlobalTensor<T> hbmKpeGm_;
    GlobalTensor<T> hbmCkvGm_;
    GlobalTensor<T> dramKpeGm_;
    GlobalTensor<T> dramCkvGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> missSrcIdsGm_;
    GlobalTensor<int32_t> missDstSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};

}  // namespace FusedCopySfaMtpNs

#endif
