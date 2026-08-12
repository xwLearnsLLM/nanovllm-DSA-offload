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

// Copies each unique union miss exactly once.  AIV work is flattened over
// [B, 8192] and striped across both vector sub-cores of every SFA AIC.  The
// source-aware Attention path reads current misses directly from DRAM, so no
// copy->Attention barrier is required; this stage only updates persistent HBM.
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
        totalPairSlots_ = static_cast<int64_t>(batchSize_) *
            MTP_SCATTER_UNION_CAPACITY;
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
        cachedBatchIdx_ = -1;
        cachedMissCount_ = 0;

        int64_t currentPair = FindNextValidPair(blockIdx_);
        CopyAddress currentAddress;
        while (currentPair < totalPairSlots_ &&
               !ResolveAddress(currentPair, currentAddress)) {
            currentPair = FindNextValidPair(
                currentPair + logicalCoreCount_);
        }
        if (currentPair >= totalPairSlots_) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            int64_t nextPair = FindNextValidPair(
                currentPair + logicalCoreCount_);
            CopyAddress nextAddress;
            while (nextPair < totalPairSlots_ &&
                   !ResolveAddress(nextPair, nextAddress)) {
                nextPair = FindNextValidPair(
                    nextPair + logicalCoreCount_);
            }
            const bool hasNext = nextPair < totalPairSlots_;
            if (hasNext) {
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            currentPair = nextPair;
            currentAddress = nextAddress;
        }

        // Publish this AIV's persistent writes before it enters Attention.
        PipeSync<HardEvent::MTE3_S>();
    }

private:
    struct CopyAddress {
        int64_t srcCkv = 0;
        int64_t dstCkv = 0;
        int64_t srcKpe = 0;
        int64_t dstKpe = 0;
    };

    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        const int64_t steps = CeilDiv(
            start - blockIdx_,
            static_cast<int64_t>(logicalCoreCount_));
        return blockIdx_ + steps * logicalCoreCount_;
    }

    __aicore__ inline int64_t FindNextValidPair(int64_t flatPair)
    {
        while (flatPair < totalPairSlots_) {
            const int64_t batchIdx =
                flatPair / MTP_SCATTER_UNION_CAPACITY;
            const int32_t copyIdx = static_cast<int32_t>(
                flatPair - batchIdx * MTP_SCATTER_UNION_CAPACITY);
            if (batchIdx != cachedBatchIdx_) {
                cachedMissCount_ = missCountsGm_.GetValue(batchIdx);
                ASSERT_MSG(
                    cachedMissCount_ >= 0 &&
                        cachedMissCount_ <= MTP_SCATTER_UNION_CAPACITY,
                    "MTP union miss_count exceeds 8192.");
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < cachedMissCount_) {
                return flatPair;
            }
            flatPair = FirstFlatPairAtOrAfter(
                (batchIdx + 1) * MTP_SCATTER_UNION_CAPACITY);
        }
        return totalPairSlots_;
    }

    __aicore__ inline bool ResolveAddress(
        int64_t flatPair, CopyAddress &address)
    {
        const int64_t batchIdx =
            flatPair / MTP_SCATTER_UNION_CAPACITY;
        const int32_t copyIdx = static_cast<int32_t>(
            flatPair - batchIdx * MTP_SCATTER_UNION_CAPACITY);
        const int64_t pairOffset =
            batchIdx * MTP_SCATTER_UNION_CAPACITY + copyIdx;
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

    __aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

private:
    TPipe *pipe_;
    const NanovllmFusedCopySfaMtpTilingData *tiling_;
    int32_t blockIdx_ = -1;
    uint32_t logicalCoreCount_ = 0;
    uint32_t batchSize_ = 0;
    int64_t totalPairSlots_ = 0;
    int32_t kpeUbOffset_ = 0;
    int64_t cachedBatchIdx_ = -1;
    int32_t cachedMissCount_ = 0;

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
