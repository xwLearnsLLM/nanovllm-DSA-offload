#ifndef NANOVLLM_FUSED_COPY_SFA_MTP_SCATTER_STAGE_H
#define NANOVLLM_FUSED_COPY_SFA_MTP_SCATTER_STAGE_H

#include "kernel_operator.h"

namespace FusedCopySfaMtpNs {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BLOCK_SHIFT = 7;
constexpr int64_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;
constexpr int64_t K_ROPE_UB_BYTES = K_ROPE_DIM * sizeof(uint16_t);
constexpr int64_t KV_CACHE_UB_BYTES = KV_CACHE_DIM * sizeof(uint16_t);

// AIV pre-stage for the MTP union miss list.  When B equals the physical AIC
// count and request copy loads are sufficiently balanced, one AIC plus its
// two AIV sub-cores own one request and Attention can begin per request.
// Skewed loads and all other batch shapes use a globally balanced fallback.
template <typename T>
class FusedMtpScatterStage {
public:
    __aicore__ inline FusedMtpScatterStage(
        TPipe *pipe,
        const NanovllmFusedCopySfaMtpTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR hbmKRoPE,
        GM_ADDR hbmKvCache,
        GM_ADDR dramKRoPE,
        GM_ADDR dramKvCache,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR srcTokenIds,
        GM_ADDR dstSlots,
        GM_ADDR copyCounts)
    {
        blockIdx_ = GetBlockIdx();
        physicalCoreCount_ = tiling_->singleCoreParams.usedCoreNum;
        logicalCoreCount_ = physicalCoreCount_ * 2U;
        batchSize_ = tiling_->baseParams.batchSize;
        copyCap_ = tiling_->copyCap;
        totalPairSlots_ = static_cast<int64_t>(batchSize_) * copyCap_;
        ownerSchedule_ = false;
        kRopeUbOffset_ = KV_CACHE_UB_BYTES / sizeof(T);

        pipe_->InitBuffer(
            copyQueue_, 2, KV_CACHE_UB_BYTES + K_ROPE_UB_BYTES);

        hbmKRoPEGm_.SetGlobalBuffer((__gm__ T *)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ T *)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ T *)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ T *)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        srcTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t *)srcTokenIds);
        dstSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)dstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= logicalCoreCount_) {
            return;
        }
        cachedBatchIdx_ = -1;
        cachedCopyCount_ = 0;
        ownerSchedule_ = CanUseOwnerSchedule();

        int64_t currentPair = FirstPair();
        CopyAddress currentAddress;
        while (currentPair < totalPairSlots_ &&
               !ResolveAddress(currentPair, currentAddress)) {
            currentPair = NextPair(currentPair);
        }
        if (currentPair >= totalPairSlots_) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            int64_t nextPair = NextPair(currentPair);
            CopyAddress nextAddress;
            while (nextPair < totalPairSlots_ &&
                   !ResolveAddress(nextPair, nextAddress)) {
                nextPair = NextPair(nextPair);
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
    }

    __aicore__ inline bool UsesOwnerSchedule() const
    {
        return ownerSchedule_;
    }

private:
    struct CopyAddress {
        int64_t srcKv = 0;
        int64_t dstKv = 0;
        int64_t srcRope = 0;
        int64_t dstRope = 0;
    };

    __aicore__ inline bool CanUseOwnerSchedule()
    {
        if (batchSize_ != physicalCoreCount_) {
            return false;
        }
        int64_t total = 0;
        int32_t maximum = 0;
        for (uint32_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            const int32_t count = copyCountsGm_.GetValue(batchIdx);
            ASSERT_MSG(count >= 0 && count <= copyCap_,
                       "miss_count exceeds the MTP union capacity.");
            total += count;
            maximum = count > maximum ? count : maximum;
        }
        if (total == 0) {
            return true;
        }
        // Owner scheduling is used only when its slowest request carries no
        // more than 1.25x the mean work.  Otherwise the flat schedule keeps
        // all 48 AIV lanes balanced.
        return static_cast<int64_t>(maximum) * batchSize_ * 4 <= total * 5;
    }

    __aicore__ inline int64_t FirstPair()
    {
        if (ownerSchedule_) {
            const int64_t batchIdx = blockIdx_ >> 1;
            const int64_t lane = blockIdx_ & 1;
            const int32_t count = copyCountsGm_.GetValue(batchIdx);
            ASSERT_MSG(count >= 0 && count <= copyCap_,
                       "miss_count exceeds the MTP union capacity.");
            if (lane >= count) {
                return totalPairSlots_;
            }
            return batchIdx * copyCap_ + lane;
        }
        return FindNextFlatPair(blockIdx_);
    }

    __aicore__ inline int64_t NextPair(int64_t current)
    {
        if (ownerSchedule_) {
            const int64_t batchIdx = blockIdx_ >> 1;
            const int64_t next = current + 2;
            const int32_t count = copyCountsGm_.GetValue(batchIdx);
            return next < batchIdx * copyCap_ + count
                       ? next
                       : totalPairSlots_;
        }
        return FindNextFlatPair(current + logicalCoreCount_);
    }

    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        const int64_t steps =
            CeilDiv(start - blockIdx_, static_cast<int64_t>(logicalCoreCount_));
        return blockIdx_ + steps * logicalCoreCount_;
    }

    __aicore__ inline int64_t FindNextFlatPair(int64_t flatPair)
    {
        while (flatPair < totalPairSlots_) {
            const int64_t batchIdx = flatPair / copyCap_;
            const int32_t copyIdx =
                static_cast<int32_t>(flatPair - batchIdx * copyCap_);
            if (batchIdx != cachedBatchIdx_) {
                cachedCopyCount_ = copyCountsGm_.GetValue(batchIdx);
                ASSERT_MSG(cachedCopyCount_ >= 0 && cachedCopyCount_ <= copyCap_,
                           "miss_count exceeds the MTP union capacity.");
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < cachedCopyCount_) {
                return flatPair;
            }
            flatPair = FirstFlatPairAtOrAfter((batchIdx + 1) * copyCap_);
        }
        return totalPairSlots_;
    }

    __aicore__ inline bool ResolveAddress(
        int64_t flatPair,
        CopyAddress &address)
    {
        const int64_t batchIdx = flatPair / copyCap_;
        const int32_t copyIdx =
            static_cast<int32_t>(flatPair - batchIdx * copyCap_);
        const int64_t pairOffset = batchIdx * copyCap_ + copyIdx;
        const int32_t srcTokenId = srcTokenIdsGm_.GetValue(pairOffset);
        const int32_t dstSlot = dstSlotsGm_.GetValue(pairOffset);
        ASSERT_MSG(srcTokenId >= 0 && dstSlot >= 0,
                   "active MTP union source/destination IDs must be non-negative.");
        if (srcTokenId < 0 || dstSlot < 0) {
            return false;
        }

        const int64_t srcBlockCol =
            static_cast<int64_t>(srcTokenId) >> BLOCK_SHIFT;
        const int64_t srcBlockOffset =
            static_cast<int64_t>(srcTokenId) & BLOCK_MASK;
        const int64_t dstBlockCol =
            static_cast<int64_t>(dstSlot) >> BLOCK_SHIFT;
        const int64_t dstBlockOffset =
            static_cast<int64_t>(dstSlot) & BLOCK_MASK;
        const int64_t hbmMaxBlocks =
            tiling_->baseParams.maxBlockNumPerBatch;
        ASSERT_MSG(srcBlockCol < tiling_->dramMaxBlockNum &&
                       dstBlockCol < hbmMaxBlocks,
                   "active source token or destination slot exceeds its block table.");
        if (srcBlockCol >= tiling_->dramMaxBlockNum ||
            dstBlockCol >= hbmMaxBlocks) {
            return false;
        }

        const int32_t srcPhysicalBlock = dramBlockTableGm_.GetValue(
            batchIdx * tiling_->dramMaxBlockNum + srcBlockCol);
        const int32_t dstPhysicalBlock = hbmBlockTableGm_.GetValue(
            batchIdx * hbmMaxBlocks + dstBlockCol);
        ASSERT_MSG(srcPhysicalBlock >= 0 && dstPhysicalBlock >= 0,
                   "MTP fused block-table entries must be non-negative.");
        if (srcPhysicalBlock < 0 || dstPhysicalBlock < 0) {
            return false;
        }

        address.srcKv =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE +
             srcBlockOffset) * KV_CACHE_DIM;
        address.dstKv =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE +
             dstBlockOffset) * KV_CACHE_DIM;
        address.srcRope =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE +
             srcBlockOffset) * K_ROPE_DIM;
        address.dstRope =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE +
             dstBlockOffset) * K_ROPE_DIM;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{
            1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{
            1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};
        DataCopyPad(local, dramKvCacheGm_[address.srcKv],
                    kvParams, padParams);
        DataCopyPad(local[kRopeUbOffset_], dramKRoPEGm_[address.srcRope],
                    ropeParams, padParams);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        DataCopyExtParams kvParams{
            1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{
            1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};
        DataCopyPad(hbmKvCacheGm_[address.dstKv], local, kvParams);
        DataCopyPad(hbmKRoPEGm_[address.dstRope],
                    local[kRopeUbOffset_], ropeParams);
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
    uint32_t physicalCoreCount_ = 0;
    uint32_t logicalCoreCount_ = 0;
    uint32_t batchSize_ = 0;
    int64_t copyCap_ = 0;
    int64_t totalPairSlots_ = 0;
    bool ownerSchedule_ = false;
    int32_t kRopeUbOffset_ = 0;
    int64_t cachedBatchIdx_ = -1;
    int32_t cachedCopyCount_ = 0;

    GlobalTensor<T> hbmKRoPEGm_;
    GlobalTensor<T> hbmKvCacheGm_;
    GlobalTensor<T> dramKRoPEGm_;
    GlobalTensor<T> dramKvCacheGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> srcTokenIdsGm_;
    GlobalTensor<int32_t> dstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};

}  // namespace FusedCopySfaMtpNs

#endif
