/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef KVCACHE_OFFLOAD_COPY_KERNEL_H
#define KVCACHE_OFFLOAD_COPY_KERNEL_H

#include "kernel_operator.h"

namespace KvcacheOffloadCopyNs {
using namespace AscendC;

class KvcacheOffloadCopyKernel {
public:
    __aicore__ inline KvcacheOffloadCopyKernel(
        TPipe* pipe, const KvcacheOffloadCopyTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKvCache, GM_ADDR dramKvCache, GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable, GM_ADDR copyCounts)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        // Two 32-KiB slots let MTE2 prefetch a source tile while MTE3 writes
        // the preceding tile to swapped-memory DRAM.
        pipe_->InitBuffer(copyQueue_, 2, static_cast<uint32_t>(tiling_->tileBytes));

        hbmKvCacheGm_.SetGlobalBuffer((__gm__ int8_t*)hbmKvCache);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ int8_t*)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)dramBlockTable);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t*)copyCounts);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        cachedBatchIdx_ = -1;
        cachedCopyCount_ = 0;

        int64_t flatPair = FindNextValidPair(blockIdx_);
        while (flatPair < tiling_->totalPairSlots) {
            CopyAddress address;
            if (ResolveAddress(flatPair, address)) {
                CopyBlock(address);
            }
            flatPair = FindNextValidPair(flatPair + tiling_->usedCoreNum);
        }
    }

private:
    struct CopyAddress {
        int64_t srcBase = 0;
        int64_t dstBase = 0;
    };

    __aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        int64_t steps = CeilDiv(start - blockIdx_, static_cast<int64_t>(tiling_->usedCoreNum));
        return blockIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline int64_t FindNextValidPair(int64_t flatPairIdx)
    {
        while (flatPairIdx < tiling_->totalPairSlots) {
            int64_t batchIdx = flatPairIdx / tiling_->copyCap;
            int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * tiling_->copyCap);
            if (batchIdx != cachedBatchIdx_) {
                cachedCopyCount_ = copyCountsGm_.GetValue(batchIdx);
                ASSERT_MSG(cachedCopyCount_ >= 0 && cachedCopyCount_ <= tiling_->copyCap,
                    "copy_count exceeds the offload block-table capacity.");
                if (cachedCopyCount_ < 0) {
                    cachedCopyCount_ = 0;
                } else if (cachedCopyCount_ > tiling_->copyCap) {
                    cachedCopyCount_ = static_cast<int32_t>(tiling_->copyCap);
                }
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < cachedCopyCount_) {
                return flatPairIdx;
            }
            // Skip the inactive row suffix, but preserve this core's cyclic
            // ownership in the following batch.
            flatPairIdx = FirstFlatPairAtOrAfter((batchIdx + 1) * tiling_->copyCap);
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool ResolveAddress(int64_t flatPairIdx, CopyAddress& address)
    {
        int32_t srcPhysicalBlock = hbmBlockTableGm_.GetValue(flatPairIdx);
        int32_t dstPhysicalBlock = dramBlockTableGm_.GetValue(flatPairIdx);
        ASSERT_MSG(srcPhysicalBlock >= 0 && srcPhysicalBlock < tiling_->hbmBlockCount,
            "active HBM physical block ID is out of range.");
        ASSERT_MSG(dstPhysicalBlock >= 0 && dstPhysicalBlock < tiling_->dramBlockCount,
            "active DRAM physical block ID is out of range.");
        if (srcPhysicalBlock < 0 || srcPhysicalBlock >= tiling_->hbmBlockCount ||
            dstPhysicalBlock < 0 || dstPhysicalBlock >= tiling_->dramBlockCount) {
            return false;
        }

        address.srcBase = static_cast<int64_t>(srcPhysicalBlock) * tiling_->blockBytes;
        address.dstBase = static_cast<int64_t>(dstPhysicalBlock) * tiling_->blockBytes;
        return true;
    }

    __aicore__ inline uint32_t GetTileBytes(int64_t blockOffset)
    {
        int64_t remaining = tiling_->blockBytes - blockOffset;
        int64_t bytes = remaining < tiling_->tileBytes ? remaining : tiling_->tileBytes;
        return static_cast<uint32_t>(bytes);
    }

    __aicore__ inline void CopyIn(int64_t srcOffset, uint32_t bytes)
    {
        LocalTensor<int8_t> local = copyQueue_.AllocTensor<int8_t>();
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyExtParams params{1, bytes, 0, 0, 0};
        DataCopyPad(local, hbmKvCacheGm_[srcOffset], params, padParams);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(int64_t dstOffset, uint32_t bytes)
    {
        LocalTensor<int8_t> local = copyQueue_.DeQue<int8_t>();
        DataCopyExtParams params{1, bytes, 0, 0, 0};
        DataCopyPad(dramKvCacheGm_[dstOffset], local, params);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyBlock(const CopyAddress& address)
    {
        int64_t blockOffset = 0;
        uint32_t currentBytes = GetTileBytes(blockOffset);
        CopyIn(address.srcBase, currentBytes);

        while (true) {
            int64_t nextOffset = blockOffset + currentBytes;
            bool hasNext = nextOffset < tiling_->blockBytes;
            uint32_t nextBytes = 0;
            if (hasNext) {
                nextBytes = GetTileBytes(nextOffset);
                CopyIn(address.srcBase + nextOffset, nextBytes);
            }

            CopyOut(address.dstBase + blockOffset, currentBytes);
            if (!hasNext) {
                break;
            }
            blockOffset = nextOffset;
            currentBytes = nextBytes;
        }
    }

private:
    TPipe* pipe_;
    const KvcacheOffloadCopyTilingData* tiling_;
    int32_t blockIdx_ = -1;
    int64_t cachedBatchIdx_ = -1;
    int32_t cachedCopyCount_ = 0;

    GlobalTensor<int8_t> hbmKvCacheGm_;
    GlobalTensor<int8_t> dramKvCacheGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};

} // namespace KvcacheOffloadCopyNs
#endif // KVCACHE_OFFLOAD_COPY_KERNEL_H
