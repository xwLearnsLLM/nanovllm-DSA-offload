#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dsa_index_update_tiling_data.h"

namespace {

constexpr uint16_t DSA_BF16_LOW_SENTINEL_RAW = 0xFF80;
constexpr uint32_t DSA_SORT_KEY_MIN = 0U;
constexpr uint32_t DSA_SORT_KEY_MAX = 0xFFFFFFFFU;

class KernelDsaIndexUpdate {
public:
    __aicore__ inline KernelDsaIndexUpdate() {}

    __aicore__ inline void Init(GM_ADDR score, GM_ADDR hbmCachedTokensPool, GM_ADDR candidateLens,
        GM_ADDR selectedLens, GM_ADDR reqPoolEntries, GM_ADDR promoteIdx, GM_ADDR demoteIdx,
        GM_ADDR copyCounts, const DsaIndexUpdateTilingData* tiling)
    {
        scoreGm_.SetGlobalBuffer((__gm__ uint16_t*)score);
        poolGm_.SetGlobalBuffer((__gm__ int32_t*)hbmCachedTokensPool);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t*)candidateLens);
        selectedLensGm_.SetGlobalBuffer((__gm__ int32_t*)selectedLens);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t*)reqPoolEntries);
        promoteIdxGm_.SetGlobalBuffer((__gm__ int32_t*)promoteIdx);
        demoteIdxGm_.SetGlobalBuffer((__gm__ int32_t*)demoteIdx);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t*)copyCounts);

        batchSize_ = tiling->batchSize;
        maxSeqLen_ = tiling->maxSeqLen;
        maxSelectedLen_ = tiling->maxSelectedLen;
        poolCapacity_ = tiling->poolCapacity;
        maxOutputLen_ = tiling->maxOutputLen;
        k_ = static_cast<int32_t>(tiling->k);
        usedCoreNum_ = tiling->usedCoreNum;
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        for (int64_t batch = blockIdx; batch < batchSize_; batch += usedCoreNum_) {
            ProcessBatch(batch);
        }
    }

private:
    __aicore__ inline int32_t MinInt32(int32_t lhs, int32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline int32_t MaxInt32(int32_t lhs, int32_t rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Bf16RawToSortKey(uint16_t raw)
    {
        if ((raw & 0x7FFFU) == 0U) {
            return 0x8000U;
        }
        if ((raw & 0x8000U) != 0U) {
            return static_cast<uint32_t>(static_cast<uint16_t>(~raw));
        }
        return static_cast<uint32_t>(raw ^ 0x8000U);
    }

    __aicore__ inline int32_t ComputeCopyCount(int32_t candidateLen, int32_t selectedLen)
    {
        if (candidateLen <= 0 || selectedLen <= 0 || k_ <= 0) {
            return 0;
        }
        const int32_t availableUncached = MaxInt32(candidateLen - selectedLen, 0);
        return MinInt32(k_, MinInt32(selectedLen, availableUncached));
    }

    __aicore__ inline bool IsBetterForBottom(
        uint32_t scoreKey, int32_t localIdx, uint32_t refKey, int32_t refIdx)
    {
        return (scoreKey < refKey) || ((scoreKey == refKey) && (localIdx < refIdx));
    }

    __aicore__ inline bool IsBetterForTop(
        uint32_t scoreKey, int32_t globalIdx, uint32_t refKey, int32_t refIdx)
    {
        return (scoreKey > refKey) || ((scoreKey == refKey) && (globalIdx < refIdx));
    }

    __aicore__ inline void InsertBottom(
        uint32_t scoreKey, int32_t localIdx, uint32_t* bottomKeys, int32_t* bottomIdx, int32_t activeK)
    {
        int32_t insertPos = -1;
        for (int32_t pos = 0; pos < activeK; ++pos) {
            if (IsBetterForBottom(scoreKey, localIdx, bottomKeys[pos], bottomIdx[pos])) {
                insertPos = pos;
                break;
            }
        }
        if (insertPos < 0) {
            return;
        }
        for (int32_t pos = activeK - 1; pos > insertPos; --pos) {
            bottomKeys[pos] = bottomKeys[pos - 1];
            bottomIdx[pos] = bottomIdx[pos - 1];
        }
        bottomKeys[insertPos] = scoreKey;
        bottomIdx[insertPos] = localIdx;
    }

    __aicore__ inline void InsertTop(
        uint32_t scoreKey, int32_t globalIdx, uint32_t* topKeys, int32_t* topIdx, int32_t activeK)
    {
        int32_t insertPos = -1;
        for (int32_t pos = 0; pos < activeK; ++pos) {
            if (IsBetterForTop(scoreKey, globalIdx, topKeys[pos], topIdx[pos])) {
                insertPos = pos;
                break;
            }
        }
        if (insertPos < 0) {
            return;
        }
        for (int32_t pos = activeK - 1; pos > insertPos; --pos) {
            topKeys[pos] = topKeys[pos - 1];
            topIdx[pos] = topIdx[pos - 1];
        }
        topKeys[insertPos] = scoreKey;
        topIdx[insertPos] = globalIdx;
    }

    __aicore__ inline void ClearActiveOutputs(int64_t outBase)
    {
        for (int32_t i = 0; i < k_; ++i) {
            promoteIdxGm_.SetValue(outBase + i, 0);
            demoteIdxGm_.SetValue(outBase + i, 0);
        }
    }

    __aicore__ inline void ProcessBatch(int64_t batch)
    {
        const int32_t candidateLen = MinInt32(
            MaxInt32(candidateLensGm_.GetValue(batch), 0),
            static_cast<int32_t>(maxSeqLen_));
        const int32_t selectedLen = MinInt32(
            MaxInt32(selectedLensGm_.GetValue(batch), 0),
            static_cast<int32_t>(maxSelectedLen_));
        const int32_t poolEntry = reqPoolEntriesGm_.GetValue(batch);
        const int64_t scoreBase = batch * maxSeqLen_;
        const int64_t outBase = batch * maxOutputLen_;

        ClearActiveOutputs(outBase);

        if (poolEntry < 0 || poolEntry >= poolCapacity_) {
            copyCountsGm_.SetValue(batch, 0);
            return;
        }
        const int32_t copyCount = ComputeCopyCount(candidateLen, selectedLen);
        copyCountsGm_.SetValue(batch, copyCount);
        if (copyCount <= 0) {
            return;
        }

        const int64_t poolBase = static_cast<int64_t>(poolEntry) * maxSelectedLen_;
        uint32_t bottomKeys[DSA_INDEX_UPDATE_MAX_K];
        int32_t bottomIdx[DSA_INDEX_UPDATE_MAX_K];
        uint32_t topKeys[DSA_INDEX_UPDATE_MAX_K];
        int32_t topIdx[DSA_INDEX_UPDATE_MAX_K];

        for (int32_t i = 0; i < copyCount; ++i) {
            bottomKeys[i] = DSA_SORT_KEY_MAX;
            bottomIdx[i] = 0;
            topKeys[i] = DSA_SORT_KEY_MIN;
            topIdx[i] = 0;
        }

        for (int32_t localIdx = 0; localIdx < selectedLen; ++localIdx) {
            const int32_t globalIdx = poolGm_.GetValue(poolBase + localIdx);
            if (globalIdx < 0 || globalIdx >= candidateLen) {
                continue;
            }
            const uint32_t scoreKey = Bf16RawToSortKey(scoreGm_.GetValue(scoreBase + globalIdx));
            InsertBottom(scoreKey, localIdx, bottomKeys, bottomIdx, copyCount);
            scoreGm_.SetValue(scoreBase + globalIdx, DSA_BF16_LOW_SENTINEL_RAW);
        }

        for (int32_t globalIdx = 0; globalIdx < candidateLen; ++globalIdx) {
            const uint32_t scoreKey = Bf16RawToSortKey(scoreGm_.GetValue(scoreBase + globalIdx));
            InsertTop(scoreKey, globalIdx, topKeys, topIdx, copyCount);
        }

        for (int32_t i = 0; i < copyCount; ++i) {
            const int32_t demoteSlot = bottomIdx[i];
            const int32_t promoteToken = topIdx[i];
            demoteIdxGm_.SetValue(outBase + i, demoteSlot);
            promoteIdxGm_.SetValue(outBase + i, promoteToken);
            poolGm_.SetValue(poolBase + demoteSlot, promoteToken);
        }
    }

private:
    AscendC::GlobalTensor<uint16_t> scoreGm_;
    AscendC::GlobalTensor<int32_t> poolGm_;
    AscendC::GlobalTensor<int32_t> candidateLensGm_;
    AscendC::GlobalTensor<int32_t> selectedLensGm_;
    AscendC::GlobalTensor<int32_t> reqPoolEntriesGm_;
    AscendC::GlobalTensor<int32_t> promoteIdxGm_;
    AscendC::GlobalTensor<int32_t> demoteIdxGm_;
    AscendC::GlobalTensor<int32_t> copyCountsGm_;

    int64_t batchSize_ = 0;
    int64_t maxSeqLen_ = 0;
    int64_t maxSelectedLen_ = 0;
    int64_t poolCapacity_ = 0;
    int64_t maxOutputLen_ = 0;
    int32_t k_ = 0;
    int64_t usedCoreNum_ = 1;
};

} // namespace

extern "C" __global__ __aicore__ void dsa_index_update(GM_ADDR score, GM_ADDR hbmCachedTokensPool,
    GM_ADDR candidateLens, GM_ADDR selectedLens, GM_ADDR reqPoolEntries,
    GM_ADDR promoteIdx, GM_ADDR demoteIdx, GM_ADDR copyCounts,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    REGISTER_TILING_DEFAULT(DsaIndexUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(DsaIndexUpdateTilingData, tilingData, tiling);
    KernelDsaIndexUpdate op;
    op.Init(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
        promoteIdx, demoteIdx, copyCounts, &tilingData);
    op.Process();
}
