/**
 * Request-level union/update stage for the Ascend 950 C8 MTP path.
 *
 * The official Quant LightningIndexer has already produced one top-2048 row
 * per verification query. One AIV worker owns one request: it deduplicates
 * the 2--4 rows, protects every union hit, selects safe victims, updates the
 * request-pool row once, and finally publishes every query's complete HBM
 * slot row.
 */

#include "kernel_operator.h"
#include "a5_fused_li_manage_mtp_c8_cache_update_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = 4;
constexpr uint32_t MIN_QUERIES_PER_REQUEST = 2;
constexpr uint32_t UNION_CAPACITY =
    SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr uint32_t UNION_HASH_CAPACITY = 16384;
constexpr uint32_t UNION_HASH_MASK = UNION_HASH_CAPACITY - 1;
constexpr uint32_t MAX_CACHE_TOKENS = 16256;
constexpr uint32_t CACHE_CHUNK = 2048;
constexpr uint32_t TOKEN_MASK = (1U << 18) - 1U;
constexpr uint32_t SLOT_SHIFT = 18;

class A5FusedLiManageMtpC8CacheUpdateKernel {
public:
    __aicore__ inline A5FusedLiManageMtpC8CacheUpdateKernel(
        TPipe *pipe,
        const A5FusedLiManageMtpC8CacheUpdateTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR topkIndices,
        GM_ADDR actualSeqLengthsQuery,
        GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR topkDestinationSlots,
        GM_ADDR missSourceIds,
        GM_ADDR missDestinationSlots,
        GM_ADDR missCounts)
    {
        coreIdx_ = GetBlockIdx();
        topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsPoolGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        topkDestinationSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkDestinationSlots);
        missSourceIdsGm_.SetGlobalBuffer((__gm__ int32_t *)missSourceIds);
        missDestinationSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)missDestinationSlots);
        missCountsGm_.SetGlobalBuffer((__gm__ int32_t *)missCounts);

        pipe_->InitBuffer(
            unionHashBuf_, UNION_HASH_CAPACITY * sizeof(int32_t));
        pipe_->InitBuffer(
            missTokensBuf_, UNION_CAPACITY * sizeof(int32_t));
        pipe_->InitBuffer(
            victimPayloadsBuf_, UNION_CAPACITY * sizeof(uint32_t));
        pipe_->InitBuffer(
            topkTokensBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(
            topkSlotsBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(
            cacheChunkBuf_, CACHE_CHUNK * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            ProcessRequest(batch);
        }
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline bool InsertUnion(
        LocalTensor<int32_t> unionHash, uint32_t token) const
    {
        uint32_t hashPos = (token * 2654435761U) & UNION_HASH_MASK;
        for (uint32_t probe = 0; probe < UNION_HASH_CAPACITY; ++probe) {
            const int32_t stored = unionHash.GetValue(hashPos);
            if (stored == static_cast<int32_t>(token)) {
                return false;
            }
            if (stored < 0) {
                unionHash.SetValue(hashPos, static_cast<int32_t>(token));
                return true;
            }
            hashPos = (hashPos + 1U) & UNION_HASH_MASK;
        }
        return false;
    }

    __aicore__ inline bool ContainsUnion(
        LocalTensor<int32_t> unionHash, uint32_t token) const
    {
        uint32_t hashPos = (token * 2654435761U) & UNION_HASH_MASK;
        for (uint32_t probe = 0; probe < UNION_HASH_CAPACITY; ++probe) {
            const int32_t stored = unionHash.GetValue(hashPos);
            if (stored == static_cast<int32_t>(token)) {
                return true;
            }
            if (stored < 0) {
                return false;
            }
            hashPos = (hashPos + 1U) & UNION_HASH_MASK;
        }
        return false;
    }

    __aicore__ inline void LoadTopk(
        uint32_t queryRow, LocalTensor<int32_t> topkTokens)
    {
        DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            topkTokens,
            topkIndicesGm_[
                static_cast<uint64_t>(queryRow) * SPARSE_COUNT],
            copy,
            pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void StoreTopkSlots(
        uint32_t queryRow, LocalTensor<int32_t> topkSlots)
    {
        DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            topkDestinationSlotsGm_[
                static_cast<uint64_t>(queryRow) * SPARSE_COUNT],
            topkSlots,
            copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }

    __aicore__ inline void StoreMissCount(
        uint32_t batch, int32_t value, LocalTensor<int32_t> scratch)
    {
        scratch.SetValue(0, value);
        DataCopyParams copy{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        DataCopyPad(missCountsGm_[batch], scratch, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
    }

    __aicore__ inline void PublishInvalidRows(
        uint32_t queryBegin,
        uint32_t queryEnd,
        LocalTensor<int32_t> topkSlots)
    {
        Duplicate(topkSlots, static_cast<int32_t>(-1), SPARSE_COUNT);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID3);
        WaitFlag<HardEvent::V_S>(EVENT_ID3);
        for (uint32_t queryRow = queryBegin;
             queryRow < queryEnd; ++queryRow) {
            StoreTopkSlots(queryRow, topkSlots);
        }
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        LocalTensor<int32_t> unionHash = unionHashBuf_.Get<int32_t>();
        LocalTensor<int32_t> missTokens = missTokensBuf_.Get<int32_t>();
        LocalTensor<uint32_t> victimPayloads =
            victimPayloadsBuf_.Get<uint32_t>();
        LocalTensor<int32_t> topkTokens = topkTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> topkSlots = topkSlotsBuf_.Get<int32_t>();
        LocalTensor<int32_t> cacheChunk = cacheChunkBuf_.Get<int32_t>();

        const int32_t queryEndValue =
            actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t queryBeginValue = batch == 0
            ? 0
            : actualSeqLengthsQueryGm_.GetValue(batch - 1);
        uint32_t safeQueryBegin = queryBeginValue < 0
            ? 0U
            : static_cast<uint32_t>(queryBeginValue);
        uint32_t safeQueryEnd = queryEndValue < queryBeginValue
            ? safeQueryBegin
            : static_cast<uint32_t>(queryEndValue);
        safeQueryBegin = MinU32(safeQueryBegin, tiling_->packedQueryCount);
        safeQueryEnd = MinU32(safeQueryEnd, tiling_->packedQueryCount);
        const int32_t queryCountValue = queryEndValue - queryBeginValue;
        const int32_t budgetValue = cacheTokensGm_.GetValue(batch);
        const int32_t candidateValue = candidateLensGm_.GetValue(batch);
        const int32_t poolRowValue = reqPoolEntriesGm_.GetValue(batch);

        if (queryBeginValue < 0 || queryEndValue < queryBeginValue ||
            queryEndValue > static_cast<int32_t>(tiling_->packedQueryCount) ||
            queryCountValue < static_cast<int32_t>(MIN_QUERIES_PER_REQUEST) ||
            queryCountValue > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
            PublishInvalidRows(safeQueryBegin, safeQueryEnd, topkSlots);
            StoreMissCount(batch, -1, topkTokens);
            return;
        }
        const uint32_t queryBegin = static_cast<uint32_t>(queryBeginValue);
        const uint32_t queryEnd = static_cast<uint32_t>(queryEndValue);
        if (budgetValue == 0) {
            PublishInvalidRows(queryBegin, queryEnd, topkSlots);
            StoreMissCount(batch, 0, topkTokens);
            return;
        }
        if (budgetValue < static_cast<int32_t>(SPARSE_COUNT) ||
            budgetValue > static_cast<int32_t>(MAX_CACHE_TOKENS) ||
            candidateValue < static_cast<int32_t>(SPARSE_COUNT) ||
            candidateValue > static_cast<int32_t>(tiling_->sourceCapacity) ||
            poolRowValue < 0 ||
            poolRowValue >= static_cast<int32_t>(tiling_->poolSize)) {
            PublishInvalidRows(queryBegin, queryEnd, topkSlots);
            StoreMissCount(batch, -1, topkTokens);
            return;
        }

        const uint32_t budget = static_cast<uint32_t>(budgetValue);
        const uint32_t candidateLen = static_cast<uint32_t>(candidateValue);
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRowValue) * tiling_->sourceCapacity;
        Duplicate(
            unionHash, static_cast<int32_t>(-1), UNION_HASH_CAPACITY);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID3);
        WaitFlag<HardEvent::V_S>(EVENT_ID3);

        uint32_t unionCount = 0;
        uint32_t missCount = 0;
        bool validTopk = true;
        for (uint32_t queryRow = queryBegin;
             queryRow < queryEnd; ++queryRow) {
            LoadTopk(queryRow, topkTokens);
            for (uint32_t index = 0; index < SPARSE_COUNT; ++index) {
                const int32_t tokenValue = topkTokens.GetValue(index);
                if (tokenValue < 0 || tokenValue >= candidateValue) {
                    validTopk = false;
                    continue;
                }
                const uint32_t token = static_cast<uint32_t>(tokenValue);
                if (!InsertUnion(unionHash, token)) {
                    continue;
                }
                ++unionCount;
                if (cacheSlotsPoolGm_.GetValue(cacheBase + token) < 0) {
                    if (missCount >= UNION_CAPACITY) {
                        validTopk = false;
                        continue;
                    }
                    missTokens.SetValue(missCount++, tokenValue);
                }
            }
        }
        if (!validTopk || unionCount > budget ||
            missCount > UNION_CAPACITY) {
            PublishInvalidRows(queryBegin, queryEnd, topkSlots);
            StoreMissCount(batch, -1, topkTokens);
            return;
        }

        uint32_t victimCount = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < candidateLen && victimCount < missCount;
             chunkBase += CACHE_CHUNK) {
            const uint32_t chunkLen = MinU32(
                CACHE_CHUNK, candidateLen - chunkBase);
            DataCopyExtParams copy{
                1, chunkLen * sizeof(int32_t), 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                cacheChunk,
                cacheSlotsPoolGm_[cacheBase + chunkBase],
                copy,
                pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t offset = 0;
                 offset < chunkLen && victimCount < missCount; ++offset) {
                const int32_t slotValue = cacheChunk.GetValue(offset);
                if (slotValue < 0 || slotValue >= budgetValue) {
                    continue;
                }
                const uint32_t token = chunkBase + offset;
                if (ContainsUnion(unionHash, token)) {
                    continue;
                }
                const uint32_t payload =
                    (static_cast<uint32_t>(slotValue) << SLOT_SHIFT) |
                    (token & TOKEN_MASK);
                victimPayloads.SetValue(victimCount++, payload);
            }
        }
        if (victimCount != missCount) {
            PublishInvalidRows(queryBegin, queryEnd, topkSlots);
            StoreMissCount(batch, -1, topkTokens);
            return;
        }

        for (uint32_t index = 0; index < missCount; ++index) {
            const uint32_t payload = victimPayloads.GetValue(index);
            const uint32_t victimToken = payload & TOKEN_MASK;
            const uint32_t slot = payload >> SLOT_SHIFT;
            const uint32_t missToken = static_cast<uint32_t>(
                missTokens.GetValue(index));
            cacheSlotsPoolGm_.SetValue(cacheBase + victimToken, -1);
            cacheSlotsPoolGm_.SetValue(
                cacheBase + missToken, static_cast<int32_t>(slot));
            victimPayloads.SetValue(index, slot);
        }
        PipeBarrier<PIPE_ALL>();

        if (missCount > 0) {
            DataCopyExtParams copy{
                1, missCount * sizeof(int32_t), 0, 0, 0};
            const uint64_t missBase =
                static_cast<uint64_t>(batch) * UNION_CAPACITY;
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad<int32_t, PaddingMode::Normal>(
                missSourceIdsGm_[missBase], missTokens, copy);
            DataCopyPad<int32_t, PaddingMode::Normal>(
                missDestinationSlotsGm_[missBase],
                victimPayloads.ReinterpretCast<int32_t>(),
                copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }

        for (uint32_t queryRow = queryBegin;
             queryRow < queryEnd; ++queryRow) {
            LoadTopk(queryRow, topkTokens);
            for (uint32_t index = 0; index < SPARSE_COUNT; ++index) {
                const int32_t tokenValue = topkTokens.GetValue(index);
                const int32_t slot = tokenValue < 0
                    ? -1
                    : cacheSlotsPoolGm_.GetValue(
                          cacheBase + static_cast<uint32_t>(tokenValue));
                topkSlots.SetValue(index, slot);
            }
            StoreTopkSlots(queryRow, topkSlots);
        }
        StoreMissCount(
            batch, static_cast<int32_t>(missCount), topkTokens);
    }

private:
    TPipe *pipe_;
    const A5FusedLiManageMtpC8CacheUpdateTilingData *tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsPoolGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> topkDestinationSlotsGm_;
    GlobalTensor<int32_t> missSourceIdsGm_;
    GlobalTensor<int32_t> missDestinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    TBuf<TPosition::VECCALC> unionHashBuf_;
    TBuf<TPosition::VECCALC> missTokensBuf_;
    TBuf<TPosition::VECCALC> victimPayloadsBuf_;
    TBuf<TPosition::VECCALC> topkTokensBuf_;
    TBuf<TPosition::VECCALC> topkSlotsBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
};
}  // namespace

extern "C" __global__ __aicore__ void
a5_fused_li_manage_mtp_c8_cache_update(
    GM_ADDR topkIndices,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR topkDestinationSlots,
    GM_ADDR missSourceIds,
    GM_ADDR missDestinationSlots,
    GM_ADDR missCounts,
    GM_ADDR cacheSlotsAlias,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)cacheSlotsAlias;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(
        A5FusedLiManageMtpC8CacheUpdateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5FusedLiManageMtpC8CacheUpdateKernel op(&pipe, &tilingData);
    op.Init(
        topkIndices,
        actualSeqLengthsQuery,
        reqPoolEntries,
        cacheSlotsPool,
        cacheTokens,
        candidateLens,
        topkDestinationSlots,
        missSourceIds,
        missDestinationSlots,
        missCounts);
    op.Process();
}
