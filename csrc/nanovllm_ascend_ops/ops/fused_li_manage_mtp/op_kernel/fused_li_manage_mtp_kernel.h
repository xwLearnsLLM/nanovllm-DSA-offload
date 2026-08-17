/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_KERNEL_H_
#define NANOVLLM_FUSED_LI_MANAGE_MTP_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_li_manage_common.h"
#include "fused_li_manage_service_vector.h"
#include "fused_li_manage_service_cube.h"

namespace LIMtpKernel {
using namespace AscendC;
using namespace LICommon;
using LIKernel::LIMatmul;
using LIKernel::LIVector;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename LIT>
class LIMtpPreload {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using MM1_OUT_T = float;

    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key,
        __gm__ uint8_t *weights, __gm__ uint8_t *reqPoolEntries,
        __gm__ uint8_t *cacheState, __gm__ uint8_t *cacheSlots,
        __gm__ uint8_t *actualQueryLens, __gm__ uint8_t *actualKeyLens,
        __gm__ uint8_t *offloadKeyLens, __gm__ uint8_t *reqValid,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *topkSlots, __gm__ uint8_t *topkSourceIds,
        __gm__ uint8_t *topkMissCounts,
        __gm__ uint8_t *missSourceIds,
        __gm__ uint8_t *missDestinationSlots, __gm__ uint8_t *missCounts,
        __gm__ uint8_t *workspace,
        const LIUMtpTilingData *__restrict tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t MIN_SOURCE_TOKENS = 2048;
    static constexpr uint32_t TOPK_TOKENS = 2048;
    static constexpr uint32_t MAX_UNION_TOKENS = 8192;

    LIMatmul<LIT> matmulService;
    LIVector<LIT> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheStateGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> actualQueryLensGm;
    GlobalTensor<int32_t> actualKeyLensGm;
    GlobalTensor<int32_t> offloadKeyLensGm;
    GlobalTensor<int32_t> reqValidGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> topkSourceIdsGm;
    GlobalTensor<int32_t> topkMissCountsGm;
    GlobalTensor<int32_t> missSourceIdsGm;
    GlobalTensor<int32_t> missDestinationSlotsGm;
    GlobalTensor<int32_t> missCountsGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> aggregateScoresGm;
    GlobalTensor<int32_t> internalTopkPayloadsGm;
    GlobalTensor<float> internalThresholdsGm;

    uint32_t tmpBlockIdx = 0;
    uint32_t aiCoreIdx = 0;
    uint32_t requestStart = 0;
    uint32_t requestCount = 0;
    ConstInfo constInfo{};

    __aicore__ inline void InitRequestRange(uint32_t requestedCoreNum);
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessChunk(const RunInfo &runInfo);
    __aicore__ inline void CleanRequest(uint32_t bIdx);
    __aicore__ inline bool IsQueryLayoutValid();
};

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::InitRequestRange(uint32_t requestedCoreNum)
{
    uint32_t activeCoreNum = Min(requestedCoreNum,
                                 static_cast<uint32_t>(constInfo.batchSize));
    if (activeCoreNum == 0U || aiCoreIdx >= activeCoreNum) {
        requestCount = 0U;
        return;
    }
    uint32_t requestsPerCore =
        static_cast<uint32_t>(constInfo.batchSize) / activeCoreNum;
    uint32_t extraRequestCores =
        static_cast<uint32_t>(constInfo.batchSize) % activeCoreNum;
    requestStart = aiCoreIdx * requestsPerCore +
                   Min(aiCoreIdx, extraRequestCores);
    requestCount = requestsPerCore +
                   (aiCoreIdx < extraRequestCores ? 1U : 0U);
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheState,
    __gm__ uint8_t *cacheSlots, __gm__ uint8_t *actualQueryLens,
    __gm__ uint8_t *actualKeyLens, __gm__ uint8_t *offloadKeyLens,
    __gm__ uint8_t *reqValid,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *topkMissCounts,
    __gm__ uint8_t *missSourceIds, __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts, __gm__ uint8_t *workspace,
    const LIUMtpTilingData *__restrict tiling, TPipe *pipe)
{
    tmpBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        aiCoreIdx = tmpBlockIdx / 2U;
    } else {
        aiCoreIdx = tmpBlockIdx;
    }

    constInfo.batchSize = tiling->bSize;
    constInfo.qSeqSize = tiling->tSize;
    constInfo.kSeqSize = tiling->s2Size;
    constInfo.kCacheBlockSize = tiling->blockSize;
    constInfo.maxBlockNumPerBatch = tiling->maxBlockNumPerBatch;
    constInfo.poolSize = tiling->poolSize;
    constInfo.cacheSlotsSize = tiling->cacheSlotsSize;
    constInfo.qHeadNum = tiling->n1Size;

    uint64_t singleCoreMm1Bytes =
        WS_DOUBLE * constInfo.qHeadNum * constInfo.s2BaseSize *
        sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1Bytes));
    uint64_t scoresOffset =
        static_cast<uint64_t>(tiling->usedCoreNum) * singleCoreMm1Bytes;
    aggregateScoresGm.SetGlobalBuffer((__gm__ float *)(workspace + scoresOffset));
    uint64_t scoreStride =
        CeilDiv(static_cast<uint64_t>(constInfo.kSeqSize),
                static_cast<uint64_t>(constInfo.s2BaseSize)) *
        constInfo.s2BaseSize;
    uint64_t topkOffset = scoresOffset +
        constInfo.batchSize * scoreStride * sizeof(float);
    internalTopkPayloadsGm.SetGlobalBuffer(
        (__gm__ int32_t *)(workspace + topkOffset));
    uint64_t thresholdOffset = topkOffset +
        constInfo.batchSize * MAX_UNION_TOKENS * sizeof(int32_t);
    internalThresholdsGm.SetGlobalBuffer(
        (__gm__ float *)(workspace + thresholdOffset));

    reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries,
                                     constInfo.batchSize);
    cacheStateGm.SetGlobalBuffer((__gm__ int32_t *)cacheState,
                                 constInfo.poolSize);
    actualQueryLensGm.SetGlobalBuffer((__gm__ int32_t *)actualQueryLens,
                                      constInfo.batchSize);
    actualKeyLensGm.SetGlobalBuffer((__gm__ int32_t *)actualKeyLens,
                                    constInfo.batchSize);
    offloadKeyLensGm.SetGlobalBuffer((__gm__ int32_t *)offloadKeyLens,
                                     constInfo.batchSize);
    reqValidGm.SetGlobalBuffer((__gm__ int32_t *)reqValid,
                               constInfo.batchSize);
    cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
    InitRequestRange(tiling->usedCoreNum);

    if ASCEND_IS_AIV {
        vectorService.InitParams(static_cast<uint32_t>(constInfo.kSeqSize),
                                 static_cast<uint32_t>(constInfo.qHeadNum),
                                 constInfo.cacheSlotsSize);
        weightsGm.SetGlobalBuffer((__gm__ K_T *)weights);
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        topkSourceIdsGm.SetGlobalBuffer((__gm__ int32_t *)topkSourceIds);
        topkMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts,
                                         constInfo.qSeqSize);
        missSourceIdsGm.SetGlobalBuffer((__gm__ int32_t *)missSourceIds);
        missDestinationSlotsGm.SetGlobalBuffer(
            (__gm__ int32_t *)missDestinationSlots);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        vectorService.InitMtpGlobalTensor(
            mm1ResGm, weightsGm, cacheSlotsGm, topkSlotsGm,
            topkSourceIdsGm,
            missSourceIdsGm, missDestinationSlotsGm, missCountsGm,
            topkMissCountsGm,
            cacheStateGm,
            aggregateScoresGm, internalTopkPayloadsGm,
            internalThresholdsGm);
        vectorService.InitMtpBuffers(pipe);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm,
                                         mm1ResGm);
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::CleanRequest(uint32_t bIdx)
{
    if ASCEND_IS_AIV {
        if ((tmpBlockIdx & 1U) == 0U) {
            vectorService.WriteMtpZeroMissCount(bIdx);
            int32_t begin = bIdx == 0U ? 0 : actualQueryLensGm.GetValue(bIdx - 1U);
            int32_t end = actualQueryLensGm.GetValue(bIdx);
            if (begin >= 0 && end >= begin &&
                static_cast<uint32_t>(end) <= constInfo.qSeqSize) {
                for (int32_t row = begin; row < end; ++row) {
                    topkMissCountsGm.SetValue(static_cast<uint32_t>(row), 0);
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline bool LIMtpPreload<LIT>::IsQueryLayoutValid()
{
    int32_t begin = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        int32_t end = actualQueryLensGm.GetValue(bIdx);
        if (end <= begin || end - begin > 4 ||
            static_cast<uint32_t>(end) > constInfo.qSeqSize) {
            return false;
        }
        begin = end;
    }
    return static_cast<uint32_t>(begin) == constInfo.qSeqSize;
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::Process()
{
    if (requestCount == 0U) {
        return;
    }
    if (!IsQueryLayoutValid()) {
        for (uint32_t requestOffset = 0; requestOffset < requestCount;
             ++requestOffset) {
            CleanRequest(requestStart + requestOffset);
        }
        return;
    }
    ProcessMain();
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::ProcessMain()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(
            constInfo.syncV1C1);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(
            constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    uint32_t loop = 0U;
    for (uint32_t requestOffset = 0; requestOffset < requestCount;
         ++requestOffset) {
        uint32_t bIdx = requestStart + requestOffset;
        if (reqValidGm.GetValue(bIdx) == 0) {
            CleanRequest(bIdx);
            continue;
        }
        int32_t queryBeginValue = bIdx == 0U ? 0 : actualQueryLensGm.GetValue(bIdx - 1U);
        int32_t queryEndValue = actualQueryLensGm.GetValue(bIdx);
        int32_t poolEntry = reqPoolEntriesGm.GetValue(bIdx);
        if (queryBeginValue < 0 || queryEndValue <= queryBeginValue ||
            queryEndValue - queryBeginValue > 4 || poolEntry < 0 ||
            static_cast<uint32_t>(poolEntry) >= constInfo.poolSize) {
            CleanRequest(bIdx);
            continue;
        }
        int32_t cacheState = cacheStateGm.GetValue(poolEntry);
        // req_pool_entries selects the request's persistent state row.  Once
        // -3 is observed, the plain-LI path bypasses cache_slots_pool and all
        // miss-union management; the row lookup itself is still unavoidable
        // because cache_state is indexed by POOL_SIZE rather than by B.
        bool isPlainLi = cacheState == -3;
        int32_t actualKeyLen = actualKeyLensGm.GetValue(bIdx);
        int32_t offloadKeyLen = offloadKeyLensGm.GetValue(bIdx);
        if (actualKeyLen < 0 || offloadKeyLen < 0 ||
            offloadKeyLen > actualKeyLen ||
            (!isPlainLi && static_cast<uint32_t>(offloadKeyLen) %
                               constInfo.kCacheBlockSize != 0U)) {
            CleanRequest(bIdx);
            continue;
        }
        int32_t keyLenValue = isPlainLi ? actualKeyLen : offloadKeyLen;
        uint32_t candidateLen = keyLenValue > 0 ?
            static_cast<uint32_t>(keyLenValue) / constInfo.kCacheBlockSize *
                constInfo.kCacheBlockSize : 0U;
        if (candidateLen < MIN_SOURCE_TOKENS ||
            candidateLen > constInfo.kSeqSize ||
            candidateLen > constInfo.cacheSlotsSize ||
            (!isPlainLi && cacheState < -1)) {
            CleanRequest(bIdx);
            continue;
        }

        // While free slots remain, recover C from the complete persistent
        // pool row rather than only from the current offloaded prefix.  This
        // is the one phase in which the 1-based negative bindings still carry
        // the capacity information. Once cache_state reaches -1, all slots
        // are valid and the established full-cache path needs no rescan.
        uint32_t cacheTokenCount =
            static_cast<uint32_t>(LIServiceVec::INVALID_SLOT14);
        if (!isPlainLi && cacheState >= 0) {
            cacheTokenCount = 0U;
            bool invalidCapacity = false;
            uint64_t cacheBase = static_cast<uint64_t>(poolEntry) *
                                 constInfo.cacheSlotsSize;
            for (uint32_t token = 0; token < constInfo.cacheSlotsSize;
                 ++token) {
                int32_t value = cacheSlotsGm.GetValue(cacheBase + token);
                uint32_t encodedCapacity = 0U;
                if (value >= 0) {
                    encodedCapacity = static_cast<uint32_t>(value) + 1U;
                } else if (value != -65536) {
                    // Avoid signed overflow for malformed INT32_MIN input.
                    encodedCapacity =
                        static_cast<uint32_t>(-(value + 1)) + 1U;
                }
                if (encodedCapacity >
                    static_cast<uint32_t>(LIServiceVec::INVALID_SLOT14)) {
                    invalidCapacity = true;
                    break;
                }
                cacheTokenCount = Max(cacheTokenCount, encodedCapacity);
            }
            uint32_t requestTopkCapacity =
                static_cast<uint32_t>(queryEndValue - queryBeginValue) *
                TOPK_TOKENS;
            uint32_t requiredCache = Min(candidateLen, requestTopkCapacity);
            if (invalidCapacity || cacheTokenCount < requiredCache) {
                CleanRequest(bIdx);
                continue;
            }
        }

        uint32_t chunkCount = CeilDiv(candidateLen, constInfo.s2BaseSize);
        uint32_t queryCount = static_cast<uint32_t>(queryEndValue - queryBeginValue);
        for (uint32_t queryIdx = 0; queryIdx < queryCount; ++queryIdx) {
            for (uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
                RunInfo runInfo{};
                runInfo.loop = loop++;
                runInfo.bIdx = bIdx;
                runInfo.queryBegin = static_cast<uint32_t>(queryBeginValue);
                runInfo.queryCount = queryCount;
                runInfo.queryRow = runInfo.queryBegin + queryIdx;
                runInfo.queryIdx = queryIdx;
                runInfo.s2Idx = chunkIdx;
                runInfo.segmentChunkIdx = chunkIdx;
                runInfo.actS2Size = candidateLen;
                runInfo.cacheTokenCount = cacheTokenCount;
                runInfo.cacheRowIdx = static_cast<uint32_t>(poolEntry);
                runInfo.cacheState = cacheState;
                runInfo.isPlainLi = isPlainLi;
                uint32_t chunkStart = chunkIdx * constInfo.s2BaseSize;
                runInfo.actualSingleProcessSInnerSize =
                    Min(constInfo.s2BaseSize, candidateLen - chunkStart);
                runInfo.actualSingleProcessSInnerSizeAlign = LICommon::Align(
                    runInfo.actualSingleProcessSInnerSize,
                    ConstInfo::BUFFER_SIZE_BYTE_32B);
                runInfo.isFirstS2InnerLoop = chunkIdx == 0U;
                runInfo.isLastS2InnerLoop = chunkIdx + 1U == chunkCount;
                runInfo.isPartialSegment = false;
                runInfo.partialSlot = 0U;
                ProcessChunk(runInfo);
            }
        }
    }

    if ASCEND_IS_AIC {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::ProcessChunk(const RunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
            constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVecMtp(runInfo);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(
            constInfo.syncV1C1);
    }
}

} // namespace LIMtpKernel

#endif
