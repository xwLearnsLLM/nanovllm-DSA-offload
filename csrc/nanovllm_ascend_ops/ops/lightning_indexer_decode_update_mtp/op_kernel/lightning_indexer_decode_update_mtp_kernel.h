/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef NANOVLLM_LIGHTNING_INDEXER_DECODE_UPDATE_MTP_KERNEL_H_
#define NANOVLLM_LIGHTNING_INDEXER_DECODE_UPDATE_MTP_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_common.h"
#include "../../lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_cube.h"
#include "../../lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h"

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
        __gm__ uint8_t *cacheSlots, __gm__ uint8_t *cacheTokens,
        __gm__ uint8_t *candidateLens, __gm__ uint8_t *blockTable,
        __gm__ uint8_t *topkSlots, __gm__ uint8_t *missSourceIds,
        __gm__ uint8_t *missDestinationSlots, __gm__ uint8_t *missCounts,
        __gm__ uint8_t *workspace,
        const LIUMtpTilingData *__restrict tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t QUERY_COUNT = 4;
    static constexpr uint32_t MIN_SOURCE_TOKENS = 2048;
    static constexpr uint32_t MAX_UNION_TOKENS = 8192;

    LIMatmul<LIT> matmulService;
    LIVector<LIT> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<uint32_t> candidateLensGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missSourceIdsGm;
    GlobalTensor<int32_t> missDestinationSlotsGm;
    GlobalTensor<int32_t> missCountsGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> aggregateScoresGm;
    GlobalTensor<int32_t> internalTopkIndicesGm;

    uint32_t tmpBlockIdx = 0;
    uint32_t aiCoreIdx = 0;
    uint32_t requestStart = 0;
    uint32_t requestCount = 0;
    ConstInfo constInfo{};

    __aicore__ inline void InitRequestRange(uint32_t requestedCoreNum);
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessChunk(const RunInfo &runInfo);
    __aicore__ inline void CleanRequest(uint32_t bIdx);
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
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *candidateLens,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *missSourceIds, __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts, __gm__ uint8_t *workspace,
    const LIUMtpTilingData *__restrict tiling, TPipe *pipe)
{
    tmpBlockIdx = GetBlockIdx();
    aiCoreIdx = ASCEND_IS_AIV ? tmpBlockIdx / 2U : tmpBlockIdx;

    constInfo.batchSize = tiling->bSize;
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
    internalTopkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + topkOffset));

    reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries,
                                     constInfo.batchSize);
    cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens,
                                  constInfo.batchSize);
    candidateLensGm.SetGlobalBuffer((__gm__ uint32_t *)candidateLens,
                                    constInfo.batchSize);
    cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
    InitRequestRange(tiling->usedCoreNum);

    if ASCEND_IS_AIV {
        vectorService.InitParams(static_cast<uint32_t>(constInfo.kSeqSize),
                                 static_cast<uint32_t>(constInfo.qHeadNum),
                                 constInfo.cacheSlotsSize);
        weightsGm.SetGlobalBuffer((__gm__ K_T *)weights);
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        missSourceIdsGm.SetGlobalBuffer((__gm__ int32_t *)missSourceIds);
        missDestinationSlotsGm.SetGlobalBuffer(
            (__gm__ int32_t *)missDestinationSlots);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        vectorService.InitMtpGlobalTensor(
            mm1ResGm, weightsGm, cacheSlotsGm, topkSlotsGm,
            missSourceIdsGm, missDestinationSlotsGm, missCountsGm,
            aggregateScoresGm, internalTopkIndicesGm);
        vectorService.InitBuffers(pipe);
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
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMtpPreload<LIT>::Process()
{
    if (requestCount == 0U) {
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
        uint32_t candidateLen = candidateLensGm.GetValue(bIdx);
        int32_t poolEntry = reqPoolEntriesGm.GetValue(bIdx);
        int32_t cacheTokenCount = cacheTokensGm.GetValue(bIdx);
        uint32_t requiredCache = Min(candidateLen, MAX_UNION_TOKENS);
        if (candidateLen < MIN_SOURCE_TOKENS ||
            candidateLen > constInfo.kSeqSize ||
            candidateLen > constInfo.cacheSlotsSize ||
            poolEntry < 0 ||
            static_cast<uint32_t>(poolEntry) >= constInfo.poolSize ||
            cacheTokenCount == 0 || cacheTokenCount < 0 ||
            static_cast<uint32_t>(cacheTokenCount) < requiredCache ||
            static_cast<uint32_t>(cacheTokenCount) >
                static_cast<uint32_t>(LIServiceVec::INVALID_SLOT14) ||
            static_cast<uint32_t>(cacheTokenCount) > candidateLen) {
            CleanRequest(bIdx);
            continue;
        }

        uint32_t chunkCount = CeilDiv(candidateLen, constInfo.s2BaseSize);
        for (uint32_t queryIdx = 0; queryIdx < QUERY_COUNT; ++queryIdx) {
            for (uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
                RunInfo runInfo{};
                runInfo.loop = loop++;
                runInfo.bIdx = bIdx;
                runInfo.queryRow = bIdx * QUERY_COUNT + queryIdx;
                runInfo.queryIdx = queryIdx;
                runInfo.s2Idx = chunkIdx;
                runInfo.segmentChunkIdx = chunkIdx;
                runInfo.actS2Size = candidateLen;
                runInfo.cacheTokenCount =
                    static_cast<uint32_t>(cacheTokenCount);
                runInfo.cacheRowIdx = static_cast<uint32_t>(poolEntry);
                uint32_t chunkStart = chunkIdx * constInfo.s2BaseSize;
                runInfo.actualSingleProcessSInnerSize =
                    Min(constInfo.s2BaseSize, candidateLen - chunkStart);
                runInfo.actualSingleProcessSInnerSizeAlign = Align(
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
