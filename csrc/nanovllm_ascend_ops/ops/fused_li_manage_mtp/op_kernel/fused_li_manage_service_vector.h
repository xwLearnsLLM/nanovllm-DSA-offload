/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_li_manage_service_vector.h
 * \brief
 */
#ifndef FUSED_LI_MANAGE_SERVICE_VECTOR_H
#define FUSED_LI_MANAGE_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_li_manage_common.h"
#include "fused_li_manage_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t OUTPUT_CAPACITY = 2048;
constexpr uint32_t S2_BASE_SIZE = 512;
constexpr uint32_t GROUP_INNER = 16;
constexpr uint32_t PAYLOAD_BUF_SLOTS = 4;
constexpr uint32_t EVICT_CANDIDATE_CAP = BASE_TOPK;
// The target MTP workload has about 300-400 unique union misses.  Preloading
// one 512-entry block keeps q3's per-chunk merge at 512+512; atypical larger
// miss sets retain exact semantics through FinalizeMtpRequest's GM fallback.
constexpr uint32_t MTP_EVICT_PRELOAD_CAP = S2_BASE_SIZE;
static_assert(MTP_EVICT_PRELOAD_CAP % S2_BASE_SIZE == 0U &&
                  MTP_EVICT_PRELOAD_CAP <= EVICT_CANDIDATE_CAP,
              "MTP eviction preload must fit the shared candidate buffer");
constexpr uint32_t SORT_TMP_FLOATS = 512 * 8;
constexpr uint32_t CHUNK_PAIR_FLOATS = 512 * VALUE_AND_INDEX_NUM;
constexpr uint32_t TOPK_PAIR_FLOATS = BASE_TOPK * VALUE_AND_INDEX_NUM;
constexpr uint32_t EVICT_PAIR_FLOATS = EVICT_CANDIDATE_CAP * VALUE_AND_INDEX_NUM;
constexpr uint32_t SORTED_SCRATCH_FLOATS = SORT_TMP_FLOATS + CHUNK_PAIR_FLOATS;
constexpr uint32_t SORT_BUFFER_FLOATS = TOPK_PAIR_FLOATS + EVICT_PAIR_FLOATS + SORTED_SCRATCH_FLOATS;
constexpr uint32_t PARTIAL_SLOTS_PER_CORE = 2;
constexpr uint32_t PARTIAL_META_INTS_PER_CORE = 8;
constexpr uint32_t MTP_QUERY_COUNT = 4;
constexpr uint32_t MTP_UNION_CAPACITY = MTP_QUERY_COUNT * BASE_TOPK;
// MTP LIM intentionally remains on the validated 18-bit source format.
constexpr uint32_t MTP_SOURCE_CAPACITY = 1U << 18;
constexpr uint32_t MTP_UNION_BITSET_WORDS = MTP_SOURCE_CAPACITY / 32U;
constexpr uint32_t MTP_MISS_SLOT_MAP_CAPACITY = 2048;
constexpr uint32_t MTP_MISS_SLOT_MAP_MAX_ITEMS =
    MTP_MISS_SLOT_MAP_CAPACITY / 2U;
// The qlen=1 path can reconstruct three additional index bits from the score.
constexpr uint32_t EXACT_PACKED_SOURCE_TOKENS = 1U << INDEX_BITS;
static_assert(S2_BASE_SIZE % (1U << INDEX_HIGH_BITS) == 0,
              "Score-tag encoding requires each chunk to preserve absolute index low bits");

#ifndef LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS
#define LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS 4
#endif

static_assert(LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS >= 0,
              "LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS must be non-negative");
static_assert(LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS <= 512,
              "LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS must not exceed 512");
constexpr uint32_t EVICT_EXTRA_SCAN_CHUNKS = LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS;

__aicore__ inline uint32_t HashEvictScanSeed(uint32_t actualSeqLen, uint32_t cacheRowIdx)
{
    uint32_t value = actualSeqLen ^ ((cacheRowIdx + 1U) * 0x9e3779b9U);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

template <typename LIT>
class LIVector {
public:
    using K_T = typename LIT::keyType;

    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessVecMtp(const LICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMtpBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(
        uint32_t kSeqSize, uint32_t qHeadNum, uint32_t cacheSlotsSize);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<int32_t> cacheSlotsGm,
                                                GlobalTensor<int32_t> topkIndexGm,
                                                GlobalTensor<int32_t> topkSlotsGm,
                                                GlobalTensor<int32_t> missCountGm,
                                                GlobalTensor<float> scoresGm,
                                                GlobalTensor<float> partialTopkGm,
                                                GlobalTensor<int32_t> partialMetaGm);
    __aicore__ inline void InitMtpGlobalTensor(
        GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
        GlobalTensor<int32_t> cacheSlotsGm,
        GlobalTensor<int32_t> topkSlotsGm,
        GlobalTensor<int32_t> topkSourceIdsGm,
        GlobalTensor<int32_t> missSourceIdsGm,
        GlobalTensor<int32_t> missDestinationSlotsGm,
        GlobalTensor<int32_t> missCountGm,
        GlobalTensor<float> scoresGm,
        GlobalTensor<int32_t> mtpTopkPayloadsGm,
        GlobalTensor<float> mtpThresholdsGm);
    __aicore__ inline void InitPartialMetadata(uint32_t coreIdx);
    __aicore__ inline void FinalizePartialRequest(uint32_t bIdx, uint32_t cacheRowIdx,
                                                  uint32_t actualSeqLen,
                                                  uint32_t cacheTokenCount,
                                                  uint32_t firstOwner, uint32_t lastOwner);
    __aicore__ inline void WriteZeroMissCount(uint32_t bIdx);
    __aicore__ inline void WriteMtpZeroMissCount(uint32_t bIdx);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<float> scoresGm;
    GlobalTensor<float> partialTopkGm;
    GlobalTensor<int32_t> partialMetaGm;
    // MTP stays on the validated 18-bit source format. Preserve TopK's full
    // (old_slot14, token18) payload so finalization does not reconstruct the
    // pre-update hit/miss state through random GM lookups.
    GlobalTensor<int32_t> mtpTopkPayloadsGm;
    GlobalTensor<int32_t> mtpTopkSourceIdsGm;
    GlobalTensor<int32_t> mtpMissSourceIdsGm;
    GlobalTensor<int32_t> mtpMissDestinationSlotsGm;
    GlobalTensor<float> mtpThresholdsGm;

private:
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> aggregateScoreBuf_;
    TBuf<TPosition::VECCALC> payloadBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> partialMetaBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> evictCandidateUb_;
    LocalTensor<float> SortedBasicBlock_;
    LocalTensor<int32_t> partialMetaLocal_;

    static constexpr int32_t s2BaseSize_ = S2_BASE_SIZE;
    uint32_t scoreStride_ = 0;
    uint32_t gSize_ = 64;
    uint32_t outerG_ = 4;
    uint32_t cacheSlotsSize_ = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    __aicore__ inline void StartPayloadCopy(LocalTensor<int32_t> &payloadLocal, uint32_t cacheRowIdx,
                                            int32_t s2BaseIdx, int32_t validLen, int32_t alignedLen);
    __aicore__ inline void FinishPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx, int32_t validLen);
    __aicore__ inline void PrepareSortScore(const LocalTensor<float> &scoreLocal,
                                            const LocalTensor<float> &reducedScoreLocal,
                                            int32_t s2BaseIdx, int32_t validLen,
                                            bool hasLongIndexTag);
    __aicore__ inline void SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                          const LocalTensor<float> &keyLocal,
                                          const LocalTensor<int32_t> &payloadBase,
                                          LocalTensor<float> &tmpSortBuf, bool hasLongIndexTag);
    __aicore__ inline void WriteScoreChunk(uint32_t bIdx, int32_t s2BaseIdx,
                                           const LocalTensor<float> &scoreLocal, int32_t alignedLen);
    __aicore__ inline void WriteMtpAggregateScoreChunk(
        uint32_t bIdx, uint32_t queryIdx, int32_t s2BaseIdx,
        const LocalTensor<float> &scoreLocal, int32_t alignedLen);
    __aicore__ inline void CollectMtpEvictCandidateChunk(
        const LICommon::RunInfo &info,
        const LocalTensor<int32_t> &payloadLocal,
        LocalTensor<float> &tmpSortBuf,
        uint32_t cachedChunkIdx);
    __aicore__ inline void StoreMtpQueryTopK(const LICommon::RunInfo &info);
    __aicore__ inline void FinalizeMtpRequest(const LICommon::RunInfo &info);
    __aicore__ inline void WritePartialTopK(uint32_t coreIdx, uint32_t partialSlot, uint32_t bIdx);
    __aicore__ inline void SortEvictCandidateChunk(uint32_t bIdx, uint32_t cacheRowIdx, uint32_t chunkIdx,
                                                   uint32_t actualSeqLen,
                                                   const LocalTensor<float> &pairOut,
                                                   LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void MergeEvictCandidateChunk(const LocalTensor<float> &chunkPairLocal,
                                                     uint32_t candidateCap,
                                                     LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void MergeEvictChunkBatch512(uint32_t bIdx, uint32_t cacheRowIdx,
                                                   uint32_t actualSeqLen, uint32_t startChunk,
                                                   uint32_t chunkNum, uint32_t firstScanOffset,
                                                   uint32_t batchChunks, LocalTensor<float> &tmpSortBuf);
    __aicore__ inline bool FindEvictCandidates(uint32_t bIdx, uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                               uint32_t scanSeed, uint32_t missCount, uint32_t candidateCap,
                                               float thresholdScore, LocalTensor<float> &tmpSortBuf);
    __aicore__ inline uint32_t CopyDecodedPayloadOut(int64_t outOffset, const LocalTensor<float> &pairLocal,
                                                     const LocalTensor<uint32_t> &payloadLocal,
                                                     const LocalTensor<int32_t> &indexLocal,
                                                     const LocalTensor<int32_t> &slotLocal,
                                                     const LocalTensor<int32_t> &scratchLocal,
                                                     int64_t count, bool hasLongIndexTag, bool mayHaveInvalid);
    __aicore__ inline bool IsTopKIndex(const LocalTensor<int32_t> &indexLocal, uint32_t candidateIndex,
                                       uint32_t count) const;
    __aicore__ inline bool FindFallbackEvict(uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                             uint32_t cacheTokenCount,
                                             const LocalTensor<int32_t> &indexLocal, uint32_t &scanCursor,
                                             uint32_t &evictIndex, int32_t &evictSlot);
    __aicore__ inline void UpdateCacheAndWriteTopkSlots(uint32_t bIdx, uint32_t cacheRowIdx,
                                                        int64_t outOffset, uint32_t actualSeqLen,
                                                        uint32_t cacheTokenCount, uint32_t scanSeed,
                                                        float thresholdScore, uint32_t missCount,
                                                        const LocalTensor<int32_t> &indexLocal,
                                                        const LocalTensor<uint32_t> &candidatePayloadLocal,
                                                        const LocalTensor<int32_t> &scalarLocal,
                                                        const LocalTensor<int32_t> &topkSlotsLocal);
    __aicore__ inline void WriteMissCount(uint32_t bIdx, int32_t missCount,
                                          const LocalTensor<int32_t> &scalarLocal);
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }
    uint32_t outNeedBufSize = TOPK_PAIR_FLOATS * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + GROUP_INNER * S2_BASE_SIZE * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;

    pipe->InitBuffer(inQueue_, 2,
                     GROUP_INNER * S2_BASE_SIZE * sizeof(float) + S2_BASE_SIZE * sizeof(float));
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
    pipe->InitBuffer(sortOutBuf_, SORT_BUFFER_FLOATS * sizeof(float));
    pipe->InitBuffer(indexBuf_, S2_BASE_SIZE * sizeof(int32_t));
    pipe->InitBuffer(payloadBuf_, S2_BASE_SIZE * PAYLOAD_BUF_SLOTS * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, S2_BASE_SIZE * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, GROUP_INNER * 8 * sizeof(float));
    pipe->InitBuffer(partialMetaBuf_, PARTIAL_META_INTS_PER_CORE * sizeof(int32_t));

    //
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    evictCandidateUb_ = globalTopkUb_[TOPK_PAIR_FLOATS];
    SortedBasicBlock_ = evictCandidateUb_[EVICT_PAIR_FLOATS];
    partialMetaLocal_ = partialMetaBuf_.Get<int32_t>();

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, S2_BASE_SIZE);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitMtpBuffers(TPipe *pipe)
{
    if ((GetBlockIdx() & 1U) != 0U) {
        return;
    }
    InitBuffers(pipe);
    // Only MTP needs the fourth-query aggregate-score scratch. Keep it out
    // of the single-query LIM UB footprint.
    pipe->InitBuffer(aggregateScoreBuf_, S2_BASE_SIZE * sizeof(float));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(
    uint32_t kSeqSize, uint32_t qHeadNum, uint32_t cacheSlotsSize)
{
    scoreStride_ = CeilDiv(kSeqSize, static_cast<uint32_t>(s2BaseSize_)) *
                   static_cast<uint32_t>(s2BaseSize_);
    gSize_ = qHeadNum;
    outerG_ = qHeadNum / GROUP_INNER;
    cacheSlotsSize_ = cacheSlotsSize;
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                    GlobalTensor<int32_t> cacheSlotsGm, GlobalTensor<int32_t> topkIndexGm,
                                    GlobalTensor<int32_t> topkSlotsGm, GlobalTensor<int32_t> missCountGm,
                                    GlobalTensor<float> scoresGm, GlobalTensor<float> partialTopkGm,
                                    GlobalTensor<int32_t> partialMetaGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkIndexGm = topkIndexGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
    this->scoresGm = scoresGm;
    this->partialTopkGm = partialTopkGm;
    this->partialMetaGm = partialMetaGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitMtpGlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
    GlobalTensor<int32_t> cacheSlotsGm,
    GlobalTensor<int32_t> topkSlotsGm,
    GlobalTensor<int32_t> topkSourceIdsGm,
    GlobalTensor<int32_t> missSourceIdsGm,
    GlobalTensor<int32_t> missDestinationSlotsGm,
    GlobalTensor<int32_t> missCountGm,
    GlobalTensor<float> scoresGm,
    GlobalTensor<int32_t> mtpTopkPayloadsGm,
    GlobalTensor<float> mtpThresholdsGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->mtpTopkSourceIdsGm = topkSourceIdsGm;
    this->mtpMissSourceIdsGm = missSourceIdsGm;
    this->mtpMissDestinationSlotsGm = missDestinationSlotsGm;
    this->missCountGm = missCountGm;
    this->scoresGm = scoresGm;
    this->mtpTopkPayloadsGm = mtpTopkPayloadsGm;
    this->mtpThresholdsGm = mtpThresholdsGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitPartialMetadata(uint32_t coreIdx)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }
    Duplicate(partialMetaLocal_, LICommon::ConstInfo::INVALID_IDX, PARTIAL_META_INTS_PER_CORE);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(partialMetaGm[coreIdx * PARTIAL_META_INTS_PER_CORE],
                          partialMetaLocal_, PARTIAL_META_INTS_PER_CORE);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteZeroMissCount(uint32_t bIdx)
{
    // SCATTER ignores source/destination entries when copy_count is zero, so
    // C=0 rows only need this scalar write.  Clearing two 2048-element output
    // rows here would make mixed short/long batches needlessly expensive.
    LocalTensor<int32_t> scalarLocal = reduceOutBuf_.Get<int32_t>();
    WriteMissCount(bIdx, 0, scalarLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteMtpZeroMissCount(uint32_t bIdx)
{
    WriteZeroMissCount(bIdx);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::StartPayloadCopy(LocalTensor<int32_t> &payloadLocal, uint32_t cacheRowIdx,
                                                       int32_t s2BaseIdx, int32_t validLen, int32_t alignedLen)
{
    if (validLen < alignedLen) {
        Duplicate(payloadLocal, LICommon::ConstInfo::INVALID_IDX, alignedLen);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    }
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    DataCopyPad(payloadLocal, cacheSlotsGm[rowBase + static_cast<uint32_t>(s2BaseIdx)],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FinishPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx,
                                                    int32_t validLen)
{
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(), payloadLocal.template ReinterpretCast<uint32_t>(),
              INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal, s2BaseIdx & static_cast<int32_t>(INDEX_MASK), validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::PrepareSortScore(
    const LocalTensor<float> &scoreLocal, const LocalTensor<float> &reducedScoreLocal,
    int32_t s2BaseIdx, int32_t validLen, bool hasLongIndexTag)
{
    if (validLen < s2BaseSize_) {
        Duplicate(scoreLocal.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, s2BaseSize_);
        PipeBarrier<PIPE_V>();
    }
    if (!hasLongIndexTag) {
        Adds(scoreLocal, reducedScoreLocal, 0.0f, validLen);
        PipeBarrier<PIPE_V>();
        return;
    }

    // Copy while clearing the three score-tag bits. This replaces the old
    // score copy plus in-place ShiftRight with one vector operation.
    int32_t indexHigh3 = (s2BaseIdx >> INDEX_BITS) & static_cast<int32_t>(INDEX_HIGH_MASK);
    LocalTensor<uint32_t> scoreBitsLocal = scoreLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> reducedScoreBitsLocal = reducedScoreLocal.template ReinterpretCast<uint32_t>();
    ShiftRight(scoreBitsLocal, reducedScoreBitsLocal, SCORE_TAG_CLEAR_SHIFT, validLen);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scoreBitsLocal, scoreBitsLocal, SCORE_TAG_CLEAR_SHIFT, validLen);
    PipeBarrier<PIPE_V>();
    Adds(scoreBitsLocal.template ReinterpretCast<int32_t>(), scoreBitsLocal.template ReinterpretCast<int32_t>(),
         indexHigh3, validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                                     const LocalTensor<float> &keyLocal,
                                                     const LocalTensor<int32_t> &payloadBase,
                                                     LocalTensor<float> &tmpSortBuf, bool hasLongIndexTag)
{
    for (uint32_t blockIdx = 0; blockIdx < PAYLOAD_BUF_SLOTS; ++blockIdx) {
        uint32_t pairOffset = blockIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM;
        LocalTensor<int32_t> payloadLocal = payloadBase[blockIdx * s2BaseSize_];
        ExtractIndex(payloadLocal.template ReinterpretCast<uint32_t>(),
                     pairLocal[pairOffset].template ReinterpretCast<uint32_t>(), s2BaseSize_);
        if (hasLongIndexTag) {
            LocalTensor<uint32_t> scoreTagLocal = tmpSortBuf.template ReinterpretCast<uint32_t>();
            ExtractScoreBits(scoreTagLocal, pairLocal[pairOffset].template ReinterpretCast<uint32_t>(), s2BaseSize_);
            BuildTaggedHitMissKey(keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), scoreTagLocal,
                                  tmpSortBuf[s2BaseSize_], s2BaseSize_);
        } else {
            BuildHitMissKey(keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), s2BaseSize_);
        }
        SortByKeyWithPayload512(pairLocal[pairOffset], keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(),
                                tmpSortBuf, s2BaseSize_ / BLOCK_BYTES);
    }

    MrgBasicBlock(tmpSortBuf, pairLocal, PAYLOAD_BUF_SLOTS, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    DataCopy(pairLocal, tmpSortBuf, BASE_TOPK * VALUE_AND_INDEX_NUM);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteScoreChunk(uint32_t bIdx, int32_t s2BaseIdx,
                                                      const LocalTensor<float> &scoreLocal, int32_t alignedLen)
{
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(scoresGm[bIdx * scoreStride_ + static_cast<uint32_t>(s2BaseIdx)], scoreLocal, alignedLen);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteMtpAggregateScoreChunk(
    uint32_t bIdx, uint32_t queryIdx, int32_t s2BaseIdx,
    const LocalTensor<float> &scoreLocal, int32_t alignedLen)
{
    uint64_t gmOffset = static_cast<uint64_t>(bIdx) * scoreStride_ +
                        static_cast<uint32_t>(s2BaseIdx);
    LocalTensor<float> previousScore = aggregateScoreBuf_.Get<float>();
    if (queryIdx == 0U) {
        if (s2BaseIdx > 0) {
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
        DataCopy(previousScore, scoreLocal, alignedLen);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LIServiceVec::CopyOut(scoresGm[gmOffset], previousScore, alignedLen);
        return;
    }

    // Do not borrow globalTopkIndice_ here.  FinishPayload relies on that
    // buffer remaining the exact 0..511 progression across all four MTP
    // queries; using it as async GM scratch corrupts query 1+ payloads on
    // Ascend910_93 even when it is rewritten before the next chunk.
    // q1..q2 write the previous chunk's aggregate from this same UB scratch.
    // Delay that write's completion until the scratch is actually reused so
    // MTE3 can overlap the intervening TopK merge and next MM/scale work.
    if (queryIdx + 1U != MTP_QUERY_COUNT && s2BaseIdx > 0) {
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopyPad(previousScore, scoresGm[gmOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(alignedLen * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Max(previousScore, previousScore, scoreLocal, alignedLen);
    PipeBarrier<PIPE_V>();
    if (queryIdx + 1U == MTP_QUERY_COUNT) {
        // q3 produces the final max(q0..q3) score.  Its consumer now builds
        // the eviction candidate prefix directly from this UB tensor, so a
        // final GM write would only be read back by the old finalize scan.
        return;
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(scoresGm[gmOffset], previousScore, alignedLen);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CollectMtpEvictCandidateChunk(
    const LICommon::RunInfo &info,
    const LocalTensor<int32_t> &payloadLocal,
    LocalTensor<float> &tmpSortBuf,
    uint32_t cachedChunkIdx)
{
    if (info.isFirstS2InnerLoop) {
        InitSortOutBuf(evictCandidateUb_,
                       MTP_EVICT_PRELOAD_CAP * VALUE_AND_INDEX_NUM);
    }

    // q3 leaves the final max(q0..q3) score in aggregateScoreBuf_.  Reuse the
    // cache-slot payload already fetched for this score chunk, sort cached
    // entries by -aggregate_score, and retain the global lowest 512 entries.
    // SortedBasicBlock_'s current ping-pong slot is free until the q3 TopK
    // sort below overwrites it.
    LocalTensor<float> aggregateScore = aggregateScoreBuf_.Get<float>();
    LocalTensor<float> keyLocal = reduceOutBuf_.Get<float>()[s2BaseSize_];
    LocalTensor<float> chunkPairLocal =
        SortedBasicBlock_[cachedChunkIdx * CHUNK_PAIR_FLOATS];
    BuildEvictCandidateKeyFromPayload(
        keyLocal, aggregateScore,
        payloadLocal.template ReinterpretCast<uint32_t>(),
        tmpSortBuf, s2BaseSize_);
    SortByKeyWithPayload512(
        chunkPairLocal, keyLocal,
        payloadLocal.template ReinterpretCast<uint32_t>(),
        tmpSortBuf, s2BaseSize_ / BLOCK_BYTES);
    PipeBarrier<PIPE_V>();
    MergeEvictCandidateChunk(chunkPairLocal, MTP_EVICT_PRELOAD_CAP,
                             tmpSortBuf);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WritePartialTopK(uint32_t coreIdx, uint32_t partialSlot, uint32_t bIdx)
{
    uint32_t resultSlot = coreIdx * PARTIAL_SLOTS_PER_CORE + partialSlot;
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    partialMetaLocal_.SetValue(partialSlot, static_cast<int32_t>(bIdx));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(partialMetaGm[coreIdx * PARTIAL_META_INTS_PER_CORE],
                          partialMetaLocal_, PARTIAL_META_INTS_PER_CORE);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(partialTopkGm[static_cast<uint64_t>(resultSlot) * TOPK_PAIR_FLOATS],
                          globalTopkUb_, TOPK_PAIR_FLOATS);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortEvictCandidateChunk(uint32_t bIdx, uint32_t cacheRowIdx,
                                                              uint32_t chunkIdx,
                                                              uint32_t actualSeqLen,
                                                              const LocalTensor<float> &pairOut,
                                                              LocalTensor<float> &tmpSortBuf)
{
    uint32_t s2BaseIdx = chunkIdx * static_cast<uint32_t>(s2BaseSize_);
    uint32_t validLen = (s2BaseIdx + static_cast<uint32_t>(s2BaseSize_) > actualSeqLen)
                            ? (actualSeqLen - s2BaseIdx)
                            : static_cast<uint32_t>(s2BaseSize_);

    LocalTensor<float> scoreLocal = reduceOutBuf_.Get<float>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    DataCopyPad(scoreLocal, scoresGm[bIdx * scoreStride_ + s2BaseIdx],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(s2BaseSize_ * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});

    LocalTensor<int32_t> payloadLocal = payloadBuf_.Get<int32_t>();
    if (validLen < static_cast<uint32_t>(s2BaseSize_)) {
        Duplicate(payloadLocal, LICommon::ConstInfo::INVALID_IDX, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    }
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    uint8_t payloadRightPadding = static_cast<uint8_t>(
        (B32_BLOCK_ALIGN_NUM - validLen % B32_BLOCK_ALIGN_NUM) % B32_BLOCK_ALIGN_NUM);
    DataCopyPad(payloadLocal, cacheSlotsGm[rowBase + s2BaseIdx],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{payloadRightPadding != 0, 0, payloadRightPadding,
                                                       LICommon::ConstInfo::INVALID_IDX});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    // Keep the evict slot beside index_low18. For long requests index_high3 is
    // already present in the low bits of scoreLocal and survives score negation.
    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(),
              payloadLocal.template ReinterpretCast<uint32_t>(), INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal,
         static_cast<int32_t>(s2BaseIdx & INDEX_MASK), validLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> keyLocal = reduceOutBuf_.Get<float>()[s2BaseSize_];
    BuildEvictCandidateKeyFromPayload(keyLocal, scoreLocal,
                                      payloadLocal.template ReinterpretCast<uint32_t>(),
                                      tmpSortBuf, s2BaseSize_);
    SortByKeyWithPayload512(pairOut, keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                            s2BaseSize_ / BLOCK_BYTES);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeEvictCandidateChunk(const LocalTensor<float> &chunkPairLocal,
                                                               uint32_t candidateCap,
                                                               LocalTensor<float> &tmpSortBuf)
{
    uint32_t candidateBlockNum = candidateCap / static_cast<uint32_t>(s2BaseSize_);
    if (candidateBlockNum == 2U || candidateBlockNum == 3U) {
        MrgBasicBlock(tmpSortBuf, evictCandidateUb_, static_cast<int64_t>(candidateBlockNum + 1U), s2BaseSize_);
        PipeBarrier<PIPE_V>();
        DataCopy(evictCandidateUb_, tmpSortBuf, candidateCap * VALUE_AND_INDEX_NUM);
        PipeBarrier<PIPE_V>();
        return;
    }
    if (candidateBlockNum == 4U) {
        MergeSort(evictCandidateUb_, static_cast<int32_t>(candidateCap), chunkPairLocal, s2BaseSize_, tmpSortBuf);
        return;
    }

    uint32_t tailBlockOffset = (candidateBlockNum - 1U) * CHUNK_PAIR_FLOATS;
    LocalTensor<float> tailBlock = evictCandidateUb_[tailBlockOffset];
    MergeSort(tailBlock, s2BaseSize_, chunkPairLocal, s2BaseSize_, tmpSortBuf);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeEvictChunkBatch512(uint32_t bIdx, uint32_t cacheRowIdx,
                                                              uint32_t actualSeqLen, uint32_t startChunk,
                                                              uint32_t chunkNum, uint32_t firstScanOffset,
                                                              uint32_t batchChunks,
                                                              LocalTensor<float> &tmpSortBuf)
{
    for (uint32_t batchIdx = 0; batchIdx < batchChunks; ++batchIdx) {
        uint32_t scanOffset = firstScanOffset + batchIdx;
        uint32_t chunkIdx = startChunk + scanOffset;
        if (chunkIdx >= chunkNum) {
            chunkIdx -= chunkNum;
        }
        LocalTensor<float> chunkPairLocal = globalTopkUb_[batchIdx * CHUNK_PAIR_FLOATS];
        SortEvictCandidateChunk(bIdx, cacheRowIdx, chunkIdx, actualSeqLen, chunkPairLocal, tmpSortBuf);
        PipeBarrier<PIPE_V>();
    }

    if (batchChunks == 1U) {
        MergeSort(evictCandidateUb_, s2BaseSize_, globalTopkUb_, s2BaseSize_, tmpSortBuf);
    } else {
        MrgBasicBlock(tmpSortBuf, globalTopkUb_, static_cast<int64_t>(batchChunks), s2BaseSize_);
        PipeBarrier<PIPE_V>();
        MergeSort(evictCandidateUb_, s2BaseSize_, tmpSortBuf, s2BaseSize_, globalTopkUb_);
    }
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::FindEvictCandidates(uint32_t bIdx, uint32_t cacheRowIdx,
                                                          uint32_t actualSeqLen, uint32_t scanSeed,
                                                          uint32_t missCount, uint32_t candidateCap,
                                                          float thresholdScore,
                                                          LocalTensor<float> &tmpSortBuf)
{
    InitSortOutBuf(evictCandidateUb_, candidateCap * VALUE_AND_INDEX_NUM);
    float stopKey = -thresholdScore;
    uint32_t chunkNum = CeilDiv(actualSeqLen, static_cast<uint32_t>(s2BaseSize_));
    uint32_t startChunk = HashEvictScanSeed(scanSeed, cacheRowIdx) % chunkNum;
    uint32_t stopScanOffset = chunkNum;

    if (candidateCap == static_cast<uint32_t>(s2BaseSize_)) {
        uint32_t scanOffset = 0;
        while (scanOffset < chunkNum) {
            uint32_t batchChunks = Min<uint32_t, uint32_t>(PAYLOAD_BUF_SLOTS, chunkNum - scanOffset);
            if (stopScanOffset != chunkNum) {
                batchChunks = Min<uint32_t, uint32_t>(batchChunks, stopScanOffset - scanOffset);
            }
            MergeEvictChunkBatch512(bIdx, cacheRowIdx, actualSeqLen, startChunk, chunkNum,
                                    scanOffset, batchChunks, tmpSortBuf);
            scanOffset += batchChunks;

            if (stopScanOffset != chunkNum) {
                if (scanOffset >= stopScanOffset) {
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    break;
                }
                continue;
            }

            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            if (evictCandidateUb_.GetValue((missCount - 1U) * VALUE_AND_INDEX_NUM) > stopKey) {
                uint32_t remainingChunks = chunkNum - scanOffset;
                uint32_t extraChunks = EVICT_EXTRA_SCAN_CHUNKS < remainingChunks
                                           ? EVICT_EXTRA_SCAN_CHUNKS
                                           : remainingChunks;
                if (extraChunks == 0U) {
                    break;
                }
                stopScanOffset = scanOffset + extraChunks;
            }
        }
        return evictCandidateUb_.GetValue((missCount - 1U) * VALUE_AND_INDEX_NUM) > stopKey;
    }

    LocalTensor<float> chunkPairLocal = tmpSortBuf[SORT_TMP_FLOATS];
    if (candidateCap > static_cast<uint32_t>(s2BaseSize_) && candidateCap < EVICT_CANDIDATE_CAP) {
        chunkPairLocal = evictCandidateUb_[candidateCap * VALUE_AND_INDEX_NUM];
    } else if (candidateCap == EVICT_CANDIDATE_CAP) {
        chunkPairLocal = globalTopkUb_;
    }
    for (uint32_t scanOffset = 0; scanOffset < chunkNum; ++scanOffset) {
        uint32_t chunkIdx = startChunk + scanOffset;
        if (chunkIdx >= chunkNum) {
            chunkIdx -= chunkNum;
        }
        SortEvictCandidateChunk(bIdx, cacheRowIdx, chunkIdx, actualSeqLen, chunkPairLocal, tmpSortBuf);
        PipeBarrier<PIPE_V>();
        MergeEvictCandidateChunk(chunkPairLocal, candidateCap, tmpSortBuf);

        if (stopScanOffset != chunkNum) {
            if (scanOffset >= stopScanOffset) {
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                break;
            }
            continue;
        }

        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        if (evictCandidateUb_.GetValue((missCount - 1U) * VALUE_AND_INDEX_NUM) > stopKey) {
            uint32_t remainingChunks = chunkNum - scanOffset - 1U;
            uint32_t extraChunks = EVICT_EXTRA_SCAN_CHUNKS < remainingChunks
                                       ? EVICT_EXTRA_SCAN_CHUNKS
                                       : remainingChunks;
            if (extraChunks == 0U) {
                break;
            }
            stopScanOffset = scanOffset + extraChunks;
        }
    }
    return false;
}

template <typename LIT>
__aicore__ inline uint32_t LIVector<LIT>::CopyDecodedPayloadOut(
    int64_t outOffset, const LocalTensor<float> &pairLocal, const LocalTensor<uint32_t> &payloadLocal,
    const LocalTensor<int32_t> &indexLocal, const LocalTensor<int32_t> &slotLocal,
    const LocalTensor<int32_t> &scratchLocal, int64_t count, bool hasLongIndexTag, bool mayHaveInvalid)
{
    ExtractIndex(payloadLocal, pairLocal.template ReinterpretCast<uint32_t>(), count);
    DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(), payloadLocal, count);
    if (hasLongIndexTag) {
        ExtractScoreBits(scratchLocal.template ReinterpretCast<uint32_t>(),
                         pairLocal.template ReinterpretCast<uint32_t>(), count);
        DecodeLongIndexHighFromScoreTag(indexLocal.template ReinterpretCast<uint32_t>(),
                                        scratchLocal.template ReinterpretCast<uint32_t>(), count);
    }
    if (mayHaveInvalid) {
        FixInvalidIndex(indexLocal, payloadLocal.template ReinterpretCast<int32_t>(), count);
    }
    DecodeSlotFromPayload(slotLocal.template ReinterpretCast<uint32_t>(), payloadLocal, scratchLocal, count);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    uint32_t missCount = 0;
    while (missCount < static_cast<uint32_t>(count) &&
           slotLocal.GetValue(missCount) == LICommon::ConstInfo::INVALID_IDX) {
        ++missCount;
    }

    // Publish the complete top-2048 row. The prefix [0, missCount) remains
    // the miss set consumed by SCATTER, while the suffix contains cache hits.
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(topkIndexGm[outOffset], indexLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    return missCount;
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::IsTopKIndex(const LocalTensor<int32_t> &indexLocal,
                                                  uint32_t candidateIndex, uint32_t count) const
{
    for (uint32_t topkIdx = 0; topkIdx < count; ++topkIdx) {
        if (static_cast<uint32_t>(indexLocal.GetValue(topkIdx)) == candidateIndex) {
            return true;
        }
    }
    return false;
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::FindFallbackEvict(uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                                        uint32_t cacheTokenCount,
                                                        const LocalTensor<int32_t> &indexLocal,
                                                        uint32_t &scanCursor, uint32_t &evictIndex,
                                                        int32_t &evictSlot)
{
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    while (scanCursor < actualSeqLen) {
        int32_t slot = cacheSlotsGm.GetValue(rowBase + scanCursor);
        uint32_t candidateIndex = scanCursor;
        ++scanCursor;
        if (slot >= 0 && static_cast<uint32_t>(slot) < cacheTokenCount &&
            !IsTopKIndex(indexLocal, candidateIndex, BASE_TOPK)) {
            evictIndex = candidateIndex;
            evictSlot = slot;
            return true;
        }
    }
    evictIndex = 0;
    evictSlot = LICommon::ConstInfo::INVALID_IDX;
    return false;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::UpdateCacheAndWriteTopkSlots(
    uint32_t bIdx, uint32_t cacheRowIdx, int64_t outOffset, uint32_t actualSeqLen,
    uint32_t cacheTokenCount, uint32_t scanSeed,
    float thresholdScore, uint32_t missCount,
    const LocalTensor<int32_t> &indexLocal, const LocalTensor<uint32_t> &candidatePayloadLocal,
    const LocalTensor<int32_t> &scalarLocal, const LocalTensor<int32_t> &topkSlotsLocal)
{
    // Start the scalar output before eviction discovery so its MTE3 transfer
    // overlaps the vector/scalar index-management work below.
    scalarLocal.SetValue(0, static_cast<int32_t>(missCount));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(missCountGm[bIdx], scalarLocal, 1);

    uint32_t candidateCap = 0;
    bool hasFullCandidatePrefix = false;
    if (missCount > 0) {
        candidateCap = ((missCount + S2_BASE_SIZE - 1U) / S2_BASE_SIZE) * S2_BASE_SIZE;
        if (candidateCap > EVICT_CANDIDATE_CAP) {
            candidateCap = EVICT_CANDIDATE_CAP;
        }
        hasFullCandidatePrefix = FindEvictCandidates(
            bIdx, cacheRowIdx, actualSeqLen, scanSeed, missCount, candidateCap,
            thresholdScore, SortedBasicBlock_);
    }

    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    LocalTensor<uint32_t> candidateBitsLocal = evictCandidateUb_.template ReinterpretCast<uint32_t>();
    bool usedPackedFastPath = false;
    bool hasLongIndexTag = actualSeqLen > EXACT_PACKED_SOURCE_TOKENS;
    if (hasFullCandidatePrefix && missCount <= static_cast<uint32_t>(s2BaseSize_)) {
        // The miss-count copy reads scalarLocal, so wait only when this buffer
        // is about to be reused for decoded candidate indices.
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        ExtractIndex(candidatePayloadLocal, candidateBitsLocal, missCount);
        DecodeIndexFromPayload(scalarLocal.template ReinterpretCast<uint32_t>(),
                               candidatePayloadLocal, missCount);
        ShiftRight(topkSlotsLocal.template ReinterpretCast<uint32_t>(),
                   candidatePayloadLocal, INDEX_BITS, missCount);
        PipeBarrier<PIPE_V>();
        if (hasLongIndexTag) {
            ExtractScoreBits(candidatePayloadLocal, candidateBitsLocal, missCount);
            DecodeLongIndexHighFromScoreTag(scalarLocal.template ReinterpretCast<uint32_t>(),
                                            candidatePayloadLocal, missCount);
        }
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

        usedPackedFastPath = true;
        for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
            uint32_t missIndex = static_cast<uint32_t>(indexLocal.GetValue(missIdx));
            uint32_t evictIndex = static_cast<uint32_t>(scalarLocal.GetValue(missIdx));
            int32_t evictSlot = topkSlotsLocal.GetValue(missIdx);
            if (missIndex >= actualSeqLen || evictIndex >= actualSeqLen || evictSlot < 0 ||
                static_cast<uint32_t>(evictSlot) >= cacheTokenCount) {
                usedPackedFastPath = false;
                break;
            }
        }

        if (usedPackedFastPath) {
            for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
                uint32_t missIndex = static_cast<uint32_t>(indexLocal.GetValue(missIdx));
                uint32_t evictIndex = static_cast<uint32_t>(scalarLocal.GetValue(missIdx));
                int32_t evictSlot = topkSlotsLocal.GetValue(missIdx);
                cacheSlotsGm.SetValue(rowBase + evictIndex, LICommon::ConstInfo::INVALID_IDX);
                cacheSlotsGm.SetValue(rowBase + missIndex, evictSlot);
            }
        } else {
            for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
                topkSlotsLocal.SetValue(missIdx, LICommon::ConstInfo::INVALID_IDX);
            }
        }
    }

    if (!usedPackedFastPath) {
        uint32_t candidateCursor = 0;
        uint32_t fallbackCursor = 0;
        float stopKey = -thresholdScore;
        for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
            uint32_t evictIndex = 0;
            int32_t evictSlot = LICommon::ConstInfo::INVALID_IDX;
            bool foundCandidate = false;
            while (candidateCursor < candidateCap) {
                float candidateKey = evictCandidateUb_.GetValue(candidateCursor * VALUE_AND_INDEX_NUM);
                if (candidateKey <= stopKey) {
                    break;
                }
                uint32_t keyBits = candidateBitsLocal.GetValue(candidateCursor * VALUE_AND_INDEX_NUM);
                uint32_t payload = candidateBitsLocal.GetValue(candidateCursor * VALUE_AND_INDEX_NUM + 1);
                ++candidateCursor;
                uint32_t index = payload & INDEX_MASK;
                if (hasLongIndexTag) {
                    index |= (keyBits & INDEX_HIGH_MASK) << INDEX_BITS;
                }
                int32_t slot = static_cast<int32_t>(payload >> INDEX_BITS);
                if (index < actualSeqLen && slot >= 0 &&
                    static_cast<uint32_t>(slot) < cacheTokenCount) {
                    evictIndex = index;
                    evictSlot = slot;
                    foundCandidate = true;
                    break;
                }
            }
            if (!foundCandidate) {
                foundCandidate = FindFallbackEvict(
                    cacheRowIdx, actualSeqLen, cacheTokenCount, indexLocal,
                    fallbackCursor, evictIndex, evictSlot);
            }
            if (foundCandidate) {
                uint32_t missIndex = static_cast<uint32_t>(indexLocal.GetValue(missIdx));
                if (missIndex >= actualSeqLen) {
                    continue;
                }
                topkSlotsLocal.SetValue(missIdx, evictSlot);
                cacheSlotsGm.SetValue(rowBase + evictIndex, LICommon::ConstInfo::INVALID_IDX);
                cacheSlotsGm.SetValue(rowBase + missIndex, evictSlot);
            }
        }
    }
    // SCATTER consumes only the first missCount entries.  The complete row is
    // also the logical-slot list for sparse attention after the cache update,
    // so publish all top-k slots without adding another index materialization
    // kernel to every decoder layer.
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(topkSlotsGm[outOffset], topkSlotsLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteMissCount(uint32_t bIdx, int32_t missCount,
                                                     const LocalTensor<int32_t> &scalarLocal)
{
    scalarLocal.SetValue(0, missCount);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(missCountGm[bIdx], scalarLocal, 1);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::StoreMtpQueryTopK(const LICommon::RunInfo &info)
{
    LocalTensor<float> valueLocal = outQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> payloadLocal = valueLocal.template ReinterpretCast<uint32_t>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    mtpThresholdsGm.SetValue(
        info.queryRow,
        globalTopkUb_.GetValue((BASE_TOPK - 1U) * VALUE_AND_INDEX_NUM));
    ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);

    uint64_t rowOffset = static_cast<uint64_t>(info.queryRow) * BASE_TOPK;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(mtpTopkPayloadsGm[rowOffset],
                payloadLocal.template ReinterpretCast<int32_t>(),
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(valueLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FinalizeMtpRequest(const LICommon::RunInfo &info)
{
    // Two VECIN buffers hold the membership bitset and ordered union misses.
    // The bitset has 2^18-token capacity, but only its active candidate prefix
    // is touched. The VECOUT buffer first stages destination slots, then is
    // reused to materialize the four per-query sparse-slot rows.
    LocalTensor<float> unionStorage = inQueue_.AllocTensor<float>();
    LocalTensor<float> missStorage = inQueue_.AllocTensor<float>();
    LocalTensor<float> slotStorage = outQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> unionBits = unionStorage.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> missTokens = missStorage.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> destinationSlots = slotStorage.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> topkPayloads = payloadBuf_.Get<int32_t>();

    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * cacheSlotsSize_;
    const uint32_t activeUnionWords =
        Min(CeilDiv(info.actS2Size, 32U), MTP_UNION_BITSET_WORDS);
    uint32_t missCount = 0;
    AscendC::DataCopyExtParams copyIn{1, BASE_TOPK * sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};

    // Build the ordered unique miss union without walking all 4*2048 TopK
    // entries on the scalar pipeline.  Vector instructions decode each row
    // and compact only its misses; the scalar bitset work is therefore
    // O(query misses), about 4*200 entries for the target workload.
    Duplicate(unionBits, 0U, activeUnionWords);
    PipeBarrier<PIPE_V>();
    {
        LocalTensor<uint32_t> rawSlots =
            destinationSlots.template ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> missMask =
            destinationSlots[BASE_TOPK].template ReinterpretCast<uint8_t>();
        LocalTensor<int32_t> rowMissTokens = destinationSlots[BASE_TOPK * 2U];
        AscendC::GatherMaskParams compactParams;
        compactParams.repeatTimes = 1;
        compactParams.src0BlockStride = 1;
        compactParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
        compactParams.src1RepeatStride = B32_VEC_REPEAT_STRIDE;

        for (uint32_t queryIdx = 0; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
            uint64_t rowOffset =
                static_cast<uint64_t>(info.bIdx * MTP_QUERY_COUNT + queryIdx) * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopyPad(topkPayloads, mtpTopkPayloadsGm[rowOffset], copyIn, intPad);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            ShiftRight(
                rawSlots,
                topkPayloads.template ReinterpretCast<uint32_t>(),
                INDEX_BITS, BASE_TOPK);
            PipeBarrier<PIPE_V>();
            DecodeIndexFromPayload(
                topkPayloads.template ReinterpretCast<uint32_t>(),
                topkPayloads.template ReinterpretCast<uint32_t>(), BASE_TOPK);
            CompareScalar(missMask,
                          rawSlots.template ReinterpretCast<int32_t>(),
                          INVALID_SLOT14,
                          AscendC::CMPMODE::EQ, BASE_TOPK);
            PipeBarrier<PIPE_V>();

            uint64_t rowMissCount = 0;
            GatherMask(rowMissTokens, topkPayloads,
                       missMask.template ReinterpretCast<uint32_t>(), true,
                       BASE_TOPK, compactParams, rowMissCount);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t rowMissIdx = 0;
                 rowMissIdx < static_cast<uint32_t>(rowMissCount);
                 ++rowMissIdx) {
                uint32_t token =
                    static_cast<uint32_t>(rowMissTokens.GetValue(rowMissIdx));
                if (token >= info.actS2Size) {
                    continue;
                }
                uint32_t wordIdx = token >> 5U;
                uint32_t mask = 1U << (token & 31U);
                uint32_t word = unionBits.GetValue(wordIdx);
                if ((word & mask) == 0U) {
                    unionBits.SetValue(wordIdx, word | mask);
                    missTokens.SetValue(missCount++, static_cast<int32_t>(token));
                }
            }
        }
    }

    uint32_t candidateCap = 0;
    if (missCount > 0U) {
        candidateCap = CeilDiv(missCount, S2_BASE_SIZE) * S2_BASE_SIZE;
        candidateCap = Min(candidateCap, MTP_EVICT_PRELOAD_CAP);
        // q3 incrementally retained the global lowest-score cached entries in
        // evictCandidateUb_.  Finalization consumes at most that 512-entry
        // prefix; atypical larger miss sets continue through the exact GM
        // fallback without scanning aggregateScoresGm end to end.
    }

    uint32_t updateCount = 0;
    LocalTensor<uint32_t> candidateBits =
        evictCandidateUb_.template ReinterpretCast<uint32_t>();
    bool safeCandidatePrefix = (missCount == 0U);
    if (missCount > 0U && missCount <= candidateCap) {
        float minThreshold = mtpThresholdsGm.GetValue(info.bIdx * MTP_QUERY_COUNT);
        bool thresholdsValid = (minThreshold == minThreshold);
        for (uint32_t queryIdx = 1U; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
            float threshold = mtpThresholdsGm.GetValue(
                info.bIdx * MTP_QUERY_COUNT + queryIdx);
            thresholdsValid = thresholdsValid && (threshold == threshold);
            minThreshold = threshold < minThreshold ? threshold : minThreshold;
        }
        const float safeStopKey = -minThreshold;
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        // Candidates are sorted by -aggregate_score in descending order.
        // Therefore the last element of the requested prefix proves the
        // complete prefix safe; checking every entry would add O(miss_count)
        // scalar work for no additional information.
        uint32_t lastCandidateIdx = missCount - 1U;
        float lastCandidateKey = evictCandidateUb_.GetValue(
            lastCandidateIdx * VALUE_AND_INDEX_NUM);
        uint32_t lastPayload = candidateBits.GetValue(
            lastCandidateIdx * VALUE_AND_INDEX_NUM + 1U);
        uint32_t lastToken = lastPayload & INDEX_MASK;
        int32_t lastSlot = static_cast<int32_t>(lastPayload >> INDEX_BITS);
        // Strict inequality keeps threshold ties on the exact path.
        safeCandidatePrefix = thresholdsValid &&
            lastCandidateKey > safeStopKey && lastSlot != INVALID_SLOT14 &&
            lastToken < info.actS2Size &&
            static_cast<uint32_t>(lastSlot) < info.cacheTokenCount;
    }

    if (safeCandidatePrefix) {
        // Every selected victim scores below all four TopK thresholds, so it
        // cannot belong to their union.  The candidate order is unchanged
        // from the exact path; only its 8192-entry membership scan is skipped.
        for (; updateCount < missCount; ++updateCount) {
            uint32_t payload = candidateBits.GetValue(
                updateCount * VALUE_AND_INDEX_NUM + 1U);
            uint32_t evictToken = payload & INDEX_MASK;
            int32_t evictSlot = static_cast<int32_t>(payload >> INDEX_BITS);
            uint32_t missToken =
                static_cast<uint32_t>(missTokens.GetValue(updateCount));
            cacheSlotsGm.SetValue(cacheBase + evictToken,
                                  LICommon::ConstInfo::INVALID_IDX);
            cacheSlotsGm.SetValue(cacheBase + missToken, evictSlot);
            destinationSlots.SetValue(updateCount, evictSlot);
        }
    } else {
        // Ties or an unsafe candidate require the original exact membership
        // path.  Reconstruct the complete union and retain the same candidate
        // and fallback ordering as before the fast path.
        Duplicate(unionBits, 0U, activeUnionWords);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t queryIdx = 0; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
            uint64_t rowOffset =
                static_cast<uint64_t>(info.bIdx * MTP_QUERY_COUNT + queryIdx) * BASE_TOPK;
            DataCopyPad(topkPayloads, mtpTopkPayloadsGm[rowOffset], copyIn, intPad);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (uint32_t topkIdx = 0; topkIdx < BASE_TOPK; ++topkIdx) {
                uint32_t token = static_cast<uint32_t>(
                    topkPayloads.GetValue(topkIdx)) & INDEX_MASK;
                if (token >= info.actS2Size) {
                    continue;
                }
                uint32_t wordIdx = token >> 5U;
                uint32_t mask = 1U << (token & 31U);
                unionBits.SetValue(wordIdx,
                                   unionBits.GetValue(wordIdx) | mask);
            }
        }

        uint32_t candidateCursor = 0;
        uint32_t fallbackCursor = 0;
        while (updateCount < missCount) {
            uint32_t evictToken = 0;
            int32_t evictSlot = LICommon::ConstInfo::INVALID_IDX;
            bool found = false;
            while (candidateCursor < candidateCap) {
                uint32_t payload = candidateBits.GetValue(
                    candidateCursor * VALUE_AND_INDEX_NUM + 1U);
                ++candidateCursor;
                uint32_t token = payload & INDEX_MASK;
                int32_t slot = static_cast<int32_t>(payload >> INDEX_BITS);
                if (slot == INVALID_SLOT14 || token >= info.actS2Size ||
                    static_cast<uint32_t>(slot) >= info.cacheTokenCount) {
                    continue;
                }
                uint32_t unionWord = unionBits.GetValue(token >> 5U);
                if ((unionWord & (1U << (token & 31U))) != 0U) {
                    continue;
                }
                evictToken = token;
                evictSlot = slot;
                found = true;
                break;
            }
            while (!found && fallbackCursor < info.actS2Size) {
                uint32_t token = fallbackCursor++;
                int32_t slot = cacheSlotsGm.GetValue(cacheBase + token);
                if (slot < 0 || static_cast<uint32_t>(slot) >= info.cacheTokenCount) {
                    continue;
                }
                uint32_t unionWord = unionBits.GetValue(token >> 5U);
                if ((unionWord & (1U << (token & 31U))) != 0U) {
                    continue;
                }
                evictToken = token;
                evictSlot = slot;
                found = true;
            }
            if (!found) {
                break;
            }

            uint32_t missToken =
                static_cast<uint32_t>(missTokens.GetValue(updateCount));
            cacheSlotsGm.SetValue(cacheBase + evictToken,
                                  LICommon::ConstInfo::INVALID_IDX);
            cacheSlotsGm.SetValue(cacheBase + missToken, evictSlot);
            destinationSlots.SetValue(updateCount, evictSlot);
            ++updateCount;
        }
    }
    LocalTensor<int32_t> unionScratch =
        unionStorage.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> allPositions = unionScratch;
    LocalTensor<int32_t> missPositions = unionScratch[BASE_TOPK];
    LocalTensor<int32_t> invalidSource = unionScratch[BASE_TOPK * 2U];
    LocalTensor<uint8_t> missMask =
        unionScratch[BASE_TOPK * 3U].template ReinterpretCast<uint8_t>();
    LocalTensor<uint32_t> missSlotMap =
        unionScratch[BASE_TOPK * 3U + BASE_TOPK / 32U]
            .template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> rowMissTokens =
        missStorage.template ReinterpretCast<int32_t>();

    if (updateCount > 0U) {
        AscendC::DataCopyParams copyOut{
            1, static_cast<uint16_t>(updateCount * sizeof(int32_t)), 0, 0};
        uint64_t missOffset = static_cast<uint64_t>(info.bIdx) * MTP_UNION_CAPACITY;
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(mtpMissSourceIdsGm[missOffset], missTokens, copyOut);
        DataCopyPad(mtpMissDestinationSlotsGm[missOffset], destinationSlots, copyOut);
    }

    // The same unique miss can occur in multiple query TopK rows.  Resolve
    // those repeated occurrences through a compact UB hash table instead of
    // issuing one random cacheSlotsGm read per occurrence.  Large/atypical
    // miss sets retain the exact GM lookup path.
    uint32_t missSlotMapCapacity = 0U;
    if (updateCount > 0U && updateCount <= MTP_MISS_SLOT_MAP_MAX_ITEMS) {
        missSlotMapCapacity = 64U;
        while (missSlotMapCapacity < updateCount * 2U) {
            missSlotMapCapacity <<= 1U;
        }
        Duplicate(missSlotMap, 0xffffffffU, missSlotMapCapacity);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        const uint32_t mapMask = missSlotMapCapacity - 1U;
        for (uint32_t missIdx = 0; missIdx < updateCount; ++missIdx) {
            uint32_t token =
                static_cast<uint32_t>(missTokens.GetValue(missIdx));
            uint32_t slot =
                static_cast<uint32_t>(destinationSlots.GetValue(missIdx));
            uint32_t packed = (slot << INDEX_BITS) | token;
            uint32_t mapIdx = (token * 0x9e3779b1U) & mapMask;
            while (missSlotMap.GetValue(mapIdx) != 0xffffffffU) {
                mapIdx = (mapIdx + 1U) & mapMask;
            }
            missSlotMap.SetValue(mapIdx, packed);
        }
    }

    // The hash uses disjoint UB storage, so its construction overlaps the
    // two miss-array copies above.  The scalar miss-count copy now acts as the
    // common MTE3 completion point before any source buffer is overwritten.
    topkPayloads.SetValue(0, static_cast<int32_t>(updateCount));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(missCountGm[info.bIdx], topkPayloads, 1);
    // topkPayloads is refilled through MTE2 next.  A direct MTE3->MTE2
    // dependency is sufficient here and avoids WriteMissCount's additional
    // MTE3->V round trip.
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    // Resolve both caller-visible rows from the original packed payload.
    // Decode all 2048 payloads with vector instructions, compact only the
    // original miss positions/tokens, then scalar-patch those O(row misses)
    // slots from the updated cache map. This keeps the scalar work near the
    // typical ~200 misses/query instead of scanning 2048 entries/query.

    ArithProgression(allPositions, 0, 1, BASE_TOPK);
    Duplicate(invalidSource, LICommon::ConstInfo::INVALID_IDX, BASE_TOPK);
    PipeBarrier<PIPE_V>();

    AscendC::GatherMaskParams compactParams;
    compactParams.repeatTimes = 1;
    compactParams.src0BlockStride = 1;
    compactParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    compactParams.src1RepeatStride = B32_VEC_REPEAT_STRIDE;
    AscendC::DataCopyParams topkCopy{
        1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0};
    LocalTensor<int32_t> slotPong = destinationSlots[BASE_TOPK];
    LocalTensor<int32_t> sourcePong = destinationSlots[BASE_TOPK * 2U];
    for (uint32_t queryIdx = 0; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
        if (queryIdx == 2U) {
            // q0/q1 used the two ping-pong output pairs.  Wait only when q2
            // is about to reuse q0's pair, after q1 computation has already
            // overlapped q0's MTE3 output.
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
        LocalTensor<int32_t> rowSlots =
            (queryIdx & 1U) == 0U ? destinationSlots : slotPong;
        LocalTensor<int32_t> rowSources =
            (queryIdx & 1U) == 0U ? topkPayloads : sourcePong;
        uint64_t rowOffset =
            static_cast<uint64_t>(info.bIdx * MTP_QUERY_COUNT + queryIdx) * BASE_TOPK;
        DataCopyPad(rowSources, mtpTopkPayloadsGm[rowOffset], copyIn, intPad);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        ShiftRight(
            rowSlots.template ReinterpretCast<uint32_t>(),
            rowSources.template ReinterpretCast<uint32_t>(),
            INDEX_BITS, BASE_TOPK);
        PipeBarrier<PIPE_V>();
        DecodeIndexFromPayload(
            rowSources.template ReinterpretCast<uint32_t>(),
            rowSources.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        CompareScalar(missMask, rowSlots,
                      INVALID_SLOT14,
                      AscendC::CMPMODE::EQ, BASE_TOPK);
        PipeBarrier<PIPE_V>();

        uint64_t rowMissCount = 0;
        GatherMask(missPositions, allPositions,
                   missMask.template ReinterpretCast<uint32_t>(), true,
                   BASE_TOPK, compactParams, rowMissCount);
        PipeBarrier<PIPE_V>();
        uint64_t tokenMissCount = 0;
        GatherMask(rowMissTokens, rowSources,
                   missMask.template ReinterpretCast<uint32_t>(), true,
                   BASE_TOPK, compactParams, tokenMissCount);
        PipeBarrier<PIPE_V>();
        // C220 vsel has no int32 tensor overload. Reinterpret the int32
        // payloads as float so vsel copies the same 32-bit lanes without a
        // numeric conversion.
        Select(rowSources.template ReinterpretCast<float>(), missMask,
               rowSources.template ReinterpretCast<float>(),
               invalidSource.template ReinterpretCast<float>(),
               AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, BASE_TOPK);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

        uint32_t patchCount = static_cast<uint32_t>(
            Min(rowMissCount, tokenMissCount));
        for (uint32_t missIdx = 0; missIdx < patchCount; ++missIdx) {
            uint32_t position =
                static_cast<uint32_t>(missPositions.GetValue(missIdx));
            uint32_t token =
                static_cast<uint32_t>(rowMissTokens.GetValue(missIdx));
            if (position >= BASE_TOPK || token >= info.actS2Size) {
                continue;
            }
            int32_t resolvedSlot = LICommon::ConstInfo::INVALID_IDX;
            if (missSlotMapCapacity > 0U) {
                const uint32_t mapMask = missSlotMapCapacity - 1U;
                uint32_t mapIdx = (token * 0x9e3779b1U) & mapMask;
                for (uint32_t probe = 0; probe < missSlotMapCapacity; ++probe) {
                    uint32_t packed = missSlotMap.GetValue(mapIdx);
                    if (packed == 0xffffffffU) {
                        break;
                    }
                    if ((packed & INDEX_MASK) == token) {
                        resolvedSlot = static_cast<int32_t>(packed >> INDEX_BITS);
                        break;
                    }
                    mapIdx = (mapIdx + 1U) & mapMask;
                }
            }
            if (resolvedSlot == LICommon::ConstInfo::INVALID_IDX) {
                resolvedSlot = cacheSlotsGm.GetValue(cacheBase + token);
            }
            rowSlots.SetValue(position, resolvedSlot);
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(topkSlotsGm[rowOffset], rowSlots, topkCopy);
        DataCopyPad(mtpTopkSourceIdsGm[rowOffset], rowSources, topkCopy);
    }
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

    outQueue_.FreeTensor(slotStorage);
    inQueue_.FreeTensor(missStorage);
    inQueue_.FreeTensor(unionStorage);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVecMtp(const LICommon::RunInfo &info)
{
    if ((GetBlockIdx() & 1U) != 0U) {
        return;
    }

    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;
    int32_t cuS2Len = info.actualSingleProcessSInnerSize;
    int64_t mmGmOffset = (info.loop % 2) * (gSize_ * s2BaseSize_);
    int64_t weightGmOffset = static_cast<int64_t>(info.queryRow) * gSize_;
    if (info.isFirstS2InnerLoop) {
        InitSortOutBuf(globalTopkUb_, TOPK_PAIR_FLOATS);
    }

    int32_t mmUbStride =
        (s2BaseSize_ - info.actualSingleProcessSInnerSizeAlign) /
        B32_BLOCK_ALIGN_NUM;
    int64_t payloadBufIdx = info.s2Idx % PAYLOAD_BUF_SLOTS;
    LocalTensor<int32_t> payloadUb =
        payloadBuf_.Get<int32_t>()[payloadBufIdx * s2BaseSize_];
    StartPayloadCopy(payloadUb, info.cacheRowIdx, cuBaseS2Idx, cuS2Len,
                     s2BaseSize_);
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    PipeBarrier<PIPE_V>();
    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
    for (uint32_t outerGidx = 0; outerGidx < outerG_; ++outerGidx) {
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb =
            mmInUb[GROUP_INNER * s2BaseSize_];
        LocalTensor<K_T> weightsInTUb =
            weightsInUb.template ReinterpretCast<K_T>();
        weightsInTUb = weightsInTUb[GROUP_INNER];
        LIServiceVec::CopyIn(
            mmInUb, weightsInTUb, mm1ResGm, weightsGm,
            mmGmOffset + outerGidx * GROUP_INNER *
                             info.actualSingleProcessSInnerSizeAlign,
            weightGmOffset + outerGidx * GROUP_INNER, GROUP_INNER,
            info.actualSingleProcessSInnerSizeAlign, mmUbStride);
        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        weightsInUb = mmInUb[GROUP_INNER * s2BaseSize_];
        LIServiceVec::DoScale(
            reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb,
            weightsInTUb, brcBuf, GROUP_INNER, s2BaseSize_, outerGidx);
        inQueue_.FreeTensor(mmInUb);
    }
    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM],
                           reduceOutInner, GROUP_INNER, s2BaseSize_);
    outQueue_.FreeTensor(reduceCacheBuf);

    LocalTensor<float> sortScoreUb = reduceOutBuff;
    PipeBarrier<PIPE_V>();
    Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(),
              LIServiceVec::NEG_INF, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
    PipeBarrier<PIPE_V>();
    FinishPayload(payloadUb, cuBaseS2Idx, cuS2Len);
    WriteMtpAggregateScoreChunk(info.bIdx, info.queryIdx, cuBaseS2Idx,
                                sortScoreUb, s2BaseSize_);

    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    uint32_t cachedChunkIdx = info.segmentChunkIdx % PAYLOAD_BUF_SLOTS;
    if (info.queryIdx + 1U == MTP_QUERY_COUNT) {
        CollectMtpEvictCandidateChunk(info, payloadUb, tmpSortBuf,
                                      cachedChunkIdx);
    }
    Sort<float, true>(
        SortedBasicBlock_[cachedChunkIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM],
        reduceOutBuff, payloadUb.template ReinterpretCast<uint32_t>(),
        tmpSortBuf, s2BaseSize_ / 32);
    PipeBarrier<PIPE_V>();
    if (cachedChunkIdx == 3U || info.isLastS2InnerLoop) {
        if (info.segmentChunkIdx < PAYLOAD_BUF_SLOTS) {
            MrgBasicBlock(globalTopkUb_, SortedBasicBlock_,
                          static_cast<int64_t>(cachedChunkIdx + 1U),
                          s2BaseSize_);
        } else {
            if (cachedChunkIdx > 0U) {
                MrgBasicBlock(tmpSortBuf, SortedBasicBlock_,
                              static_cast<int64_t>(cachedChunkIdx + 1U),
                              s2BaseSize_);
                PipeBarrier<PIPE_V>();
                DataCopy(SortedBasicBlock_, tmpSortBuf,
                         (cachedChunkIdx + 1U) * s2BaseSize_ *
                             VALUE_AND_INDEX_NUM);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       s2BaseSize_ * (cachedChunkIdx + 1U));
        }
    }
    PipeBarrier<PIPE_V>();
    outQueue_.FreeTensor(tmpSortBuf);

    if (info.isLastS2InnerLoop) {
        StoreMtpQueryTopK(info);
        if (info.queryIdx + 1U == MTP_QUERY_COUNT) {
            FinalizeMtpRequest(info);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }

    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;
    int32_t cuS2Len = info.actualSingleProcessSInnerSize;
    int64_t mmGmOffset = (info.loop % 2) * (gSize_ * s2BaseSize_);
    int64_t weightGmOffset = static_cast<int64_t>(info.bIdx) * gSize_;
    if (info.isFirstS2InnerLoop) {
        InitSortOutBuf(globalTopkUb_, TOPK_PAIR_FLOATS);
    }

    int32_t mmUbStride = (s2BaseSize_ - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
    int64_t payloadBufIdx = info.s2Idx % PAYLOAD_BUF_SLOTS;
    LocalTensor<int32_t> payloadUb = payloadBuf_.Get<int32_t>()[payloadBufIdx * s2BaseSize_];
    StartPayloadCopy(payloadUb, info.cacheRowIdx, cuBaseS2Idx, cuS2Len, s2BaseSize_);
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    PipeBarrier<PIPE_V>();
    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
    for (uint32_t outerGidx = 0; outerGidx < outerG_; ++outerGidx) {
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[GROUP_INNER * s2BaseSize_];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>();
        weightsInTUb = weightsInTUb[GROUP_INNER];
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                             mmGmOffset + outerGidx * GROUP_INNER * info.actualSingleProcessSInnerSizeAlign,
                             weightGmOffset + outerGidx * GROUP_INNER, GROUP_INNER,
                             info.actualSingleProcessSInnerSizeAlign, mmUbStride);
        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        weightsInUb = mmInUb[GROUP_INNER * s2BaseSize_];
        LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                              brcBuf, GROUP_INNER, s2BaseSize_, outerGidx);
        inQueue_.FreeTensor(mmInUb);
    }

    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, GROUP_INNER, s2BaseSize_);
    outQueue_.FreeTensor(reduceCacheBuf);

    LocalTensor<float> sortScoreUb = reduceOutBuff;
    PipeBarrier<PIPE_V>();
    bool hasLongIndexTag = info.actS2Size > EXACT_PACKED_SOURCE_TOKENS;
    PrepareSortScore(sortScoreUb, reduceOutInner, cuBaseS2Idx, cuS2Len, hasLongIndexTag);
    WriteScoreChunk(info.bIdx, cuBaseS2Idx, sortScoreUb, s2BaseSize_);
    FinishPayload(payloadUb, cuBaseS2Idx, cuS2Len);
    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    uint32_t cachedChunkIdx = info.segmentChunkIdx % PAYLOAD_BUF_SLOTS;
    Sort<float, true>(SortedBasicBlock_[cachedChunkIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM], reduceOutBuff,
                      payloadUb.template ReinterpretCast<uint32_t>(), tmpSortBuf, s2BaseSize_ / 32);
    PipeBarrier<PIPE_V>();
    if (cachedChunkIdx == 3 || info.isLastS2InnerLoop) {
        if (info.segmentChunkIdx < PAYLOAD_BUF_SLOTS) {
            MrgBasicBlock(globalTopkUb_, SortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), s2BaseSize_);
        } else {
            if (cachedChunkIdx > 0) {
                MrgBasicBlock(tmpSortBuf, SortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), s2BaseSize_);
                PipeBarrier<PIPE_V>();
                DataCopy(SortedBasicBlock_, tmpSortBuf, (cachedChunkIdx + 1) * s2BaseSize_ * VALUE_AND_INDEX_NUM);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       s2BaseSize_ * (cachedChunkIdx + 1));
        }
    }
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    if (info.isLastS2InnerLoop && info.isPartialSegment) {
        WritePartialTopK(GetBlockIdx() / 2U, info.partialSlot, info.bIdx);
        outQueue_.FreeTensor(tmpSortBuf);
        return;
    }

    float thresholdScore = 0.0f;
    if (info.isLastS2InnerLoop) {
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        thresholdScore = globalTopkUb_.GetValue((BASE_TOPK - 1) * VALUE_AND_INDEX_NUM);
        SortTopKBySlot(globalTopkUb_, reduceOutBuff, payloadBuf_.Get<int32_t>(), tmpSortBuf, hasLongIndexTag);
    }
    outQueue_.FreeTensor(tmpSortBuf);

    if (info.isLastS2InnerLoop) {
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<uint32_t> payloadLocal = valueULocal.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
        LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
        LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 3];
        int64_t outOffset = static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY;
        uint32_t missCount = CopyDecodedPayloadOut(
            outOffset, globalTopkUb_, payloadLocal, indexLocal, slotLocal, scratchLocal,
            BASE_TOPK, hasLongIndexTag, info.actS2Size < BASE_TOPK);
        UpdateCacheAndWriteTopkSlots(
            info.bIdx, info.cacheRowIdx, outOffset, info.actS2Size,
            info.cacheTokenCount, info.actS2Size, thresholdScore, missCount,
            indexLocal, payloadLocal, scratchLocal, slotLocal);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FinalizePartialRequest(uint32_t bIdx, uint32_t cacheRowIdx,
                                                             uint32_t actualSeqLen,
                                                             uint32_t cacheTokenCount,
                                                             uint32_t firstOwner, uint32_t lastOwner)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }

    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    bool hasPartial = false;
    for (uint32_t coreIdx = firstOwner; coreIdx <= lastOwner; ++coreIdx) {
        for (uint32_t partialSlot = 0; partialSlot < PARTIAL_SLOTS_PER_CORE; ++partialSlot) {
            uint32_t resultSlot = coreIdx * PARTIAL_SLOTS_PER_CORE + partialSlot;
            uint32_t metaOffset = coreIdx * PARTIAL_META_INTS_PER_CORE + partialSlot;
            if (partialMetaGm.GetValue(metaOffset) != static_cast<int32_t>(bIdx)) {
                continue;
            }

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(SortedBasicBlock_,
                        partialTopkGm[static_cast<uint64_t>(resultSlot) * TOPK_PAIR_FLOATS],
                        AscendC::DataCopyExtParams{
                            1, static_cast<uint32_t>(TOPK_PAIR_FLOATS * sizeof(float)), 0, 0, 0},
                        AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            if (!hasPartial) {
                DataCopy(globalTopkUb_, SortedBasicBlock_, TOPK_PAIR_FLOATS);
                hasPartial = true;
            } else {
                SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK, BASE_TOPK);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    if (!hasPartial) {
        outQueue_.FreeTensor(tmpSortBuf);
        return;
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    float thresholdScore = globalTopkUb_.GetValue((BASE_TOPK - 1U) * VALUE_AND_INDEX_NUM);
    bool hasLongIndexTag = actualSeqLen > EXACT_PACKED_SOURCE_TOKENS;
    SortTopKBySlot(globalTopkUb_, reduceOutBuf_.Get<float>(), payloadBuf_.Get<int32_t>(), tmpSortBuf,
                   hasLongIndexTag);
    outQueue_.FreeTensor(tmpSortBuf);

    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> payloadLocal = valueULocal.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
    LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 3];
    int64_t outOffset = static_cast<int64_t>(bIdx) * OUTPUT_CAPACITY;
    uint32_t missCount = CopyDecodedPayloadOut(
        outOffset, globalTopkUb_, payloadLocal, indexLocal, slotLocal, scratchLocal,
        BASE_TOPK, hasLongIndexTag, false);
    UpdateCacheAndWriteTopkSlots(
        bIdx, cacheRowIdx, outOffset, actualSeqLen, cacheTokenCount,
        actualSeqLen, thresholdScore, missCount, indexLocal, payloadLocal,
        scratchLocal, slotLocal);
    outQueue_.FreeTensor(valueULocal);
}

} // namespace LIKernel
#endif
