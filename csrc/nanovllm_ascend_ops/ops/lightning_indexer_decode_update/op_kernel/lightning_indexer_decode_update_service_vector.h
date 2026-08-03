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
 * \file lightning_indexer_decode_update_service_vector.h
 * \brief
 */
#ifndef lightning_indexer_decode_update_SERVICE_VECTOR_H
#define lightning_indexer_decode_update_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_decode_update_common.h"
#include "lightning_indexer_decode_update_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t OUTPUT_CAPACITY = 2048;
constexpr uint32_t S2_BASE_SIZE = 512;
constexpr uint32_t GROUP_INNER = 16;
constexpr uint32_t PAYLOAD_BUF_SLOTS = 4;
constexpr uint32_t EVICT_CANDIDATE_CAP = BASE_TOPK;
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
constexpr uint32_t MTP_SOURCE_CAPACITY = 1U << 18;
constexpr uint32_t MTP_UNION_BITSET_WORDS = MTP_SOURCE_CAPACITY / 32U;

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
        GlobalTensor<int32_t> missSourceIdsGm,
        GlobalTensor<int32_t> missDestinationSlotsGm,
        GlobalTensor<int32_t> missCountGm,
        GlobalTensor<float> scoresGm,
        GlobalTensor<int32_t> mtpTopkIndicesGm);
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
    GlobalTensor<int32_t> mtpTopkIndicesGm;
    GlobalTensor<int32_t> mtpMissSourceIdsGm;
    GlobalTensor<int32_t> mtpMissDestinationSlotsGm;

private:
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
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
                                            int32_t s2BaseIdx,
                                            int32_t validLen, int32_t alignedLen);
    __aicore__ inline void FinishPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx, int32_t validLen);
    __aicore__ inline void SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                          const LocalTensor<float> &keyLocal,
                                          const LocalTensor<int32_t> &payloadBase,
                                          LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void WriteScoreChunk(uint32_t bIdx, int32_t s2BaseIdx,
                                           const LocalTensor<float> &scoreLocal, int32_t alignedLen);
    __aicore__ inline void WriteMtpAggregateScoreChunk(
        uint32_t bIdx, uint32_t queryIdx, int32_t s2BaseIdx,
        const LocalTensor<float> &scoreLocal, int32_t alignedLen);
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
    __aicore__ inline uint32_t CopyDecodedPayloadOut(int64_t outOffset, const LocalTensor<uint32_t> &payloadLocal,
                                                     const LocalTensor<int32_t> &indexLocal,
                                                     const LocalTensor<int32_t> &slotLocal,
                                                     const LocalTensor<int32_t> &scratchLocal,
                                                     int64_t count, bool mayHaveInvalid);
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
    GlobalTensor<int32_t> missSourceIdsGm,
    GlobalTensor<int32_t> missDestinationSlotsGm,
    GlobalTensor<int32_t> missCountGm,
    GlobalTensor<float> scoresGm,
    GlobalTensor<int32_t> mtpTopkIndicesGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->mtpMissSourceIdsGm = missSourceIdsGm;
    this->mtpMissDestinationSlotsGm = missDestinationSlotsGm;
    this->missCountGm = missCountGm;
    this->scoresGm = scoresGm;
    this->mtpTopkIndicesGm = mtpTopkIndicesGm;
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
    Adds(payloadLocal, payloadLocal, s2BaseIdx, validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                                     const LocalTensor<float> &keyLocal,
                                                     const LocalTensor<int32_t> &payloadBase,
                                                     LocalTensor<float> &tmpSortBuf)
{
    for (uint32_t blockIdx = 0; blockIdx < PAYLOAD_BUF_SLOTS; ++blockIdx) {
        uint32_t pairOffset = blockIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM;
        LocalTensor<int32_t> payloadLocal = payloadBase[blockIdx * s2BaseSize_];
        ExtractIndex(payloadLocal.template ReinterpretCast<uint32_t>(),
                     pairLocal[pairOffset].template ReinterpretCast<uint32_t>(), s2BaseSize_);
        BuildHitMissKey(keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), s2BaseSize_);
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
    if (queryIdx == 0U) {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LIServiceVec::CopyOut(scoresGm[gmOffset], scoreLocal, alignedLen);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        return;
    }

    LocalTensor<float> previousScore = globalTopkIndice_.template ReinterpretCast<float>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopyPad(previousScore, scoresGm[gmOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(alignedLen * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Max(previousScore, previousScore, scoreLocal, alignedLen);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(scoresGm[gmOffset], previousScore, alignedLen);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    // FinishPayload needs this 0..511 progression for every following chunk.
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, S2_BASE_SIZE);
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

    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(), payloadLocal.template ReinterpretCast<uint32_t>(),
              INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal, static_cast<int32_t>(s2BaseIdx), validLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> keyLocal = reduceOutBuf_.Get<float>()[s2BaseSize_];
    BuildEvictCandidateKeyFromPayload(keyLocal, scoreLocal, payloadLocal.template ReinterpretCast<uint32_t>(),
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
__aicore__ inline uint32_t LIVector<LIT>::CopyDecodedPayloadOut(int64_t outOffset,
                                                                const LocalTensor<uint32_t> &payloadLocal,
                                                                const LocalTensor<int32_t> &indexLocal,
                                                                const LocalTensor<int32_t> &slotLocal,
                                                                const LocalTensor<int32_t> &scratchLocal,
                                                                int64_t count, bool mayHaveInvalid)
{
    DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(), payloadLocal, BASE_TOPK);
    if (mayHaveInvalid) {
        FixInvalidIndex(indexLocal, payloadLocal.template ReinterpretCast<int32_t>(), BASE_TOPK);
    }
    DecodeSlotFromPayload(slotLocal.template ReinterpretCast<uint32_t>(), payloadLocal, scratchLocal, BASE_TOPK);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    uint32_t missCount = 0;
    while (missCount < static_cast<uint32_t>(count) &&
           slotLocal.GetValue(missCount) == LICommon::ConstInfo::INVALID_IDX) {
        ++missCount;
    }
    if (missCount > 0) {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(topkIndexGm[outOffset], indexLocal,
                    {1, static_cast<uint16_t>(missCount * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
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
    const LocalTensor<int32_t> &indexLocal, const LocalTensor<int32_t> &topkSlotsLocal)
{
    uint32_t candidateCap = 0;
    if (missCount > 0) {
        candidateCap = ((missCount + S2_BASE_SIZE - 1U) / S2_BASE_SIZE) * S2_BASE_SIZE;
        if (candidateCap > EVICT_CANDIDATE_CAP) {
            candidateCap = EVICT_CANDIDATE_CAP;
        }
        FindEvictCandidates(bIdx, cacheRowIdx, actualSeqLen, scanSeed, missCount, candidateCap,
                            thresholdScore, SortedBasicBlock_);
    }

    uint32_t candidateCursor = 0;
    uint32_t fallbackCursor = 0;
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    float stopKey = -thresholdScore;
    LocalTensor<uint32_t> candidateBitsLocal = evictCandidateUb_.template ReinterpretCast<uint32_t>();
    // Validate each packed candidate before publishing its destination slot.
    // The previous bulk decode could expose the packed invalid-slot sentinel
    // (16383) to the fused Attention+SCATTER consumer.
    for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
        uint32_t evictIndex = 0;
        int32_t evictSlot = LICommon::ConstInfo::INVALID_IDX;
        bool foundCandidate = false;
        while (candidateCursor < candidateCap) {
            float candidateKey = evictCandidateUb_.GetValue(candidateCursor * VALUE_AND_INDEX_NUM);
            if (candidateKey <= stopKey) {
                break;
            }
            uint32_t payload = candidateBitsLocal.GetValue(candidateCursor * VALUE_AND_INDEX_NUM + 1);
            ++candidateCursor;
            int32_t slot = static_cast<int32_t>(payload >> INDEX_BITS);
            uint32_t index = payload & INDEX_MASK;
            if (slot != INVALID_SLOT14 &&
                static_cast<uint32_t>(slot) < cacheTokenCount &&
                index < actualSeqLen) {
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
            topkSlotsLocal.SetValue(missIdx, evictSlot);
            cacheSlotsGm.SetValue(rowBase + evictIndex, LICommon::ConstInfo::INVALID_IDX);
            cacheSlotsGm.SetValue(rowBase + missIndex, evictSlot);
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
    LocalTensor<int32_t> indexLocal =
        valueLocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
    PipeBarrier<PIPE_V>();
    DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(),
                           payloadLocal, BASE_TOPK);
    PipeBarrier<PIPE_V>();

    uint64_t rowOffset = static_cast<uint64_t>(info.queryRow) * BASE_TOPK;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(mtpTopkIndicesGm[rowOffset], indexLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(valueLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FinalizeMtpRequest(const LICommon::RunInfo &info)
{
    // Two VECIN buffers hold the 2^18-token membership bitset and the ordered
    // union misses.  The VECOUT buffer first stages destination slots, then is
    // reused to materialize the four per-query sparse-slot rows.
    LocalTensor<float> unionStorage = inQueue_.AllocTensor<float>();
    LocalTensor<float> missStorage = inQueue_.AllocTensor<float>();
    LocalTensor<float> slotStorage = outQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> unionBits = unionStorage.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> missTokens = missStorage.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> destinationSlots = slotStorage.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> topkIndices = payloadBuf_.Get<int32_t>();

    Duplicate(unionBits, 0U, MTP_UNION_BITSET_WORDS);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * cacheSlotsSize_;
    uint32_t missCount = 0;
    AscendC::DataCopyExtParams copyIn{1, BASE_TOPK * sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};
    for (uint32_t queryIdx = 0; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
        uint64_t rowOffset =
            static_cast<uint64_t>(info.bIdx * MTP_QUERY_COUNT + queryIdx) * BASE_TOPK;
        DataCopyPad(topkIndices, mtpTopkIndicesGm[rowOffset], copyIn, intPad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t topkIdx = 0; topkIdx < BASE_TOPK; ++topkIdx) {
            int32_t signedToken = topkIndices.GetValue(topkIdx);
            if (signedToken < 0 || static_cast<uint32_t>(signedToken) >= info.actS2Size) {
                continue;
            }
            uint32_t token = static_cast<uint32_t>(signedToken);
            uint32_t wordIdx = token >> 5U;
            uint32_t mask = 1U << (token & 31U);
            uint32_t word = unionBits.GetValue(wordIdx);
            if ((word & mask) != 0U) {
                continue;
            }
            unionBits.SetValue(wordIdx, word | mask);
            int32_t slot = cacheSlotsGm.GetValue(cacheBase + token);
            if (slot < 0 || static_cast<uint32_t>(slot) >= info.cacheTokenCount) {
                missTokens.SetValue(missCount++, static_cast<int32_t>(token));
            }
        }
    }

    uint32_t candidateCap = 0;
    if (missCount > 0U) {
        candidateCap = CeilDiv(missCount, S2_BASE_SIZE) * S2_BASE_SIZE;
        candidateCap = Min(candidateCap, EVICT_CANDIDATE_CAP);
        // A very high threshold stops the score-guided scan after it has a
        // full candidate prefix plus the configured extra chunks.  ScoresGm
        // contains max(q0..q3), so low aggregate-score cache entries win.
        uint32_t guidedCount = Min(missCount, candidateCap);
        FindEvictCandidates(info.bIdx, info.cacheRowIdx, info.actS2Size,
                            info.actS2Size, guidedCount, candidateCap,
                            3.0e38f, SortedBasicBlock_);
    }

    uint32_t candidateCursor = 0;
    uint32_t fallbackCursor = 0;
    uint32_t updateCount = 0;
    LocalTensor<uint32_t> candidateBits =
        evictCandidateUb_.template ReinterpretCast<uint32_t>();
    while (updateCount < missCount) {
        uint32_t evictToken = 0;
        int32_t evictSlot = LICommon::ConstInfo::INVALID_IDX;
        bool found = false;
        while (candidateCursor < candidateCap) {
            uint32_t payload =
                candidateBits.GetValue(candidateCursor * VALUE_AND_INDEX_NUM + 1U);
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

        uint32_t missToken = static_cast<uint32_t>(missTokens.GetValue(updateCount));
        cacheSlotsGm.SetValue(cacheBase + evictToken, LICommon::ConstInfo::INVALID_IDX);
        cacheSlotsGm.SetValue(cacheBase + missToken, evictSlot);
        destinationSlots.SetValue(updateCount, evictSlot);
        ++updateCount;
    }
    PipeBarrier<PIPE_ALL>();

    if (updateCount > 0U) {
        AscendC::DataCopyParams copyOut{
            1, static_cast<uint16_t>(updateCount * sizeof(int32_t)), 0, 0};
        uint64_t missOffset = static_cast<uint64_t>(info.bIdx) * MTP_UNION_CAPACITY;
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(mtpMissSourceIdsGm[missOffset], missTokens, copyOut);
        DataCopyPad(mtpMissDestinationSlotsGm[missOffset], destinationSlots, copyOut);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
    WriteMissCount(info.bIdx, static_cast<int32_t>(updateCount), topkIndices);

    // Resolve all four top-k rows against the single final cache state.
    AscendC::DataCopyParams topkCopy{
        1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0};
    for (uint32_t queryIdx = 0; queryIdx < MTP_QUERY_COUNT; ++queryIdx) {
        uint64_t rowOffset =
            static_cast<uint64_t>(info.bIdx * MTP_QUERY_COUNT + queryIdx) * BASE_TOPK;
        DataCopyPad(topkIndices, mtpTopkIndicesGm[rowOffset], copyIn, intPad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t topkIdx = 0; topkIdx < BASE_TOPK; ++topkIdx) {
            int32_t token = topkIndices.GetValue(topkIdx);
            int32_t slot = token < 0
                               ? LICommon::ConstInfo::INVALID_IDX
                               : cacheSlotsGm.GetValue(cacheBase + static_cast<uint32_t>(token));
            destinationSlots.SetValue(topkIdx, slot);
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(topkSlotsGm[rowOffset], destinationSlots, topkCopy);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

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
    Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
    PipeBarrier<PIPE_V>();
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
        thresholdScore = globalTopkUb_.GetValue((BASE_TOPK - 1) * VALUE_AND_INDEX_NUM);
        SortTopKBySlot(globalTopkUb_, reduceOutBuff, payloadBuf_.Get<int32_t>(), tmpSortBuf);
    }
    outQueue_.FreeTensor(tmpSortBuf);

    if (info.isLastS2InnerLoop) {
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<uint32_t> payloadLocal = valueULocal.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
        LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
        LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 3];
        ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        PipeBarrier<PIPE_V>();
        int64_t outOffset = static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY;
        uint32_t missCount = CopyDecodedPayloadOut(outOffset, payloadLocal, indexLocal, slotLocal, scratchLocal,
                                                   BASE_TOPK, info.actS2Size < BASE_TOPK);
        UpdateCacheAndWriteTopkSlots(
            info.bIdx, info.cacheRowIdx, outOffset, info.actS2Size,
            info.cacheTokenCount, info.actS2Size, thresholdScore, missCount,
            indexLocal, slotLocal);
        WriteMissCount(info.bIdx, static_cast<int32_t>(missCount), scratchLocal);
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
    SortTopKBySlot(globalTopkUb_, reduceOutBuf_.Get<float>(), payloadBuf_.Get<int32_t>(), tmpSortBuf);
    outQueue_.FreeTensor(tmpSortBuf);

    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> payloadLocal = valueULocal.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
    LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 3];
    ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
    PipeBarrier<PIPE_V>();
    int64_t outOffset = static_cast<int64_t>(bIdx) * OUTPUT_CAPACITY;
    uint32_t missCount =
        CopyDecodedPayloadOut(outOffset, payloadLocal, indexLocal, slotLocal, scratchLocal, BASE_TOPK, false);
    UpdateCacheAndWriteTopkSlots(
        bIdx, cacheRowIdx, outOffset, actualSeqLen, cacheTokenCount,
        actualSeqLen, thresholdScore, missCount, indexLocal, slotLocal);
    WriteMissCount(bIdx, static_cast<int32_t>(missCount), scratchLocal);
    outQueue_.FreeTensor(valueULocal);
}

} // namespace LIKernel
#endif
