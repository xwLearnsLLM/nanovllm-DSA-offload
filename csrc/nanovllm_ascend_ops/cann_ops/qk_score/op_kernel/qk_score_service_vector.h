/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qk_score_service_vector.h
 * \brief vector post-processing for qk_score.
 */
#ifndef QK_SCORE_SERVICE_VECTOR_H
#define QK_SCORE_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "qk_score_common.h"
#include "qk_score_vector.h"

namespace QKKernel {
using namespace QKCommon;
using namespace QKServiceVec;

template <typename QKT>
class QKVector {
public:
    using K_T = typename QKT::keyType;
    static constexpr QK_LAYOUT LAYOUT_T = QKT::layout;

    using MM1_OUT_T = float;

    __aicore__ inline QKVector(){};
    __aicore__ inline void ProcessVec(const QKCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QKCommon::ConstInfo &constInfo,
                                      const QkScoreTilingData *__restrict);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<float> scoreOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset, int64_t cleanCount);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<float> scoreOutGm;

private:
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;

    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct QKCommon::ConstInfo constInfo_;
};

template <typename QKT>
__aicore__ inline void QKVector<QKT>::InitBuffers(TPipe *pipe)
{
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);

    if constexpr (QKT::s2BaseSize > 512) {
        pipe->InitBuffer(inQueue_, 1, groupInner_ * s2BaseSize_ * sizeof(float) + s2BaseSize_ * sizeof(float));
    } else {
        pipe->InitBuffer(inQueue_, 2, groupInner_ * s2BaseSize_ * sizeof(float) + s2BaseSize_ * sizeof(float));
    }
    pipe->InitBuffer(outQueue_, 1, reduceCacheSize);
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
}

template <typename QKT>
__aicore__ inline void QKVector<QKT>::InitParams(const struct QKCommon::ConstInfo &constInfo,
                                                 const QkScoreTilingData *__restrict)
{
    this->constInfo_ = constInfo;
    gSize_ = constInfo.gSize;
    kHeadNum_ = constInfo.kHeadNum;
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    groupInner_ = 16;
    blockId_ = GetBlockIdx();
}

template <typename QKT>
__aicore__ inline void
QKVector<QKT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                    GlobalTensor<float> scoreOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->scoreOutGm = scoreOutGm;
}

template <typename QKT>
__aicore__ inline void QKVector<QKT>::CleanInvalidOutput(int64_t invalidS1offset, int64_t cleanCount)
{
    if (cleanCount <= 0) {
        return;
    }
    LocalTensor<float> invalidLocal = outQueue_.AllocTensor<float>();
    int64_t invalidCopyBase = static_cast<int64_t>(groupInner_) * s2BaseSize_;
    invalidCopyBase = invalidCopyBase > cleanCount ? cleanCount : invalidCopyBase;
    Duplicate(invalidLocal.template ReinterpretCast<int32_t>(), QKServiceVec::NEG_INF, invalidCopyBase);
    outQueue_.EnQue<float>(invalidLocal);
    invalidLocal = outQueue_.DeQue<float>();

    int64_t remaining = cleanCount;
    int64_t offset = invalidS1offset;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    while (remaining > 0) {
        int64_t copyCount = remaining > invalidCopyBase ? invalidCopyBase : remaining;
        QKServiceVec::CopyOut(scoreOutGm[offset], invalidLocal, copyCount, offset);
        offset += copyCount;
        remaining -= copyCount;
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(invalidLocal);
}

template <typename QKT>
__aicore__ inline void QKVector<QKT>::ProcessVec(const QKCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    int64_t mmGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_) * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // Pure decode has one S1 row; split the S2 tile across paired AIVs to avoid leaving the odd AIV idle.
    bool splitS2ForSingleRow = cuS1ProcNum == 1;
    int32_t cuS1ProcNumPerAiv =
        splitS2ForSingleRow ? 1 : (blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2));
    int32_t aivS1Offset = splitS2ForSingleRow ? 0 : (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    cuS1BeginIdxPerAiv += aivS1Offset;

    weightGmOffset += aivS1Offset * kHeadNum_ * gSize_;
    mmGmOffset += aivS1Offset * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    int32_t outerG = CeilDiv(gSize_, groupInner_);
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }

    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        int64_t rowOutOffset = info.scoreOutOffset + static_cast<int64_t>(cuS1Idx) * constInfo_.scoreCount;

        if (cuRealAcSeq <= 0) {
            if (!splitS2ForSingleRow || blockId_ % 2 == 0) {
                CleanInvalidOutput(rowOutOffset, constInfo_.scoreCount);
            }
            continue;
        }

        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        if (cuS2Len <= 0) {
            continue;
        }

        int32_t s2SplitBase = 0;
        int32_t processS2BaseSize = s2BaseSize_;
        int32_t processS2Len = cuS2Len;
        int32_t processS2LenAlign = info.actualSingleProcessSInnerSizeAlign;
        if (splitS2ForSingleRow) {
            processS2BaseSize = s2BaseSize_ / 2;
            s2SplitBase = (blockId_ % 2) * processS2BaseSize;
            processS2Len = cuS2Len > s2SplitBase ? Min(cuS2Len - s2SplitBase, processS2BaseSize) : 0;
            if (processS2Len <= 0) {
                continue;
            }
            processS2LenAlign = QKCommon::Align(processS2Len, static_cast<int32_t>(B32_BLOCK_ALIGN_NUM));
        }

        int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
        int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
        LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
        PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
        for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
            int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> weightsInUb = mmInUb[procGnum * processS2BaseSize];
            LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>();
            if constexpr (!IsSameType<K_T, float>::value) {
                weightsInTUb = weightsInTUb[groupInner_];
            }
            int64_t mmInOffset = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                 outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
            int64_t weightInOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;
            if (splitS2ForSingleRow) {
                QKServiceVec::CopyInS2Split(mmInUb, weightsInTUb, mm1ResGm, weightsGm, mmInOffset + s2SplitBase,
                                            weightInOffset, procGnum, info.actualSingleProcessSInnerSizeAlign,
                                            processS2BaseSize, processS2LenAlign);
            } else {
                QKServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm, mmInOffset, weightInOffset, procGnum,
                                     info.actualSingleProcessSInnerSizeAlign, mmUbStride);
            }

            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            weightsInUb = mmInUb[procGnum * processS2BaseSize];
            QKServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                                  brcBuf, procGnum, processS2BaseSize, outerGidx);
            inQueue_.FreeTensor(mmInUb);
        }

        int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
        QKServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, processS2BaseSize);
        outQueue_.FreeTensor(reduceCacheBuf);

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        QKServiceVec::CopyOut(scoreOutGm[rowOutOffset + cuBaseS2Idx + s2SplitBase], reduceOutInner, processS2Len,
                              rowOutOffset + cuBaseS2Idx + s2SplitBase);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

        bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
        if ((!splitS2ForSingleRow || blockId_ % 2 == 0) && isS2End &&
            cuRealAcSeq < static_cast<int32_t>(constInfo_.scoreCount)) {
            CleanInvalidOutput(rowOutOffset + cuRealAcSeq, constInfo_.scoreCount - cuRealAcSeq);
        }
    }

    if (LAYOUT_T == QK_LAYOUT::BSND) {
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        if (invalidS1Num > 0 && isS1LoopEnd && info.s2Idx == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.scoreOutOffset +
                                       static_cast<int64_t>(s1OffsetPerAiv + innerS1Idx) * constInfo_.scoreCount,
                                   constInfo_.scoreCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && info.s2Idx == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                       constInfo_.scoreCount,
                                   constInfo_.scoreCount);
            }
        }
    }
}
} // namespace QKKernel
#endif
