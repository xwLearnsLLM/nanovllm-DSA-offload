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
 * \file qk_score_vector.h
 * \brief
 */
#ifndef QK_SCORE_VECTOR_H
#define QK_SCORE_VECTOR_H

#include "kernel_operator.h"

namespace QKServiceVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t BLOCK_BYTES = 32;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    const int64_t mmCopyCount = groupInner * s2Inner;
    const int64_t mmCopyBytes = mmCopyCount * static_cast<int64_t>(sizeof(float));
    const bool mmAligned = mmUbStride == 0 && (mmCopyBytes % BLOCK_BYTES == 0) &&
                           ((MMout_gmoffset * static_cast<int64_t>(sizeof(float))) % BLOCK_BYTES == 0);
    if (mmAligned) {
        AscendC::DataCopy(mmOutUb, mMoutGm[MMout_gmoffset], mmCopyCount);
    } else {
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams dataCopymMoutParams;
        dataCopymMoutParams.blockCount = groupInner;
        dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
        dataCopymMoutParams.srcStride = 0;
        dataCopymMoutParams.dstStride = mmUbStride;
        dataCopymMoutParams.rsv = 0;
        AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);
    }

    const int64_t weightCopyBytes = groupInner * static_cast<int64_t>(sizeof(T));
    const bool weightAligned = (weightCopyBytes % BLOCK_BYTES == 0) &&
                               ((weights_gmoffset * static_cast<int64_t>(sizeof(T))) % BLOCK_BYTES == 0);
    if (weightAligned) {
        AscendC::DataCopy(weightsUb, weightScaleGm[weights_gmoffset], groupInner);
    } else {
        AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams dataCopyweightParams;
        dataCopyweightParams.blockCount = 1;
        dataCopyweightParams.blockLen = groupInner * sizeof(T);
        dataCopyweightParams.srcStride = 0;
        dataCopyweightParams.dstStride = 0;
        dataCopyweightParams.rsv = 0;
        AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
    }
}

template <typename T>
__aicore__ inline void CopyInS2Split(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb,
                                     GlobalTensor<float> &mMoutGm, GlobalTensor<T> &weightScaleGm,
                                     int64_t MMout_gmoffset, int64_t weights_gmoffset, int64_t groupInner,
                                     int64_t srcS2Stride, int64_t dstS2Stride, int64_t s2CopyLen)
{
    AscendC::DataCopy(mmOutUb, mMoutGm[MMout_gmoffset],
                      AscendC::DataCopyParams(groupInner, s2CopyLen / B32_BLOCK_ALIGN_NUM,
                                              (srcS2Stride - s2CopyLen) / B32_BLOCK_ALIGN_NUM,
                                              (dstS2Stride - s2CopyLen) / B32_BLOCK_ALIGN_NUM));

    const int64_t weightCopyBytes = groupInner * static_cast<int64_t>(sizeof(T));
    const bool weightAligned = (weightCopyBytes % BLOCK_BYTES == 0) &&
                               ((weights_gmoffset * static_cast<int64_t>(sizeof(T))) % BLOCK_BYTES == 0);
    if (weightAligned) {
        AscendC::DataCopy(weightsUb, weightScaleGm[weights_gmoffset], groupInner);
    } else {
        AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams dataCopyweightParams;
        dataCopyweightParams.blockCount = 1;
        dataCopyweightParams.blockLen = groupInner * sizeof(T);
        dataCopyweightParams.srcStride = 0;
        dataCopyweightParams.dstStride = 0;
        dataCopyweightParams.rsv = 0;
        AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
    }
}


template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount,
                               int64_t dstElementOffset)
{
    if (copyCount <= 0) {
        return;
    }
    int64_t elemBytes = static_cast<int64_t>(sizeof(T));
    if (((copyCount * elemBytes) % BLOCK_BYTES == 0) && ((dstElementOffset * elemBytes) % BLOCK_BYTES == 0)) {
        AscendC::DataCopy(dstGm, srcUb, copyCount);
        return;
    }
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}


template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, QKCommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - clz(value);
        return (1 << pow);
    }
}


__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

} // namespace QKServiceVec
#endif // QK_SCORE_VECTOR_H
