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
 * \file qk_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "qk_score_template_tiling_key.h"
#include "qk_score_kernel.h"

using namespace QKKernel;

#define INVOKE_QK_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<QKType<__VA_ARGS__>> op;                                                                         \
        QK_COPY_TILING_DATA(QkScoreTilingData, tiling);                                                                     \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, scores, user,           \
                tiling_data, &tPipe);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

#define QK_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                  \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;


template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int S2_BASE_T>
__global__ __aicore__ void qk_score(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                             __gm__ uint8_t *blocktable, __gm__ uint8_t *scores,
                                             __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if constexpr (DT_Q == QK_TPL_FP16 && DT_K == QK_TPL_FP16 && DT_OUT == QK_TPL_FLOAT) {
        INVOKE_QK_NO_KFC_OP_IMPL(QKPreload, half, half, float, PAGE_ATTENTION,
                                 QK_LAYOUT(LAYOUT_T), QK_LAYOUT(K_LAYOUT_T), S2_BASE_T);
    } else {
        INVOKE_QK_NO_KFC_OP_IMPL(QKPreload, bfloat16_t, bfloat16_t, float, PAGE_ATTENTION,
                                 QK_LAYOUT(LAYOUT_T), QK_LAYOUT(K_LAYOUT_T), S2_BASE_T);
    }
#endif
}
