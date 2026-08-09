/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_li_manage_mtp_template_tiling_key.h"
#include "fused_li_manage_mtp_kernel.h"

using namespace LICommon;
using namespace LIMtpKernel;
using namespace AscendC;

#define COPY_MTP_TILING_DATA(tilingDataStruct, tiling)                         \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);     \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

#define INVOKE_MTP_LIDU(templateClass, ...)                                    \
    do {                                                                        \
        templateClass<LIType<__VA_ARGS__>> op;                                  \
        COPY_MTP_TILING_DATA(LIUMtpTilingData, tiling);                         \
        op.Init(query, key, weights, reqPoolEntries, cacheSlots, cacheTokens,   \
                candidateLens, blockTable, topkSlots, topkSourceIds,            \
                missSourceIds,                                                  \
                missDestinationSlots, missCounts, user, tiling_data, &tPipe);   \
        op.Process();                                                            \
    } while (0)

template <int DT>
__global__ __aicore__ void nanovllm_fused_li_manage_mtp(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *candidateLens,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *missSourceIds, __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts, __gm__ uint8_t *cacheSlotsOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)cacheSlotsOut;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT == LI_MTP_TPL_FP16) {
        INVOKE_MTP_LIDU(LIMtpPreload, half);
    } else {
        INVOKE_MTP_LIDU(LIMtpPreload, bfloat16_t);
    }
#endif
}
