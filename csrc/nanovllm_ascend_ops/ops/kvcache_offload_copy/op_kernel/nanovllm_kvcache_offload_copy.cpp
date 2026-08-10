/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#include "kvcache_offload_copy_kernel.h"

using namespace AscendC;
using namespace KvcacheOffloadCopyNs;

extern "C" __global__ __aicore__ void nanovllm_kvcache_offload_copy(
    GM_ADDR hbmKvCache, GM_ADDR dramKvCache, GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable, GM_ADDR copyCounts, GM_ADDR dramKvCacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KvcacheOffloadCopyKernel op(&pipe, &tilingData);
        op.Init(hbmKvCache, dramKvCache, hbmBlockTable, dramBlockTable, copyCounts);
        op.Process();
    }
}
