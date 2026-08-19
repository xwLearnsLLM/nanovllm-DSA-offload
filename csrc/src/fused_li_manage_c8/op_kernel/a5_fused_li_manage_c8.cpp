/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file a5_fused_li_manage_c8.cpp
 * \brief Decode-only fused C8 LightningIndexer + request-pool cache update kernel.
 *
 * 单算子融合，与 BF16 版 A5FusedLiManage 同构（重构后差异收敛为三处）：
 *   1. fp8 cube 打分（Mmad fp8×fp8→fp32 累加）+ Fixpipe<float 直通>；
 *   2. 无 ×1/1024、无 fp16 往返（QK 保持 fp32，relu 由 vf 完成，
 *      与官方"Relu在cube随路做"数值等价）；
 *   3. weight×qScale 预乘 + kScale 经 lidu vf 6 参重载在 Σ 后乘。
 * 乘权归约/bf16-sortable 复用 lidu 的 vf（lightning_indexer_vector1.h），
 * uint16 直方图 payload topk / hit-miss 分类 / victim 驱逐 / request-pool
 * 缓存提交与 BF16 版完全一致。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_fused_li_manage_c8_template_tiling_key.h"
#include "a5_fused_li_manage_c8_kernel.h"

using namespace QLIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                   \
    do {                                                                                               \
        templateClass<QLIType<__VA_ARGS__>> op;                                                        \
        LI_COPY_TILING_DATA(LIC8TilingData, tiling);                                                   \
        op.Init(query, key, weights, queryDequantScale, keyDequantScale,                               \
                actualSeqLengthsQuery, candidateLens, blockTable, reqPoolEntries, cacheSlotsPool,      \
                cacheTokens, sourceIds, destinationSlots, missCounts,                                  \
                user, tiling_data, &tPipe);                                                            \
        op.Process();                                                                                  \
    } while (0)

#define LI_COPY_TILING_DATA(tilingDataStruct, tiling)                                                  \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

template <int DT>
__global__ __aicore__ void a5_fused_li_manage_c8(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                 __gm__ uint8_t *queryDequantScale, __gm__ uint8_t *keyDequantScale,
                                                 __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *reqPoolEntries,
                                                 __gm__ uint8_t *cacheSlotsPool, __gm__ uint8_t *cacheTokens,
                                                 __gm__ uint8_t *candidateLens, __gm__ uint8_t *blockTable,
                                                 __gm__ uint8_t *sourceIds, __gm__ uint8_t *destinationSlots,
                                                 __gm__ uint8_t *missCounts, __gm__ uint8_t *cacheSlotsAlias,
                                                 __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    (void)cacheSlotsAlias;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(DT == LI_C8_TPL_UINT8, "A5FusedLiManageC8 tiling key must be the uint8 storage id");
    // 固定组合：fp8_e4m3fn query/key（ops.json 以 uint8 顶替，数据按 fp8 语义处理）
    // + bf16 weights + fp32 scales + fp32 QK 累加 + uint16 score
    // （官方 QuantLI 的 DT_FLOAT8_E4M3FN 分支）
    INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                             1, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND,
                             bfloat16_t, float32_t, float32_t, uint16_t);
}
