/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_fused_li_manage_mtp.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *keyDequantScale,
    const aclTensor *key, const aclTensor *blockTable,
    const aclTensor *actualQueryLens, const aclTensor *actualKeyLens,
    const aclTensor *offloadKeyLens, const aclTensor *reqValid,
    const aclTensor *reqPoolEntries, const aclTensor *cacheState,
    const aclTensor *cacheSlots, const aclTensor *topkSourceIdsOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheStateOut, const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

aclnnStatus aclnnNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *keyDequantScale,
    const aclTensor *key, const aclTensor *blockTable,
    const aclTensor *actualQueryLens, const aclTensor *actualKeyLens,
    const aclTensor *offloadKeyLens, const aclTensor *reqValid,
    const aclTensor *reqPoolEntries, const aclTensor *cacheState,
    const aclTensor *cacheSlots, const aclTensor *topkSourceIdsOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheStateOut, const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
        weights, queryDequantScale, query, keyDequantScale, key, blockTable,
        actualQueryLens, actualKeyLens, offloadKeyLens, reqValid,
        reqPoolEntries, cacheState, cacheSlots, topkSourceIdsOut, topkSlotsOut,
        missSourceIdsOut,
        missDestinationSlotsOut, missCountsOut, cacheStateOut, cacheSlotsOut, workspaceSize,
        executor);
}

aclnnStatus aclnnNanovllmFusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerNanovllmFusedLiManageMtp(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
