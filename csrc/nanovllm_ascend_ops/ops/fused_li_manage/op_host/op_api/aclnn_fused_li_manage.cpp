/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_fused_li_manage.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNanovllmFusedLiManageGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *topkIndexOut, const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut,
    const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNanovllmFusedLiManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnNanovllmFusedLiManageGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlots,
    const aclTensor *cacheTokens,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable,
    const aclTensor *topkIndexOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut,
    const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerNanovllmFusedLiManageGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, cacheTokens,
        actualSeqLengthsKey, blockTable, topkIndexOut, topkSlotsOut,
        missCountOut, cacheSlotsOut, workspaceSize, executor);
}

aclnnStatus aclnnNanovllmFusedLiManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerNanovllmFusedLiManage(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

