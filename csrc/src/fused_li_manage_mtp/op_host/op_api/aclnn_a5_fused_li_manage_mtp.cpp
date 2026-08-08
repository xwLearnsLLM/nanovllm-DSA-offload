/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_a5_fused_li_manage_mtp.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerA5FusedLiManageMtpGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *cacheSlots, const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *topkIndexOut, const aclTensor *topkSlotsOut,
    const aclTensor *missIndexOut, const aclTensor *missSlotsOut,
    const aclTensor *missCountOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerA5FusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnA5FusedLiManageMtpGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *cacheSlots,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable,
    const aclTensor *topkIndexOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missIndexOut,
    const aclTensor *missSlotsOut,
    const aclTensor *missCountOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerA5FusedLiManageMtpGetWorkspaceSize(
        query, key, weights, cacheSlots, actualSeqLengthsQuery, actualSeqLengthsKey, blockTable,
        topkIndexOut, topkSlotsOut,
        missIndexOut, missSlotsOut,
        missCountOut, workspaceSize, executor);
}

aclnnStatus aclnnA5FusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerA5FusedLiManageMtp(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
