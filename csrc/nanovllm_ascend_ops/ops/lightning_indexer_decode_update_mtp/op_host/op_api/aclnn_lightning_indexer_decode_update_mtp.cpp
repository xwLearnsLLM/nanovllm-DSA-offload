/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_lightning_indexer_decode_update_mtp.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNanovllmLiduDecodeUpdateMtpGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens, const aclTensor *candidateLens,
    const aclTensor *blockTable, const aclTensor *topkSlotsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheSlotsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNanovllmLiduDecodeUpdateMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

aclnnStatus aclnnNanovllmLiduDecodeUpdateMtpGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens, const aclTensor *candidateLens,
    const aclTensor *blockTable, const aclTensor *topkSlotsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheSlotsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerNanovllmLiduDecodeUpdateMtpGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, cacheTokens,
        candidateLens, blockTable, topkSlotsOut, missSourceIdsOut,
        missDestinationSlotsOut, missCountsOut, cacheSlotsOut, workspaceSize,
        executor);
}

aclnnStatus aclnnNanovllmLiduDecodeUpdateMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerNanovllmLiduDecodeUpdateMtp(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
