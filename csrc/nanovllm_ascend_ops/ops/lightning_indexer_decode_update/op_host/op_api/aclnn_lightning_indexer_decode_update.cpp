/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_lightning_indexer_decode_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNanovllmLiduDecodeUpdateGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *topkIndexOut, const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut,
    const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNanovllmLiduDecodeUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnNanovllmLiduDecodeUpdateGetWorkspaceSize(
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
    return aclnnInnerNanovllmLiduDecodeUpdateGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, cacheTokens,
        actualSeqLengthsKey, blockTable, topkIndexOut, topkSlotsOut,
        missCountOut, cacheSlotsOut, workspaceSize, executor);
}

aclnnStatus aclnnNanovllmLiduDecodeUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerNanovllmLiduDecodeUpdate(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
