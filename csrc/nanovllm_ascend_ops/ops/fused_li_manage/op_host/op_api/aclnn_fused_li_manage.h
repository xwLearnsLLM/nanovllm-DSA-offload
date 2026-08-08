/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_NANOVLLM_FUSED_LI_MANAGE_H
#define ACLNN_NANOVLLM_FUSED_LI_MANAGE_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnNanovllmFusedLiManage(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_NANOVLLM_FUSED_LI_MANAGE_H

