/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_NANOVLLM_FUSED_LI_MANAGE_MTP_H_
#define ACLNN_NANOVLLM_FUSED_LI_MANAGE_MTP_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens, const aclTensor *candidateLens,
    const aclTensor *blockTable, const aclTensor *topkSlotsOut,
    const aclTensor *topkSourceIdsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheSlotsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnNanovllmFusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
