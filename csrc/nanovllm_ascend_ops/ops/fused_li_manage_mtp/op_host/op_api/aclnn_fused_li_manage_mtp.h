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
    const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *keyDequantScale,
    const aclTensor *key, const aclTensor *blockTable,
    const aclTensor *actualQueryLens, const aclTensor *actualKeyLens,
    const aclTensor *offloadKeyLens, const aclTensor *reqValid,
    const aclTensor *reqPoolEntries, const aclTensor *cacheState,
    const aclTensor *cacheSlots, const aclTensor *topkSourceIdsOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *topkMissCountsOut,
    const aclTensor *missSourceIdsOut,
    const aclTensor *missDestinationSlotsOut, const aclTensor *missCountsOut,
    const aclTensor *cacheStateOut, const aclTensor *cacheSlotsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnNanovllmFusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
