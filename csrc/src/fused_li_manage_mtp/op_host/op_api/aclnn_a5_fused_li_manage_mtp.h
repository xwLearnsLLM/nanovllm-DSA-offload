/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_a5_fused_li_manage_mtp_H
#define ACLNN_a5_fused_li_manage_mtp_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnA5FusedLiManageMtp(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_a5_fused_li_manage_mtp_H
