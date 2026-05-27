#ifndef ACLNN_DSA_INDEX_UPDATE_H_
#define ACLNN_DSA_INDEX_UPDATE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnDsaIndexUpdateGetWorkspaceSize(
    const aclTensor* score,
    const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens,
    const aclTensor* selectedLens,
    const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens,
    const aclTensor* promoteIdx,
    const aclTensor* demoteIdx,
    const aclTensor* copyCounts,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnDsaIndexUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_DSA_INDEX_UPDATE_H_
