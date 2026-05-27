#ifndef OP_API_INC_LEVEL0_DSA_INDEX_UPDATE_H_
#define OP_API_INC_LEVEL0_DSA_INDEX_UPDATE_H_

#include "opdev/op_executor.h"

namespace l0op {

bool DsaIndexUpdate(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens, const aclTensor* promoteIdx, const aclTensor* demoteIdx,
    const aclTensor* copyCounts, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_DSA_INDEX_UPDATE_H_
