#include "dsa_index_update.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DsaIndexUpdate);

static const std::initializer_list<op::DataType> SCORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_BF16
};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32
};

static bool IsAiCoreSupport(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    const aclTensor* promoteIdx, const aclTensor* demoteIdx, const aclTensor* copyCounts)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "DsaIndexUpdate not supported on this platform: npuArch=%d.",
            static_cast<int>(npuArch)),
        return false);

    OP_CHECK(CheckType(score->GetDataType(), SCORE_DTYPE_SUPPORT_LIST),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DsaIndexUpdate requires score dtype BF16."),
        return false);
    OP_CHECK(CheckType(hbmCachedTokensPool->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(candidateLens->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(selectedLens->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(reqPoolEntries->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(promoteIdx->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(demoteIdx->GetDataType(), INDEX_DTYPE_SUPPORT_LIST) &&
                 CheckType(copyCounts->GetDataType(), INDEX_DTYPE_SUPPORT_LIST),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DsaIndexUpdate index tensors must be INT32."),
        return false);
    return true;
}

static bool DsaIndexUpdateAiCore(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens, const aclTensor* promoteIdx, const aclTensor* demoteIdx,
    const aclTensor* copyCounts, aclOpExecutor* executor)
{
    L0_DFX(DsaIndexUpdateAiCore, score, hbmCachedTokensPool, candidateLens, selectedLens,
        reqPoolEntries, promoteIdx, demoteIdx, copyCounts);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DsaIndexUpdate,
        OP_INPUT(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries),
        OP_OUTPUT(promoteIdx, demoteIdx, copyCounts),
        OP_ATTR(maxCopyTokens));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DsaIndexUpdateAiCore failed."),
        return false);
    return true;
}

bool DsaIndexUpdate(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens, const aclTensor* promoteIdx, const aclTensor* demoteIdx,
    const aclTensor* copyCounts, aclOpExecutor* executor)
{
    OP_CHECK(maxCopyTokens > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DsaIndexUpdate requires max_copy_tokens > 0, got %ld.",
            maxCopyTokens),
        return false);
    OP_CHECK(IsAiCoreSupport(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
                 promoteIdx, demoteIdx, copyCounts),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DsaIndexUpdate IsAiCoreSupport check failed."),
        return false);
    return DsaIndexUpdateAiCore(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
        maxCopyTokens, promoteIdx, demoteIdx, copyCounts, executor);
}

} // namespace l0op
