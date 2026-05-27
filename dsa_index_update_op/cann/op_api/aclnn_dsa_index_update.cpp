#include "aclnn_dsa_index_update.h"
#include "dsa_index_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace {

constexpr int64_t ACLNN_MAX_SHAPE_RANK = 8;
constexpr int64_t DSA_INDEX_UPDATE_MAX_K_API = 128;

static bool CheckNotNull(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    const aclTensor* promoteIdx, const aclTensor* demoteIdx, const aclTensor* copyCounts,
    const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_NULL(score, return false);
    OP_CHECK_NULL(hbmCachedTokensPool, return false);
    OP_CHECK_NULL(candidateLens, return false);
    OP_CHECK_NULL(selectedLens, return false);
    OP_CHECK_NULL(reqPoolEntries, return false);
    OP_CHECK_NULL(promoteIdx, return false);
    OP_CHECK_NULL(demoteIdx, return false);
    OP_CHECK_NULL(copyCounts, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool CheckDtype(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    const aclTensor* promoteIdx, const aclTensor* demoteIdx, const aclTensor* copyCounts)
{
    OP_CHECK(score->GetDataType() == DataType::DT_BF16,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "score must be BF16."),
        return false);
    OP_CHECK(hbmCachedTokensPool->GetDataType() == DataType::DT_INT32 &&
                 candidateLens->GetDataType() == DataType::DT_INT32 &&
                 selectedLens->GetDataType() == DataType::DT_INT32 &&
                 reqPoolEntries->GetDataType() == DataType::DT_INT32 &&
                 promoteIdx->GetDataType() == DataType::DT_INT32 &&
                 demoteIdx->GetDataType() == DataType::DT_INT32 &&
                 copyCounts->GetDataType() == DataType::DT_INT32,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index tensors must be INT32."),
        return false);
    return true;
}

static bool CheckFormat(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    const aclTensor* promoteIdx, const aclTensor* demoteIdx, const aclTensor* copyCounts)
{
    OP_CHECK(!(IsPrivateFormat(score->GetStorageFormat()) ||
                  IsPrivateFormat(hbmCachedTokensPool->GetStorageFormat()) ||
                  IsPrivateFormat(candidateLens->GetStorageFormat()) ||
                  IsPrivateFormat(selectedLens->GetStorageFormat()) ||
                  IsPrivateFormat(reqPoolEntries->GetStorageFormat()) ||
                  IsPrivateFormat(promoteIdx->GetStorageFormat()) ||
                  IsPrivateFormat(demoteIdx->GetStorageFormat()) ||
                  IsPrivateFormat(copyCounts->GetStorageFormat())),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format is not supported."),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens, const aclTensor* promoteIdx, const aclTensor* demoteIdx,
    const aclTensor* copyCounts)
{
    OP_CHECK_MAX_DIM(score, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(hbmCachedTokensPool, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(candidateLens, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(selectedLens, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(reqPoolEntries, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(promoteIdx, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(demoteIdx, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(copyCounts, ACLNN_MAX_SHAPE_RANK, return false);

    const auto& scoreShape = score->GetViewShape();
    const auto& poolShape = hbmCachedTokensPool->GetViewShape();
    const auto& candidateLensShape = candidateLens->GetViewShape();
    const auto& selectedLensShape = selectedLens->GetViewShape();
    const auto& reqPoolEntriesShape = reqPoolEntries->GetViewShape();
    const auto& promoteShape = promoteIdx->GetViewShape();
    const auto& demoteShape = demoteIdx->GetViewShape();
    const auto& copyCountsShape = copyCounts->GetViewShape();

    OP_CHECK(scoreShape.GetDimNum() == 2 && poolShape.GetDimNum() == 2 &&
                 candidateLensShape.GetDimNum() == 1 && selectedLensShape.GetDimNum() == 1 &&
                 reqPoolEntriesShape.GetDimNum() == 1 && promoteShape.GetDimNum() == 2 &&
                 demoteShape.GetDimNum() == 2 && copyCountsShape.GetDimNum() == 1,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid tensor ranks."),
        return false);

    const int64_t batchSize = scoreShape.GetDim(0);
    OP_CHECK(batchSize > 0 && scoreShape.GetDim(1) > 0 &&
                 poolShape.GetDim(0) > 0 && poolShape.GetDim(1) > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid empty score or hbm_cached_tokens_pool shape."),
        return false);
    OP_CHECK(candidateLensShape.GetDim(0) == batchSize &&
                 selectedLensShape.GetDim(0) == batchSize &&
                 reqPoolEntriesShape.GetDim(0) == batchSize &&
                 promoteShape.GetDim(0) == batchSize &&
                 demoteShape.GetDim(0) == batchSize &&
                 copyCountsShape.GetDim(0) == batchSize &&
                 promoteShape.GetDim(1) >= maxCopyTokens &&
                 demoteShape.GetDim(1) == promoteShape.GetDim(1),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DsaIndexUpdate shape mismatch."),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* score, const aclTensor* hbmCachedTokensPool,
    const aclTensor* candidateLens, const aclTensor* selectedLens, const aclTensor* reqPoolEntries,
    int64_t maxCopyTokens, const aclTensor* promoteIdx, const aclTensor* demoteIdx,
    const aclTensor* copyCounts,
    const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_COND(CheckNotNull(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
                   promoteIdx, demoteIdx, copyCounts, workspaceSize, executor),
        ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed.");
    CHECK_COND(maxCopyTokens > 0 && maxCopyTokens <= DSA_INDEX_UPDATE_MAX_K_API,
        ACLNN_ERR_PARAM_INVALID, "max_copy_tokens must be in (0, %ld], got %ld.",
        DSA_INDEX_UPDATE_MAX_K_API, maxCopyTokens);
    CHECK_COND(CheckDtype(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
                   promoteIdx, demoteIdx, copyCounts),
        ACLNN_ERR_PARAM_INVALID, "CheckDtype failed.");
    CHECK_COND(CheckFormat(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
                   promoteIdx, demoteIdx, copyCounts),
        ACLNN_ERR_PARAM_INVALID, "CheckFormat failed.");
    CHECK_COND(CheckShape(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
                   maxCopyTokens, promoteIdx, demoteIdx, copyCounts),
        ACLNN_ERR_PARAM_INVALID, "CheckShape failed.");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnDsaIndexUpdateGetWorkspaceSize(
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
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDsaIndexUpdate,
        DFX_IN(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries),
        DFX_OUT(promoteIdx, demoteIdx, copyCounts));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
        maxCopyTokens, promoteIdx, demoteIdx, copyCounts, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool ok = l0op::DsaIndexUpdate(score, hbmCachedTokensPool, candidateLens, selectedLens, reqPoolEntries,
        maxCopyTokens, promoteIdx, demoteIdx, copyCounts, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnDsaIndexUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDsaIndexUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
