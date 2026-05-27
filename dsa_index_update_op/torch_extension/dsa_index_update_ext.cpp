#include <torch/extension.h>

#include "aclnn_torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace {

void CheckTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t dim)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch.");
    TORCH_CHECK(tensor.dim() == dim, name, " rank mismatch.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void DsaIndexUpdatePy(
    const at::Tensor& score,
    at::Tensor hbmCachedTokensPool,
    at::Tensor promoteIdx,
    at::Tensor demoteIdx,
    at::Tensor copyCounts,
    const at::Tensor& candidateLens,
    const at::Tensor& selectedLens,
    const at::Tensor& reqPoolEntries,
    int64_t maxCopyTokens)
{
    CheckTensor(score, "score", at::kBFloat16, 2);
    CheckTensor(hbmCachedTokensPool, "hbm_cached_tokens_pool", at::kInt, 2);
    CheckTensor(promoteIdx, "promote_idx", at::kInt, 2);
    CheckTensor(demoteIdx, "demote_idx", at::kInt, 2);
    CheckTensor(copyCounts, "copy_counts", at::kInt, 1);
    CheckTensor(candidateLens, "candidate_lens", at::kInt, 1);
    CheckTensor(selectedLens, "selected_lens", at::kInt, 1);
    CheckTensor(reqPoolEntries, "req_pool_entries", at::kInt, 1);

    const int64_t batchSize = score.size(0);
    TORCH_CHECK(batchSize > 0, "batch size must be positive.");
    TORCH_CHECK(maxCopyTokens > 0 && maxCopyTokens <= 128,
        "max_copy_tokens must be in (0, 128], got ", maxCopyTokens);
    TORCH_CHECK(candidateLens.size(0) == batchSize &&
                    selectedLens.size(0) == batchSize &&
                    reqPoolEntries.size(0) == batchSize &&
                    promoteIdx.size(0) == batchSize &&
                    demoteIdx.size(0) == batchSize &&
                    copyCounts.size(0) == batchSize,
        "batch dimensions must match.");
    TORCH_CHECK(promoteIdx.size(1) >= maxCopyTokens &&
                    demoteIdx.size(1) == promoteIdx.size(1),
        "promote/demote output capacity must be >= max_copy_tokens and equal.");

    EXEC_NPU_CMD(aclnnDsaIndexUpdate,
        score,
        hbmCachedTokensPool,
        candidateLens,
        selectedLens,
        reqPoolEntries,
        maxCopyTokens,
        promoteIdx,
        demoteIdx,
        copyCounts);
}

} // namespace

PYBIND11_MODULE(_dsa_index_update_C, m)
{
    m.def("dsa_index_update", &DsaIndexUpdatePy,
        pybind11::arg("score"),
        pybind11::arg("hbm_cached_tokens_pool"),
        pybind11::arg("promote_idx"),
        pybind11::arg("demote_idx"),
        pybind11::arg("copy_counts"),
        pybind11::arg("candidate_lens"),
        pybind11::arg("selected_lens"),
        pybind11::arg("req_pool_entries"),
        pybind11::arg("max_copy_tokens"));
}
