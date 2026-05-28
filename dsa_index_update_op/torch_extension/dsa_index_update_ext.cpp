#include <torch/extension.h>

#include <dlfcn.h>
#include <string>
#include <vector>

#include "aclnn_torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace {

#ifndef DSA_INDEX_UPDATE_CUST_OPAPI_PATH
#define DSA_INDEX_UPDATE_CUST_OPAPI_PATH ""
#endif

constexpr const char* kDsaIndexUpdateBindingVersion =
    "manual_acl_tensor_aiv_only_v3_direct_cust_opapi";
constexpr const char* kDsaIndexUpdateCustOpApiPath =
    DSA_INDEX_UPDATE_CUST_OPAPI_PATH;

std::string& DsaIndexUpdateCustOpApiLoadError()
{
    static std::string loadError;
    return loadError;
}

void* GetDsaIndexUpdateCustOpApiHandler()
{
    static void* handler = []() -> void* {
        if (kDsaIndexUpdateCustOpApiPath[0] != '\0') {
            dlerror();
            void* pathHandler = dlopen(kDsaIndexUpdateCustOpApiPath, RTLD_NOW | RTLD_LOCAL);
            if (pathHandler == nullptr) {
                const char* err = dlerror();
                DsaIndexUpdateCustOpApiLoadError() =
                    err != nullptr ? err : "unknown dlopen error";
            }
            return pathHandler;
        }
        dlerror();
        void* defaultHandler = dlopen(GetCustOpApiLibName(), RTLD_NOW | RTLD_LOCAL);
        if (defaultHandler == nullptr) {
            const char* err = dlerror();
            DsaIndexUpdateCustOpApiLoadError() =
                err != nullptr ? err : "unknown dlopen error";
        }
        return defaultHandler;
    }();
    return handler;
}

void* GetDsaIndexUpdateCustOpApiFuncAddr(const char* apiName)
{
    void* handler = GetDsaIndexUpdateCustOpApiHandler();
    if (handler == nullptr) {
        return nullptr;
    }
    return dlsym(handler, apiName);
}

class AclTensorGuard {
public:
    explicit AclTensorGuard(const at::Tensor& tensor)
    {
        static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
        TORCH_CHECK(aclCreateTensor != nullptr, "aclCreateTensor not found in ", GetOpApiLibName());

        aclDataType aclDataType =
            kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
        TORCH_CHECK(
            aclDataType != ACL_DT_UNDEFINED,
            std::string(c10::toString(tensor.scalar_type())) + " is not supported by aclTensor.");

        auto sizes = tensor.sizes();
        auto strides = tensor.strides();
        std::vector<int64_t> shape(sizes.begin(), sizes.end());
        std::vector<int64_t> stride(strides.begin(), strides.end());
        tensor_ = aclCreateTensor(
            shape.data(),
            shape.size(),
            aclDataType,
            stride.data(),
            0,
            ACL_FORMAT_ND,
            shape.data(),
            shape.size(),
            tensor.data_ptr());
        TORCH_CHECK(tensor_ != nullptr, "aclCreateTensor failed.");
    }

    ~AclTensorGuard()
    {
        if (tensor_ == nullptr) {
            return;
        }
        static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
        if (aclDestroyTensor != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    AclTensorGuard(const AclTensorGuard&) = delete;
    AclTensorGuard& operator=(const AclTensorGuard&) = delete;

    const aclTensor* get() const
    {
        return tensor_;
    }

private:
    aclTensor* tensor_ = nullptr;
};

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

    static const auto getWorkspaceSizeFuncAddr =
        GetDsaIndexUpdateCustOpApiFuncAddr("aclnnDsaIndexUpdateGetWorkspaceSize");
    static const auto opApiFuncAddr =
        GetDsaIndexUpdateCustOpApiFuncAddr("aclnnDsaIndexUpdate");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        "aclnnDsaIndexUpdate or aclnnDsaIndexUpdateGetWorkspaceSize not found in ",
        kDsaIndexUpdateCustOpApiPath[0] != '\0' ? kDsaIndexUpdateCustOpApiPath : GetCustOpApiLibName(),
        ", dlerror: ",
        DsaIndexUpdateCustOpApiLoadError(),
        ".");

    AclTensorGuard scoreAcl(score);
    AclTensorGuard poolAcl(hbmCachedTokensPool);
    AclTensorGuard candidateLensAcl(candidateLens);
    AclTensorGuard selectedLensAcl(selectedLens);
    AclTensorGuard reqPoolEntriesAcl(reqPoolEntries);
    AclTensorGuard promoteAcl(promoteIdx);
    AclTensorGuard demoteAcl(demoteIdx);
    AclTensorGuard copyCountsAcl(copyCounts);

    using InitHugeMemThreadLocal = int (*)(void*, bool);
    using UnInitHugeMemThreadLocal = void (*)(void*, bool);
    using ReleaseHugeMem = void (*)(void*, bool);
    InitHugeMemThreadLocal initMemFunc =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc =
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    ReleaseHugeMem releaseMemFunc =
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    if (initMemFunc != nullptr) {
        initMemFunc(nullptr, false);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    using GetWorkspaceSizeFunc = int (*)(
        const aclTensor*,
        const aclTensor*,
        const aclTensor*,
        const aclTensor*,
        const aclTensor*,
        int64_t,
        const aclTensor*,
        const aclTensor*,
        const aclTensor*,
        uint64_t*,
        aclOpExecutor**);
    auto getWorkspaceSizeFunc =
        reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);
    int workspaceStatus = getWorkspaceSizeFunc(
        scoreAcl.get(),
        poolAcl.get(),
        candidateLensAcl.get(),
        selectedLensAcl.get(),
        reqPoolEntriesAcl.get(),
        maxCopyTokens,
        promoteAcl.get(),
        demoteAcl.get(),
        copyCountsAcl.get(),
        &workspaceSize,
        &executor);
    TORCH_CHECK(
        workspaceStatus == 0,
        "call aclnnDsaIndexUpdateGetWorkspaceSize failed, detail:",
        aclGetRecentErrMsg());

    void* workspaceAddr = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize != 0) {
        at::TensorOptions options =
            at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspaceTensor = at::empty(
            {static_cast<int64_t>(workspaceSize)},
            options.dtype(kByte));
        workspaceAddr = const_cast<void*>(workspaceTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto aclCall = [&]() -> int {
        using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
        auto opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        int apiRet = opApiFunc(workspaceAddr, workspaceSize, executor, aclStream);
        TORCH_CHECK(
            apiRet == 0,
            "call aclnnDsaIndexUpdate failed, detail:",
            aclGetRecentErrMsg());
        if (releaseMemFunc != nullptr) {
            releaseMemFunc(nullptr, false);
        }
        return apiRet;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnDsaIndexUpdate");
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
    if (unInitMemFunc != nullptr) {
        unInitMemFunc(nullptr, false);
    }
}

} // namespace

PYBIND11_MODULE(_dsa_index_update_C, m)
{
    m.def("binding_version", []() {
        return kDsaIndexUpdateBindingVersion;
    });
    m.def("custom_opapi_path", []() {
        return kDsaIndexUpdateCustOpApiPath;
    });
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
