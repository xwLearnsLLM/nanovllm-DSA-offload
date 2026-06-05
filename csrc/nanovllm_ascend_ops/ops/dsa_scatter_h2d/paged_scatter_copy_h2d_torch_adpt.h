#pragma once

#include <climits>
#include <cstdint>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include <c10/core/Device.h>
#include <torch/torch.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "common/torch_adapter/op_api_common.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#ifndef ACL_HOST_REG_MAPPED
#define ACL_HOST_REG_MAPPED 0x2UL
#endif

namespace vllm_ascend {
namespace paged_scatter_copy_h2d_detail {

using HostRegisterV2Fn = aclError (*)(void*, uint64_t, uint32_t);
using HostGetDevicePointerFn = aclError (*)(void*, void**, uint32_t);
using HostUnregisterFn = aclError (*)(void*);
using HostFreeFn = aclError (*)(void*);

struct HostMappingApi {
    HostRegisterV2Fn register_v2;
    HostGetDevicePointerFn get_device_pointer;
    HostUnregisterFn unregister;
    HostFreeFn free_host;
};

inline void* resolve_acl_symbol(const char* name)
{
    void* symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol != nullptr) {
        return symbol;
    }
    void* handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
        symbol = dlsym(handle, name);
    }
    TORCH_CHECK(symbol != nullptr,
                "paged_scatter_copy_h2d required AscendCL symbol ", name, " was not found");
    return symbol;
}

inline const HostMappingApi& host_mapping_api()
{
    static const HostMappingApi api = {
        reinterpret_cast<HostRegisterV2Fn>(resolve_acl_symbol("aclrtHostRegisterV2")),
        reinterpret_cast<HostGetDevicePointerFn>(resolve_acl_symbol("aclrtHostGetDevicePointer")),
        reinterpret_cast<HostUnregisterFn>(resolve_acl_symbol("aclrtHostUnregister")),
        reinterpret_cast<HostFreeFn>(resolve_acl_symbol("aclrtFreeHost")),
    };
    return api;
}

inline std::mutex& host_mapped_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::map<void*, void*>& host_to_dev()
{
    static std::map<void*, void*> mapping;
    return mapping;
}

inline std::optional<void*> lookup_device_ptr(const at::Tensor& tensor)
{
    std::lock_guard<std::mutex> guard(host_mapped_mutex());
    auto it = host_to_dev().find(tensor.data_ptr());
    if (it == host_to_dev().end()) {
        return std::nullopt;
    }
    return it->second;
}

inline std::optional<at::Tensor> host_mapped_device_view(const at::Tensor& host_tensor,
                                                         c10::Device device)
{
    TORCH_CHECK(host_tensor.device().is_cpu(),
                "paged_scatter_copy_h2d host-mapped source must be a CPU tensor");
    auto ptr_d = lookup_device_ptr(host_tensor);
    if (!ptr_d.has_value()) {
        return std::nullopt;
    }
    return at_npu::native::from_blob(
        *ptr_d, host_tensor.sizes().vec(), host_tensor.strides().vec(),
        torch::TensorOptions().dtype(host_tensor.scalar_type()).device(device));
}

inline int64_t token_bytes(const at::Tensor& tensor)
{
    int64_t width = tensor.element_size();
    for (int64_t dim = 2; dim < tensor.dim(); ++dim) {
        width *= tensor.size(dim);
    }
    return width;
}

inline void validate_cache_plane(const at::Tensor& dst,
                                 const at::Tensor& src,
                                 int64_t block_size,
                                 const char* name)
{
    TORCH_CHECK(!dst.device().is_cpu(), name, " dst cache must be an NPU tensor");
    TORCH_CHECK(src.device().is_cpu(), name, " src cache must be a CPU tensor");
    TORCH_CHECK(dst.dim() == 4 && src.dim() == 4,
                name, " cache tensors must be [blocks, block_size, kv_heads, head_dim]");
    TORCH_CHECK(dst.size(1) == block_size && src.size(1) == block_size,
                name, " cache block size mismatch");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
                name, " src/dst dtype mismatch");
    TORCH_CHECK(dst.scalar_type() == at::kHalf || dst.scalar_type() == at::kBFloat16,
                name, " only supports float16/bfloat16 cache tensors");
    for (int64_t dim = 2; dim < dst.dim(); ++dim) {
        TORCH_CHECK(dst.size(dim) == src.size(dim),
                    name, " src/dst trailing dimensions mismatch at dim ", dim);
    }
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(),
                name, " src/dst cache tensors must be contiguous");
    const int64_t row_bytes = token_bytes(src);
    TORCH_CHECK(row_bytes > 0 && row_bytes <= INT32_MAX,
                name, " token row is too large: ", row_bytes);
    TORCH_CHECK(row_bytes % 32 == 0,
                name, " token row bytes must be 32-byte aligned, got ", row_bytes);
}

inline at::Tensor ensure_int32_tensor(const at::Tensor& tensor,
                                      c10::Device device,
                                      const char* name,
                                      int64_t rank)
{
    TORCH_CHECK(tensor.device() == device, name, " must be on the destination NPU device");
    TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
                name, " must be torch.int32 or torch.long/int64");
    TORCH_CHECK(tensor.dim() == rank, name, " must be ", rank, "-D");
    if (tensor.scalar_type() == at::kInt) {
        return tensor.is_contiguous() ? tensor : tensor.contiguous();
    }
    return tensor.to(tensor.options().dtype(at::kInt)).contiguous();
}

} // namespace paged_scatter_copy_h2d_detail

inline at::Tensor paged_scatter_copy_h2d_alloc_host_mapped_empty(
    const at::Tensor& dtype_template,
    at::IntArrayRef sizes)
{
    TORCH_CHECK(dtype_template.device().is_cpu(),
                "paged_scatter_copy_h2d host-mapped dtype template must be on CPU");
    TORCH_CHECK(!sizes.empty(),
                "paged_scatter_copy_h2d host-mapped allocation sizes must not be empty");

    int64_t numel = 1;
    std::vector<int64_t> shape;
    shape.reserve(sizes.size());
    for (int64_t dim : sizes) {
        TORCH_CHECK(dim >= 0, "paged_scatter_copy_h2d got negative dim: ", dim);
        shape.push_back(dim);
        numel *= dim;
    }

    auto opts = torch::TensorOptions().dtype(dtype_template.scalar_type()).device(torch::kCPU);
    if (numel == 0) {
        return torch::empty(shape, opts);
    }

    const size_t nbytes =
        static_cast<size_t>(numel) *
        static_cast<size_t>(c10::elementSize(dtype_template.scalar_type()));

    void* ptr_h = nullptr;
    void* ptr_d = nullptr;
    const auto& host_api = paged_scatter_copy_h2d_detail::host_mapping_api();
    TORCH_CHECK(aclrtMallocHost(&ptr_h, nbytes) == ACL_SUCCESS,
                "paged_scatter_copy_h2d aclrtMallocHost failed for ", nbytes, " bytes");

    const auto register_ret = host_api.register_v2(ptr_h, nbytes, ACL_HOST_REG_MAPPED);
    if (register_ret != ACL_SUCCESS) {
        host_api.free_host(ptr_h);
        TORCH_CHECK(false, "paged_scatter_copy_h2d aclrtHostRegisterV2 failed, ret=", register_ret);
    }
    const auto get_dev_ret = host_api.get_device_pointer(ptr_h, &ptr_d, 0);
    if (get_dev_ret != ACL_SUCCESS) {
        host_api.unregister(ptr_h);
        host_api.free_host(ptr_h);
        TORCH_CHECK(false, "paged_scatter_copy_h2d aclrtHostGetDevicePointer failed, ret=", get_dev_ret);
    }

    {
        std::lock_guard<std::mutex> guard(paged_scatter_copy_h2d_detail::host_mapped_mutex());
        auto& mapping = paged_scatter_copy_h2d_detail::host_to_dev();
        TORCH_CHECK(mapping.find(ptr_h) == mapping.end(),
                    "paged_scatter_copy_h2d host pointer was already registered");
        mapping.emplace(ptr_h, ptr_d);
    }

    auto deleter = [](void* ptr) {
        if (ptr == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(paged_scatter_copy_h2d_detail::host_mapped_mutex());
            paged_scatter_copy_h2d_detail::host_to_dev().erase(ptr);
        }
        paged_scatter_copy_h2d_detail::host_mapping_api().unregister(ptr);
        paged_scatter_copy_h2d_detail::host_mapping_api().free_host(ptr);
    };
    return torch::from_blob(ptr_h, shape, deleter, opts);
}

inline void paged_scatter_copy_h2d(
    at::Tensor& npu_krope_cache,
    at::Tensor& npu_knope_cache,
    const at::Tensor& cpu_krope_cache,
    const at::Tensor& cpu_knope_cache,
    const at::Tensor& npu_block_table,
    const at::Tensor& cpu_block_table,
    const at::Tensor& npu_dst_token_index,
    const at::Tensor& cpu_src_token_index,
    const at::Tensor& copy_counts,
    int64_t block_size)
{
    using namespace paged_scatter_copy_h2d_detail;
    TORCH_CHECK(block_size > 0, "paged_scatter_copy_h2d block_size must be positive");
    TORCH_CHECK(npu_krope_cache.device() == npu_knope_cache.device(),
                "npu_krope_cache and npu_knope_cache must be on the same NPU device");

    validate_cache_plane(npu_krope_cache, cpu_krope_cache, block_size, "krope");
    validate_cache_plane(npu_knope_cache, cpu_knope_cache, block_size, "knope");
    at::Tensor npu_block_table_i32 =
        ensure_int32_tensor(npu_block_table, npu_krope_cache.device(), "npu_block_table", 2);
    at::Tensor cpu_block_table_i32 =
        ensure_int32_tensor(cpu_block_table, npu_krope_cache.device(), "cpu_block_table", 2);
    at::Tensor npu_dst_token_index_i32 =
        ensure_int32_tensor(npu_dst_token_index, npu_krope_cache.device(), "npu_dst_token_index", 2);
    at::Tensor cpu_src_token_index_i32 =
        ensure_int32_tensor(cpu_src_token_index, npu_krope_cache.device(), "cpu_src_token_index", 2);
    at::Tensor copy_counts_i32 =
        ensure_int32_tensor(copy_counts, npu_krope_cache.device(), "copy_counts", 1);
    TORCH_CHECK(npu_dst_token_index_i32.sizes() == cpu_src_token_index_i32.sizes(),
                "npu_dst_token_index and cpu_src_token_index shape mismatch");
    TORCH_CHECK(npu_block_table_i32.size(0) == npu_dst_token_index_i32.size(0) &&
                    cpu_block_table_i32.size(0) == npu_dst_token_index_i32.size(0),
                "block table batch size must match token index batch size");
    TORCH_CHECK(copy_counts_i32.size(0) == npu_dst_token_index_i32.size(0),
                "copy_counts batch size must match token index batch size");

    const c10_npu::OptionalNPUGuard npu_guard(npu_krope_cache.device());
    auto krope_src = host_mapped_device_view(cpu_krope_cache, npu_krope_cache.device());
    auto knope_src = host_mapped_device_view(cpu_knope_cache, npu_knope_cache.device());
    TORCH_CHECK(krope_src.has_value() && knope_src.has_value(),
                "paged_scatter_copy_h2d requires host-mapped CPU cache tensors");

    const int64_t krope_unit_bytes = token_bytes(cpu_krope_cache);
    const int64_t knope_unit_bytes = token_bytes(cpu_knope_cache);
    EXEC_NPU_CMD(aclnnPagedScatterCopyH2d,
                 krope_src.value(), knope_src.value(),
                 npu_block_table_i32, cpu_block_table_i32,
                 npu_dst_token_index_i32, cpu_src_token_index_i32, copy_counts_i32,
                 krope_unit_bytes, knope_unit_bytes, block_size,
                 npu_krope_cache, npu_knope_cache);
}

} // namespace vllm_ascend
