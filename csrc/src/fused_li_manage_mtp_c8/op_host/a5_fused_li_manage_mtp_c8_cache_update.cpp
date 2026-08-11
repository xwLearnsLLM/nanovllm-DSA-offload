/** Host definition and tiling for request-level C8 MTP cache management. */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_fused_li_manage_mtp_c8_cache_update_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t TOPK_INDICES = 0;
constexpr size_t ACTUAL_SEQ_LENGTHS_QUERY = 1;
constexpr size_t REQ_POOL_ENTRIES = 2;
constexpr size_t CACHE_SLOTS_POOL = 3;
constexpr size_t CACHE_TOKENS = 4;
constexpr size_t CANDIDATE_LENS = 5;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_QUERIES_PER_REQUEST = 4;
constexpr int64_t MIN_QUERIES_PER_REQUEST = 2;
constexpr int64_t UNION_CAPACITY = SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5FusedLiManageMtpC8CacheUpdate(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = TOPK_INDICES; index <= CANDIDATE_LENS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape topk =
        context->GetInputShape(TOPK_INDICES)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const gert::Shape req =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape pool =
        context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape lengths =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    if (topk.GetDimNum() != 3 || topk.GetDim(0) <= 0 ||
        topk.GetDim(1) != 1 || topk.GetDim(2) != SPARSE_COUNT ||
        actualQ.GetDimNum() != 1 || actualQ.GetDim(0) <= 0 ||
        req.GetDimNum() != 1 || budgets.GetDimNum() != 1 ||
        lengths.GetDimNum() != 1 || pool.GetDimNum() != 2 ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) <= 0 ||
        pool.GetDim(1) > MAX_SOURCE_CAPACITY) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = actualQ.GetDim(0);
    const int64_t packedQueries = topk.GetDim(0);
    if (req.GetDim(0) != batch || budgets.GetDim(0) != batch ||
        lengths.GetDim(0) != batch ||
        packedQueries < batch * MIN_QUERIES_PER_REQUEST ||
        packedQueries > batch * MAX_QUERIES_PER_REQUEST) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        batch < static_cast<int64_t>(aivCount) ? batch : aivCount);
    auto *tiling = context->GetTilingData<
        A5FusedLiManageMtpC8CacheUpdateTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->packedQueryCount = static_cast<uint32_t>(packedQueries);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(pool.GetDim(1));
    context->SetBlockDim(usedCoreNum);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5FusedLiManageMtpC8CacheUpdateShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(TOPK_INDICES) == nullptr ||
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY) == nullptr ||
        context->GetInputShape(CACHE_SLOTS_POOL) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 5; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape *topk = context->GetInputShape(TOPK_INDICES);
    const gert::Shape *actualQ =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY);
    const gert::Shape *pool = context->GetInputShape(CACHE_SLOTS_POOL);
    *context->GetOutputShape(0) = *topk;
    for (size_t index = 1; index <= 2; ++index) {
        context->GetOutputShape(index)->SetDimNum(2);
        context->GetOutputShape(index)->SetDim(0, actualQ->GetDim(0));
        context->GetOutputShape(index)->SetDim(1, UNION_CAPACITY);
    }
    context->GetOutputShape(3)->SetDimNum(1);
    context->GetOutputShape(3)->SetDim(0, actualQ->GetDim(0));
    *context->GetOutputShape(4) = *pool;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5FusedLiManageMtpC8CacheUpdateDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 5; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

class A5FusedLiManageMtpC8CacheUpdate : public OpDef {
public:
    explicit A5FusedLiManageMtpC8CacheUpdate(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("topk_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_slots_pool").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("topk_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_source_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("cache_slots_alias").ParamType(REQUIRED).DataType(ints).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5FusedLiManageMtpC8CacheUpdate)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5FusedLiManageMtpC8CacheUpdate);

IMPL_OP_INFERSHAPE(A5FusedLiManageMtpC8CacheUpdate)
    .InferShape(InferA5FusedLiManageMtpC8CacheUpdateShape)
    .InferDataType(InferA5FusedLiManageMtpC8CacheUpdateDataType);
}  // namespace ops
