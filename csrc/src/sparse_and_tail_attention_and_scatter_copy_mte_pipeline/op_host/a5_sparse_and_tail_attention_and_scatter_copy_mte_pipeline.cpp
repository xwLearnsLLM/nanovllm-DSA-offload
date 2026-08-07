/**
 * Ascend 950 hit-first MTE-prefetch sparse attention pipeline.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

#include "a5_sparse_and_tail_attention_and_scatter_copy_mte_pipeline_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t QUERY = 0;
constexpr size_t KEY = 1;
constexpr size_t VALUE = 2;
constexpr size_t SPARSE_SLOTS = 3;
constexpr size_t HBM_BLOCK_TABLE = 4;
constexpr size_t ACTUAL_Q = 5;
constexpr size_t ACTUAL_KV = 6;
constexpr size_t QUERY_ROPE = 7;
constexpr size_t HBM_KEY_ROPE = 8;
constexpr size_t CACHE_TOKENS = 9;
constexpr size_t DRAM_KEY_ROPE = 10;
constexpr size_t DRAM_KV_CACHE = 11;
constexpr size_t DRAM_BLOCK_TABLE = 12;
constexpr size_t SOURCE_TOKEN_IDS = 13;
constexpr size_t COPY_COUNTS = 14;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t CKV_DIM = 512;
constexpr int64_t KPE_DIM = 64;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_REGULAR_LOCAL_HEADS = 64;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;

bool IsShape(
    const gert::Shape &shape,
    std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : dims) {
        if (dim >= 0 && shape.GetDim(index) != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

ge::graphStatus CheckFusedInputs(
    gert::TilingContext *context,
    uint32_t &copyCap,
    uint32_t &dramMaxBlockNum)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = QUERY; index <= COPY_COUNTS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape query =
        context->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape hbmKv =
        context->GetInputShape(KEY)->GetStorageShape();
    const gert::Shape value =
        context->GetInputShape(VALUE)->GetStorageShape();
    const gert::Shape slots =
        context->GetInputShape(SPARSE_SLOTS)->GetStorageShape();
    const gert::Shape hbmTable =
        context->GetInputShape(HBM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_Q)->GetStorageShape();
    const gert::Shape actualKv =
        context->GetInputShape(ACTUAL_KV)->GetStorageShape();
    const gert::Shape queryRope =
        context->GetInputShape(QUERY_ROPE)->GetStorageShape();
    const gert::Shape hbmRope =
        context->GetInputShape(HBM_KEY_ROPE)->GetStorageShape();
    const gert::Shape cacheTokens =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape dramRope =
        context->GetInputShape(DRAM_KEY_ROPE)->GetStorageShape();
    const gert::Shape dramKv =
        context->GetInputShape(DRAM_KV_CACHE)->GetStorageShape();
    const gert::Shape dramTable =
        context->GetInputShape(DRAM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape sourceIds =
        context->GetInputShape(SOURCE_TOKEN_IDS)->GetStorageShape();
    const gert::Shape copyCounts =
        context->GetInputShape(COPY_COUNTS)->GetStorageShape();

    if (!IsShape(query, {-1, -1, CKV_DIM}) ||
        query.GetDim(0) <= 0 ||
        query.GetDim(1) <= 0 ||
        query.GetDim(1) > MAX_REGULAR_LOCAL_HEADS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalQueryTokens = query.GetDim(0);
    const int64_t localHeads = query.GetDim(1);
    const int64_t batchSize = actualQ.GetDimNum() == 1
        ? actualQ.GetDim(0)
        : -1;
    if (batchSize <= 0 ||
        totalQueryTokens != batchSize ||
        !IsShape(hbmKv, {-1, BLOCK_SIZE, 1, CKV_DIM}) ||
        !IsShape(value, {hbmKv.GetDim(0), BLOCK_SIZE, 1, CKV_DIM}) ||
        !IsShape(hbmRope, {hbmKv.GetDim(0), BLOCK_SIZE, 1, KPE_DIM}) ||
        !IsShape(queryRope, {totalQueryTokens, localHeads, KPE_DIM}) ||
        !IsShape(slots, {totalQueryTokens, 1, SPARSE_COUNT}) ||
        !IsShape(hbmTable, {batchSize, -1}) ||
        !IsShape(actualKv, {batchSize}) ||
        !IsShape(cacheTokens, {batchSize}) ||
        !IsShape(dramKv, {-1, BLOCK_SIZE, CKV_DIM}) ||
        !IsShape(dramRope, {dramKv.GetDim(0), BLOCK_SIZE, KPE_DIM}) ||
        !IsShape(dramTable, {batchSize, -1}) ||
        !IsShape(sourceIds, {batchSize, SPARSE_COUNT}) ||
        !IsShape(copyCounts, {batchSize}) ||
        hbmTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) <= 0 ||
        hbmTable.GetDim(1) * BLOCK_SIZE > MAX_SOURCE_CAPACITY ||
        dramTable.GetDim(1) * BLOCK_SIZE > MAX_SOURCE_CAPACITY) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType floatingType =
        context->GetInputDesc(QUERY)->GetDataType();
    if (floatingType != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index :
         {KEY, VALUE, QUERY_ROPE, HBM_KEY_ROPE,
          DRAM_KEY_ROPE, DRAM_KV_CACHE}) {
        if (context->GetInputDesc(index)->GetDataType() != floatingType) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t index :
         {SPARSE_SLOTS, HBM_BLOCK_TABLE, ACTUAL_Q, ACTUAL_KV,
          CACHE_TOKENS, DRAM_BLOCK_TABLE, SOURCE_TOKEN_IDS, COPY_COUNTS}) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    copyCap = static_cast<uint32_t>(sourceIds.GetDim(1));
    dramMaxBlockNum = static_cast<uint32_t>(dramTable.GetDim(1));
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
namespace {
constexpr uint32_t PREFETCH_ROWS_ATTR_INDEX = 9;
constexpr uint32_t FUTURE_WORKSPACE_MAX_MISS = 400;
constexpr uint32_t S2_BASE_SIZE = 128;
constexpr uint32_t KV_ROW_ELEMENTS = 576;
constexpr uint32_t KV_ELEMENT_BYTES = 2;
constexpr uint32_t FUTURE_WORKSPACE_TILE_COUNT =
    (FUTURE_WORKSPACE_MAX_MISS + 2U * S2_BASE_SIZE - 2U) /
    S2_BASE_SIZE;
constexpr size_t GE_RUNTIME_APPEND_RESERVE_BYTES = 16U;
static_assert(FUTURE_WORKSPACE_TILE_COUNT == 5U);
} // namespace

static ge::graphStatus
TilingA5SparseAndTailAttentionAndScatterCopyMtePipeline(
    gert::TilingContext *context)
{
    SFATilingInfo sfaInfo;
    SFAInfoParser parser(context);
    if (parser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t copyCap = 0;
    uint32_t dramMaxBlockNum = 0;
    if (CheckFusedInputs(context, copyCap, dramMaxBlockNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *prefetchRowsAttr = attrs->GetAttrPointer<int64_t>(
        PREFETCH_ROWS_ATTR_INDEX);
    if (prefetchRowsAttr == nullptr || *prefetchRowsAttr < 0 ||
        *prefetchRowsAttr > 16) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t prefetchRowsPerStep =
        static_cast<uint32_t>(*prefetchRowsAttr);

    SFATilingCheck checker(sfaInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SFAMlaTiling tiler(context);
    if (tiler.DoOpTiling(&sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        sfaInfo.platformInfo);
    uint32_t futureWorkspaceOwnerCount = platform.GetCoreNumAic();
    if (sfaInfo.gSize > 64U) {
        futureWorkspaceOwnerCount >>= 1U;
    }
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t libapiWorkspaceBytes =
        platform.GetLibApiWorkSpaceSize();
    if (workspaceSizes[0] < libapiWorkspaceBytes ||
        workspaceSizes[0] - libapiWorkspaceBytes >
            std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t futureWorkspaceOffsetBytes =
        static_cast<uint32_t>(
            workspaceSizes[0] - libapiWorkspaceBytes);
    constexpr uint64_t futureTileBytes =
        static_cast<uint64_t>(S2_BASE_SIZE) * KV_ROW_ELEMENTS *
        KV_ELEMENT_BYTES;
    workspaceSizes[0] +=
        static_cast<uint64_t>(futureWorkspaceOwnerCount) *
        FUTURE_WORKSPACE_TILE_COUNT * futureTileBytes;

    auto *raw = context->GetRawTilingData();
    if (raw == nullptr) {
        return ge::GRAPH_FAILED;
    }
    A5SparseAndTailAttentionAndScatterCopyMtePipelineTilingData fusedTiling;
    const size_t baseSize = raw->GetDataSize();
    constexpr size_t suffixSize = sizeof(uint32_t) * 6U;
    const size_t payloadSize = baseSize + suffixSize;
    const size_t registeredSize = fusedTiling.GetDataSize();
    if (registeredSize <
            payloadSize + GE_RUNTIME_APPEND_RESERVE_BYTES ||
        raw->GetCapacity() <
            payloadSize + GE_RUNTIME_APPEND_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }
    auto *payload = static_cast<uint8_t *>(raw->GetData());
    std::memset(payload + baseSize, 0, suffixSize);
    std::memcpy(payload + baseSize, &copyCap, sizeof(copyCap));
    std::memcpy(
        payload + baseSize + sizeof(copyCap),
        &dramMaxBlockNum,
        sizeof(dramMaxBlockNum));
    std::memcpy(
        payload + baseSize + sizeof(copyCap) + sizeof(dramMaxBlockNum),
        &prefetchRowsPerStep,
        sizeof(prefetchRowsPerStep));
    std::memcpy(
        payload + baseSize + sizeof(copyCap) + sizeof(dramMaxBlockNum) +
            sizeof(prefetchRowsPerStep),
        &FUTURE_WORKSPACE_MAX_MISS,
        sizeof(FUTURE_WORKSPACE_MAX_MISS));
    std::memcpy(
        payload + baseSize + sizeof(copyCap) + sizeof(dramMaxBlockNum) +
            sizeof(prefetchRowsPerStep) + sizeof(FUTURE_WORKSPACE_MAX_MISS),
        &FUTURE_WORKSPACE_TILE_COUNT,
        sizeof(FUTURE_WORKSPACE_TILE_COUNT));
    std::memcpy(
        payload + baseSize + sizeof(copyCap) + sizeof(dramMaxBlockNum) +
            sizeof(prefetchRowsPerStep) + sizeof(FUTURE_WORKSPACE_MAX_MISS) +
            sizeof(FUTURE_WORKSPACE_TILE_COUNT),
        &futureWorkspaceOffsetBytes,
        sizeof(futureWorkspaceOffsetBytes));
    // GE appends atomic_index after the callback returns.
    raw->SetDataSize(payloadSize);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
static ge::graphStatus
InferShapeForA5SparseAndTailAttentionAndScatterCopyMtePipeline(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(QUERY) == nullptr ||
        context->GetInputShape(HBM_KEY_ROPE) == nullptr ||
        context->GetInputShape(KEY) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr ||
        context->GetOutputShape(3) == nullptr ||
        context->GetOutputShape(4) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(QUERY);
    // torch_npu::OpCommand cannot pass a REQUIRED zero-storage output to GE.
    // These one-element placeholders are unused when return_softmax_lse=false.
    *context->GetOutputShape(1) = gert::Shape({1});
    *context->GetOutputShape(2) = gert::Shape({1});
    *context->GetOutputShape(3) =
        *context->GetInputShape(HBM_KEY_ROPE);
    *context->GetOutputShape(4) =
        *context->GetInputShape(KEY);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus
InferDataTypeForA5SparseAndTailAttentionAndScatterCopyMtePipeline(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(
        0, context->GetInputDataType(QUERY));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(
        3, context->GetInputDataType(HBM_KEY_ROPE));
    context->SetOutputDataType(
        4, context->GetInputDataType(KEY));
    return ge::GRAPH_SUCCESS;
}

class A5SparseAndTailAttentionAndScatterCopyMtePipeline : public OpDef {
public:
    explicit A5SparseAndTailAttentionAndScatterCopyMtePipeline(
        const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {ge::DT_BF16};
        const std::vector<ge::DataType> intTypes = {ge::DT_INT32};
        const std::vector<ge::DataType> fp32Types = {ge::DT_FLOAT};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("key").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("value").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("sparse_indices").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("block_table").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("query_rope").ParamType(OPTIONAL).DataType(floatTypes).Format(formats);
        this->Input("key_rope").ParamType(OPTIONAL).DataType(floatTypes).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("dram_key_rope").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("dram_kv_cache").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("dram_block_table").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("source_token_ids").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("copy_counts").ParamType(REQUIRED).DataType(intTypes).Format(formats);

        this->Output("attention_out").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Output("softmax_max").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Output("softmax_sum").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Output("hbm_key_rope_out").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Output("hbm_kv_cache_out").ParamType(REQUIRED).DataType(floatTypes).Format(formats);

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
        this->Attr("prefetch_rows_per_step").AttrType(OPTIONAL).Int(5);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(
                optiling::
                    TilingA5SparseAndTailAttentionAndScatterCopyMtePipeline)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5SparseAndTailAttentionAndScatterCopyMtePipeline);

IMPL_OP_INFERSHAPE(A5SparseAndTailAttentionAndScatterCopyMtePipeline)
    .InferShape(
        InferShapeForA5SparseAndTailAttentionAndScatterCopyMtePipeline)
    .InferDataType(
        InferDataTypeForA5SparseAndTailAttentionAndScatterCopyMtePipeline);
} // namespace ops
