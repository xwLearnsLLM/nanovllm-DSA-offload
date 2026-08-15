/** Host registration and tiling for one-kernel A5 C8 MTP LI + management. */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../op_kernel/a5_fused_li_manage_mtp_c8_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    QUERY = 0,
    KEY,
    WEIGHTS,
    QUERY_DEQUANT_SCALE,
    KEY_DEQUANT_SCALE,
    ACTUAL_SEQ_LENGTHS_QUERY,
    REQ_POOL_ENTRIES,
    CACHE_SLOTS_POOL,
    CACHE_TOKENS,
    CANDIDATE_LENS,
    BLOCK_TABLE,
};

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_QUERIES_PER_REQUEST = 4;
constexpr int64_t MIN_QUERIES_PER_REQUEST = 2;
constexpr int64_t UNION_CAPACITY =
    SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;

bool IsShape(
    const gert::Shape &shape, std::initializer_list<int64_t> dimensions)
{
    if (shape.GetDimNum() != dimensions.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t expected : dimensions) {
        if (expected >= 0 && shape.GetDim(index) != expected) {
            return false;
        }
        ++index;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingA5FusedLiManageMtpC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = QUERY; index <= BLOCK_TABLE; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto dtype = [context](size_t index) {
        return context->GetInputDesc(index)->GetDataType();
    };
    if (dtype(QUERY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(KEY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(WEIGHTS) != ge::DT_BF16 ||
        dtype(QUERY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(KEY_DEQUANT_SCALE) != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {
             ACTUAL_SEQ_LENGTHS_QUERY, REQ_POOL_ENTRIES,
             CACHE_SLOTS_POOL, CACHE_TOKENS, CANDIDATE_LENS,
             BLOCK_TABLE}) {
        if (dtype(index) != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape query =
        context->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape key =
        context->GetInputShape(KEY)->GetStorageShape();
    const gert::Shape weights =
        context->GetInputShape(WEIGHTS)->GetStorageShape();
    const gert::Shape queryScale =
        context->GetInputShape(QUERY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape keyScale =
        context->GetInputShape(KEY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const gert::Shape entries =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape pool =
        context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape lengths =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape blockTable =
        context->GetInputShape(BLOCK_TABLE)->GetStorageShape();

    if (!IsShape(query, {-1, -1, HEAD_DIM}) ||
        !IsShape(key, {-1, BLOCK_SIZE, 1, HEAD_DIM}) ||
        !IsShape(weights, {-1, -1}) ||
        !IsShape(queryScale, {-1, -1}) ||
        !IsShape(keyScale, {-1, BLOCK_SIZE, 1}) ||
        !IsShape(actualQ, {-1}) || !IsShape(entries, {-1}) ||
        !IsShape(pool, {-1, -1}) || !IsShape(budgets, {-1}) ||
        !IsShape(lengths, {-1}) || !IsShape(blockTable, {-1, -1})) {
        return ge::GRAPH_FAILED;
    }

    const int64_t packedQueries = query.GetDim(0);
    const int64_t heads = query.GetDim(1);
    const int64_t batch = actualQ.GetDim(0);
    const int64_t sourceCapacity = pool.GetDim(1);
    if (batch <= 0 || packedQueries <= 0 ||
        packedQueries < batch * MIN_QUERIES_PER_REQUEST ||
        packedQueries > batch * MAX_QUERIES_PER_REQUEST ||
        (heads != 32 && heads != 64) || key.GetDim(0) <= 0 ||
        weights.GetDim(0) != packedQueries || weights.GetDim(1) != heads ||
        queryScale.GetDim(0) != packedQueries ||
        queryScale.GetDim(1) != heads ||
        keyScale.GetDim(0) != key.GetDim(0) ||
        entries.GetDim(0) != batch || budgets.GetDim(0) != batch ||
        lengths.GetDim(0) != batch || blockTable.GetDim(0) != batch ||
        blockTable.GetDim(1) <= 0 || pool.GetDim(0) <= 0 ||
        sourceCapacity <= 0 || sourceCapacity > MAX_SOURCE_CAPACITY ||
        blockTable.GetDim(1) * BLOCK_SIZE != sourceCapacity) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aicCount == 0 || aivCount < 2) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = std::min<uint32_t>(
        static_cast<uint32_t>(batch), aicCount);
    const uint32_t s1BaseSize =
        (256U + static_cast<uint32_t>(heads) - 1U) /
        static_cast<uint32_t>(heads);
    const uint64_t scoreStride64 =
        static_cast<uint64_t>(s1BaseSize) * sourceCapacity *
        sizeof(uint16_t);
    if (scoreStride64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling =
        context->GetTilingData<A5FusedLiManageMtpC8TilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->packedQueryCount = static_cast<uint32_t>(packedQueries);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch =
        static_cast<uint32_t>(blockTable.GetDim(1));
    tiling->maxCandidateLen = static_cast<uint32_t>(sourceCapacity);
    tiling->keyStride = BLOCK_SIZE * HEAD_DIM;
    tiling->scaleStride = BLOCK_SIZE;
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreStride64);

    const uint64_t topkWorkspaceBytes =
        static_cast<uint64_t>(packedQueries) * SPARSE_COUNT *
        sizeof(int32_t);
    context->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() +
        scoreStride64 * static_cast<uint64_t>(usedCoreNum) +
        topkWorkspaceBytes;
    context->SetBlockDim(platform.CalcTschBlockDim(
        usedCoreNum * 2U, usedCoreNum, usedCoreNum * 2U));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferA5FusedLiManageMtpC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY) == nullptr ||
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY) == nullptr ||
        context->GetInputShape(CACHE_SLOTS_POOL) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 5; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t packedQueries =
        context->GetInputShape(QUERY)->GetDim(0);
    const int64_t batch =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetDim(0);
    *context->GetOutputShape(0) =
        gert::Shape({packedQueries, 1, SPARSE_COUNT});
    *context->GetOutputShape(1) =
        gert::Shape({batch, UNION_CAPACITY});
    *context->GetOutputShape(2) =
        gert::Shape({batch, UNION_CAPACITY});
    *context->GetOutputShape(3) = gert::Shape({batch});
    *context->GetOutputShape(4) =
        *context->GetInputShape(CACHE_SLOTS_POOL);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5FusedLiManageMtpC8DataType(
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

class A5FusedLiManageMtpC8 : public OpDef {
public:
    explicit A5FusedLiManageMtpC8(const char *name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(formats);
        this->Input("key").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(formats);
        this->Input("weights").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format(formats);
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(formats);
        this->Input("key_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("cache_slots_pool").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("block_table").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("topk_destination_slots").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("miss_source_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("miss_destination_slots").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("cache_slots_alias").ParamType(REQUIRED)
            .DataType(ints).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5FusedLiManageMtpC8)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5FusedLiManageMtpC8);

IMPL_OP_INFERSHAPE(A5FusedLiManageMtpC8)
    .InferShape(InferA5FusedLiManageMtpC8Shape)
    .InferDataType(InferA5FusedLiManageMtpC8DataType);
} // namespace ops
