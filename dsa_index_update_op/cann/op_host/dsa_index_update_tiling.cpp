#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "dsa_index_update_tiling_data.h"

namespace optiling {

constexpr size_t WORKSPACE_NUM = 1;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum <= 0,
        OP_LOGE(context, "DsaIndexUpdate: coreNum is invalid: %ld.", *coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRank(gert::TilingContext* context, const gert::Shape& shape,
    size_t expectedRank, const char* name)
{
    OP_CHECK_IF(shape.GetDimNum() != expectedRank,
        OP_LOGE(context, "DsaIndexUpdate: %s rank must be %zu, got %zu.",
            name, expectedRank, shape.GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static gert::Shape PickShape(const gert::StorageShape* storage, size_t expectedRank)
{
    const gert::Shape originShape = storage->GetOriginShape();
    if (originShape.GetDimNum() == expectedRank) {
        return originShape;
    }
    return storage->GetStorageShape();
}

static ge::graphStatus DsaIndexUpdateTilingFunc(gert::TilingContext* context)
{
    int64_t platformCoreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &platformCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DsaIndexUpdate: GetPlatformInfo failed."),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    OP_CHECK_IF(*kPtr <= 0 || *kPtr > DSA_INDEX_UPDATE_MAX_K,
        OP_LOGE(context, "DsaIndexUpdate: max_copy_tokens must be in (0, %d], got %ld.",
            DSA_INDEX_UPDATE_MAX_K, *kPtr),
        return ge::GRAPH_FAILED);

    auto scoreStorage = context->GetInputShape(0);
    auto poolStorage = context->GetInputShape(1);
    auto candidateLensStorage = context->GetInputShape(2);
    auto selectedLensStorage = context->GetInputShape(3);
    auto reqPoolEntriesStorage = context->GetInputShape(4);
    auto promoteStorage = context->GetOutputShape(0);
    auto demoteStorage = context->GetOutputShape(1);
    auto copyCountsStorage = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateLensStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedLensStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, reqPoolEntriesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, promoteStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, demoteStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, copyCountsStorage);

    const gert::Shape scoreShape = PickShape(scoreStorage, 2);
    const gert::Shape poolShape = PickShape(poolStorage, 2);
    const gert::Shape candidateLensShape = PickShape(candidateLensStorage, 1);
    const gert::Shape selectedLensShape = PickShape(selectedLensStorage, 1);
    const gert::Shape reqPoolEntriesShape = PickShape(reqPoolEntriesStorage, 1);
    const gert::Shape promoteShape = PickShape(promoteStorage, 2);
    const gert::Shape demoteShape = PickShape(demoteStorage, 2);
    const gert::Shape copyCountsShape = PickShape(copyCountsStorage, 1);

    OP_CHECK_IF(CheckRank(context, scoreShape, 2, "score") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, poolShape, 2, "hbm_cached_tokens_pool") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, candidateLensShape, 1, "candidate_lens") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, selectedLensShape, 1, "selected_lens") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, reqPoolEntriesShape, 1, "req_pool_entries") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, promoteShape, 2, "promote_idx") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, demoteShape, 2, "demote_idx") != ge::GRAPH_SUCCESS ||
                    CheckRank(context, copyCountsShape, 1, "copy_counts") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DsaIndexUpdate: rank check failed."),
        return ge::GRAPH_FAILED);

    const int64_t batchSize = scoreShape.GetDim(0);
    const int64_t maxSeqLen = scoreShape.GetDim(1);
    const int64_t poolCapacity = poolShape.GetDim(0);
    const int64_t maxSelectedLen = poolShape.GetDim(1);
    OP_CHECK_IF(batchSize <= 0 || maxSeqLen <= 0 || poolCapacity <= 0 || maxSelectedLen <= 0,
        OP_LOGE(context, "DsaIndexUpdate: invalid shape values."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(candidateLensShape.GetDim(0) != batchSize ||
                    selectedLensShape.GetDim(0) != batchSize ||
                    reqPoolEntriesShape.GetDim(0) != batchSize ||
                    promoteShape.GetDim(0) != batchSize ||
                    demoteShape.GetDim(0) != batchSize ||
                    copyCountsShape.GetDim(0) != batchSize ||
                    promoteShape.GetDim(1) < *kPtr ||
                    demoteShape.GetDim(1) != promoteShape.GetDim(1),
        OP_LOGE(context, "DsaIndexUpdate: shape mismatch."),
        return ge::GRAPH_FAILED);

    auto scoreDesc = context->GetInputDesc(0);
    auto poolDesc = context->GetInputDesc(1);
    auto candidateLensDesc = context->GetInputDesc(2);
    auto selectedLensDesc = context->GetInputDesc(3);
    auto reqPoolEntriesDesc = context->GetInputDesc(4);
    auto promoteDesc = context->GetOutputDesc(0);
    auto demoteDesc = context->GetOutputDesc(1);
    auto copyCountsDesc = context->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateLensDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedLensDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, reqPoolEntriesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, promoteDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, demoteDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, copyCountsDesc);

    OP_CHECK_IF(scoreDesc->GetDataType() != ge::DT_BF16 ||
                    poolDesc->GetDataType() != ge::DT_INT32 ||
                    candidateLensDesc->GetDataType() != ge::DT_INT32 ||
                    selectedLensDesc->GetDataType() != ge::DT_INT32 ||
                    reqPoolEntriesDesc->GetDataType() != ge::DT_INT32 ||
                    promoteDesc->GetDataType() != ge::DT_INT32 ||
                    demoteDesc->GetDataType() != ge::DT_INT32 ||
                    copyCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context, "DsaIndexUpdate: dtype mismatch."),
        return ge::GRAPH_FAILED);

    DsaIndexUpdateTilingData* tiling = context->GetTilingData<DsaIndexUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(DsaIndexUpdateTilingData), 0, sizeof(DsaIndexUpdateTilingData)) != EOK,
        OP_LOGE(context, "DsaIndexUpdate: memset tiling failed."),
        return ge::GRAPH_FAILED);

    int64_t usedCoreNum = batchSize < platformCoreNum ? batchSize : platformCoreNum;
    const int64_t coreNumPerBatch = 1;

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DsaIndexUpdate: GetWorkspaceSize failed."),
        return ge::GRAPH_FAILED);

    tiling->batchSize = batchSize;
    tiling->maxSeqLen = maxSeqLen;
    tiling->maxSelectedLen = maxSelectedLen;
    tiling->poolCapacity = poolCapacity;
    tiling->maxOutputLen = promoteShape.GetDim(1);
    tiling->k = *kPtr;
    tiling->usedCoreNum = usedCoreNum;
    tiling->coreNumPerBatch = coreNumPerBatch;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDsaIndexUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DsaIndexUpdateCompileInfo {};

IMPL_OP_OPTILING(DsaIndexUpdate)
    .Tiling(DsaIndexUpdateTilingFunc)
    .TilingParse<DsaIndexUpdateCompileInfo>(TilingParseForDsaIndexUpdate);

} // namespace optiling
