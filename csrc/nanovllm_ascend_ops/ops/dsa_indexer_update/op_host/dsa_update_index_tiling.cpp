#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "dsa_update_index_tiling_data.h"

namespace optiling {

constexpr size_t WORKSPACE_NUM = 1;
constexpr size_t DSA_WORKSPACE_LIST_NUM = 2;
constexpr size_t DSA_WORKSPACE_PAIR_FLOATS = DSA_UPDATE_INDEX_MAX_K * 2;

static ge::graphStatus DsaUpdateIndexTilingFunc(gert::TilingContext* context)
{
    // --- platform info ---
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t platformCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (platformCoreNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    // --- k attribute ---
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    if (kPtr == nullptr || *kPtr <= 0 || *kPtr > DSA_UPDATE_INDEX_MAX_K) {
        return ge::GRAPH_FAILED;
    }

    // --- input shapes ---
    auto scoreStorage = context->GetInputShape(0);
    auto selectedStorage = context->GetInputShape(1);
    auto seqLenStorage = context->GetInputShape(2);
    auto selectedLenStorage = context->GetInputShape(3);
    auto promoteStorage = context->GetOutputShape(0);
    auto demoteStorage = context->GetOutputShape(1);
    if (!scoreStorage || !selectedStorage || !seqLenStorage ||
        !selectedLenStorage || !promoteStorage || !demoteStorage) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape scoreShape = scoreStorage->GetStorageShape();
    const gert::Shape selectedShape = selectedStorage->GetStorageShape();
    const gert::Shape seqLenShape = seqLenStorage->GetStorageShape();
    const gert::Shape selectedLenShape = selectedLenStorage->GetStorageShape();
    const gert::Shape promoteShape = promoteStorage->GetStorageShape();
    const gert::Shape demoteShape = demoteStorage->GetStorageShape();

    // --- rank checks ---
    if (scoreShape.GetDimNum() != 2 || selectedShape.GetDimNum() != 2 ||
        seqLenShape.GetDimNum() != 1 || selectedLenShape.GetDimNum() != 1 ||
        promoteShape.GetDimNum() != 2 || demoteShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = scoreShape.GetDim(0);
    const int64_t maxSeqLen = scoreShape.GetDim(1);
    const int64_t maxSelectedLen = selectedShape.GetDim(1);
    if (batchSize <= 0 || maxSeqLen <= 0 || maxSelectedLen <= 0) {
        return ge::GRAPH_FAILED;
    }

    // --- shape consistency ---
    if (selectedShape.GetDim(0) != batchSize ||
        seqLenShape.GetDim(0) != batchSize ||
        selectedLenShape.GetDim(0) != batchSize ||
        promoteShape.GetDim(0) != batchSize ||
        demoteShape.GetDim(0) != batchSize ||
        promoteShape.GetDim(1) != *kPtr ||
        demoteShape.GetDim(1) != *kPtr) {
        return ge::GRAPH_FAILED;
    }

    // --- dtype checks ---
    auto scoreDesc = context->GetInputDesc(0);
    auto selectedDesc = context->GetInputDesc(1);
    auto seqLenDesc = context->GetInputDesc(2);
    auto selectedLenDesc = context->GetInputDesc(3);
    auto promoteDesc = context->GetOutputDesc(0);
    auto demoteDesc = context->GetOutputDesc(1);
    if (!scoreDesc || !selectedDesc || !seqLenDesc ||
        !selectedLenDesc || !promoteDesc || !demoteDesc) {
        return ge::GRAPH_FAILED;
    }

    if (scoreDesc->GetDataType() != ge::DT_BF16 ||
        selectedDesc->GetDataType() != ge::DT_INT32 ||
        seqLenDesc->GetDataType() != ge::DT_INT32 ||
        selectedLenDesc->GetDataType() != ge::DT_INT32 ||
        promoteDesc->GetDataType() != ge::DT_INT32 ||
        demoteDesc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    // --- tiling data ---
    DsaUpdateIndexTilingData* tiling = context->GetTilingData<DsaUpdateIndexTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::memset(tiling, 0, sizeof(DsaUpdateIndexTilingData));

    int64_t usedCoreNum = (batchSize < platformCoreNum) ? batchSize : platformCoreNum;
    const int64_t coreNumPerBatch = 1;

    // --- workspace size ---
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() +
        static_cast<size_t>(usedCoreNum) * DSA_WORKSPACE_LIST_NUM *
            DSA_WORKSPACE_PAIR_FLOATS * sizeof(float);

    tiling->batchSize = batchSize;
    tiling->maxSeqLen = maxSeqLen;
    tiling->maxSelectedLen = maxSelectedLen;
    tiling->k = *kPtr;
    tiling->usedCoreNum = usedCoreNum;
    tiling->coreNumPerBatch = coreNumPerBatch;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDsaUpdateIndex(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct DsaUpdateIndexCompileInfo {};

IMPL_OP_OPTILING(DsaUpdateIndex)
    .Tiling(DsaUpdateIndexTilingFunc)
    .TilingParse<DsaUpdateIndexCompileInfo>(TilingParseForDsaUpdateIndex);

}  // namespace optiling
