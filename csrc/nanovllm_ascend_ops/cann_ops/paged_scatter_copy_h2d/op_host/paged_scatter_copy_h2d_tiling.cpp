#include "paged_scatter_copy_h2d_tiling.h"

#include "log/ops_log.h"

namespace optiling {
static ge::graphStatus TilingForPagedScatterCopyH2d(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PagedScatterCopyH2d", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context->GetNodeName(), "PlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(aivNum == 0, OPS_LOG_E(context->GetNodeName(), "AIV core number is 0."),
               return ge::GRAPH_FAILED);

    auto tokenShape = context->GetInputShape(NPU_DST_TOKEN_INDEX);
    OPS_ERR_IF(tokenShape == nullptr, OPS_LOG_E(context->GetNodeName(), "npu_dst_token_index shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tokenShape->GetStorageShape().GetDimNum() != 2,
               OPS_LOG_E(context->GetNodeName(), "npu_dst_token_index must be [batch, token_count]."),
               return ge::GRAPH_FAILED);
    const uint32_t batchSize = static_cast<uint32_t>(tokenShape->GetStorageShape().GetDim(0));
    const uint32_t tokenCountPerBatch = static_cast<uint32_t>(tokenShape->GetStorageShape().GetDim(1));

    auto cpuTokenShape = context->GetInputShape(CPU_SRC_TOKEN_INDEX);
    auto copyCountsShape = context->GetInputShape(COPY_COUNTS_INDEX);
    auto npuTableShape = context->GetInputShape(NPU_BLOCK_TABLE_INDEX);
    auto cpuTableShape = context->GetInputShape(CPU_BLOCK_TABLE_INDEX);
    OPS_ERR_IF(cpuTokenShape == nullptr || copyCountsShape == nullptr ||
                   npuTableShape == nullptr || cpuTableShape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "token/table shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cpuTokenShape->GetStorageShape().GetDimNum() != 2 ||
                   copyCountsShape->GetStorageShape().GetDimNum() != 1 ||
                   npuTableShape->GetStorageShape().GetDimNum() != 2 ||
                   cpuTableShape->GetStorageShape().GetDimNum() != 2,
               OPS_LOG_E(context->GetNodeName(), "token/table inputs have invalid rank."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cpuTokenShape->GetStorageShape().GetDim(0) != batchSize ||
                   cpuTokenShape->GetStorageShape().GetDim(1) != tokenCountPerBatch ||
                   copyCountsShape->GetStorageShape().GetDim(0) != batchSize ||
                   npuTableShape->GetStorageShape().GetDim(0) != batchSize ||
                   cpuTableShape->GetStorageShape().GetDim(0) != batchSize,
               OPS_LOG_E(context->GetNodeName(), "batch/token dimensions mismatch."),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context->GetNodeName(), "Attrs is nullptr."),
               return ge::GRAPH_FAILED);
    auto kropeUnitBytesPtr = attrs->GetAttrPointer<int64_t>(ATTR_KROPE_UNIT_BYTES);
    auto knopeUnitBytesPtr = attrs->GetAttrPointer<int64_t>(ATTR_KNOPE_UNIT_BYTES);
    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE);
    OPS_ERR_IF(kropeUnitBytesPtr == nullptr || knopeUnitBytesPtr == nullptr || blockSizePtr == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Required attrs are nullptr."),
               return ge::GRAPH_FAILED);

    const uint32_t totalTasks = batchSize * tokenCountPerBatch;
    uint32_t blockDim = totalTasks == 0 ? 1 : totalTasks;
    if (blockDim > aivNum) {
        blockDim = aivNum;
    }
    if (blockDim > 16) {
        blockDim = 16;
    }
    context->SetBlockDim(blockDim);

    PagedScatterCopyH2dTilingData tilingData;
    tilingData.set_batchSize(batchSize);
    tilingData.set_tokenCountPerBatch(tokenCountPerBatch);
    tilingData.set_npuBlockTableWidth(static_cast<uint32_t>(npuTableShape->GetStorageShape().GetDim(1)));
    tilingData.set_cpuBlockTableWidth(static_cast<uint32_t>(cpuTableShape->GetStorageShape().GetDim(1)));
    tilingData.set_blockSize(static_cast<uint32_t>(*blockSizePtr));
    tilingData.set_kropeUnitBytes(static_cast<uint32_t>(*kropeUnitBytesPtr));
    tilingData.set_knopeUnitBytes(static_cast<uint32_t>(*knopeUnitBytesPtr));
    tilingData.set_usedCoreNum(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PagedScatterCopyH2d).Tiling(TilingForPagedScatterCopyH2d);
} // namespace optiling
