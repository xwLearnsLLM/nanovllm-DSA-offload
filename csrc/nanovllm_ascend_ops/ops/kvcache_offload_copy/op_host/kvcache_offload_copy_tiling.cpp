/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kvcache_offload_copy_tiling.h"

namespace optiling {

constexpr int32_t HBM_KV_CACHE_IDX = 0;
constexpr int32_t DRAM_KV_CACHE_IDX = 1;
constexpr int32_t HBM_BLOCK_TABLE_IDX = 2;
constexpr int32_t DRAM_BLOCK_TABLE_IDX = 3;
constexpr int32_t COPY_COUNTS_IDX = 4;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr int64_t MAX_COPY_CAP = 65536;
constexpr int64_t MAX_TILE_BYTES = 32 * 1024;
constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

ge::graphStatus KvcacheOffloadCopyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheOffloadCopyTiling::GetShapeInfo()
{
    auto hbmKv = context_->GetInputShape(HBM_KV_CACHE_IDX);
    auto dramKv = context_->GetInputShape(DRAM_KV_CACHE_IDX);
    auto hbmTable = context_->GetInputShape(HBM_BLOCK_TABLE_IDX);
    auto dramTable = context_->GetInputShape(DRAM_BLOCK_TABLE_IDX);
    auto copyCounts = context_->GetInputShape(COPY_COUNTS_IDX);
    OPS_ERR_IF(hbmKv == nullptr || dramKv == nullptr || hbmTable == nullptr ||
                   dramTable == nullptr || copyCounts == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get input shape nullptr."), return ge::GRAPH_FAILED);

    gert::Shape hbmKvShape = hbmKv->GetStorageShape();
    gert::Shape dramKvShape = dramKv->GetStorageShape();
    gert::Shape hbmTableShape = hbmTable->GetStorageShape();
    gert::Shape dramTableShape = dramTable->GetStorageShape();
    gert::Shape copyCountsShape = copyCounts->GetStorageShape();

    OPS_ERR_IF(hbmKvShape.GetDimNum() < 2 || dramKvShape.GetDimNum() != hbmKvShape.GetDimNum(),
        OPS_LOG_E(context_->GetNodeName(), "cache tensors must have matching rank >= 2."),
        return ge::GRAPH_FAILED);
    for (size_t dim = 1; dim < hbmKvShape.GetDimNum(); ++dim) {
        OPS_ERR_IF(hbmKvShape.GetDim(dim) != dramKvShape.GetDim(dim),
            OPS_LOG_E(context_->GetNodeName(),
                "cache tensors must have matching trailing dimensions."),
            return ge::GRAPH_FAILED);
    }

    int64_t hbmBlockCount = hbmKvShape.GetDim(DIM_0);
    int64_t dramBlockCount = dramKvShape.GetDim(DIM_0);
    OPS_ERR_IF(hbmBlockCount <= 0 || dramBlockCount <= 0,
        OPS_LOG_E(context_->GetNodeName(), "cache physical block counts must be positive."),
        return ge::GRAPH_FAILED);
    int64_t hbmElements = hbmKvShape.GetShapeSize();
    int64_t dramElements = dramKvShape.GetShapeSize();
    OPS_ERR_IF(hbmElements <= 0 || dramElements <= 0 || hbmElements % hbmBlockCount != 0 ||
                   dramElements % dramBlockCount != 0,
        OPS_LOG_E(context_->GetNodeName(), "cache shapes must describe complete non-empty blocks."),
        return ge::GRAPH_FAILED);
    blockBytes_ = hbmElements / hbmBlockCount;
    OPS_ERR_IF(dramElements / dramBlockCount != blockBytes_,
        OPS_LOG_E(context_->GetNodeName(), "HBM and DRAM cache block sizes must match."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(hbmTableShape.GetDimNum() != 2 || dramTableShape.GetDimNum() != 2 ||
                   copyCountsShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(),
            "block tables must be [B, C] and copy_counts must be [B]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(hbmTableShape.GetDim(DIM_0) != dramTableShape.GetDim(DIM_0) ||
                   hbmTableShape.GetDim(DIM_1) != dramTableShape.GetDim(DIM_1),
        OPS_LOG_E(context_->GetNodeName(), "HBM and DRAM block tables must have identical shapes."),
        return ge::GRAPH_FAILED);

    batchSize_ = copyCountsShape.GetDim(DIM_0);
    copyCap_ = hbmTableShape.GetDim(DIM_1);
    OPS_ERR_IF(hbmTableShape.GetDim(DIM_0) != batchSize_,
        OPS_LOG_E(context_->GetNodeName(),
            "block-table and copy_counts batch dimensions must match."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(copyCap_ <= 0 || copyCap_ > MAX_COPY_CAP,
        OPS_LOG_E(context_->GetNodeName(), "block-table capacity must be in [1, 65536]."),
        return ge::GRAPH_FAILED);

    int64_t logicalTileBytes = blockBytes_ < MAX_TILE_BYTES ? blockBytes_ : MAX_TILE_BYTES;
    int64_t tileBytes =
        ((logicalTileBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES) * DATA_BLOCK_BYTES;
    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_copyCap(copyCap_);
    tilingData_.set_blockBytes(blockBytes_);
    tilingData_.set_hbmBlockCount(hbmBlockCount);
    tilingData_.set_dramBlockCount(dramBlockCount);
    tilingData_.set_tileBytes(tileBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheOffloadCopyTiling::GetDtypeInfo()
{
    for (int32_t i = HBM_KV_CACHE_IDX; i <= DRAM_KV_CACHE_IDX; ++i) {
        auto desc = context_->GetInputDesc(i);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT8,
            OPS_LOG_E(context_->GetNodeName(), "HBM and DRAM KVCache tensors must be int8."),
            return ge::GRAPH_FAILED);
    }
    for (int32_t i = HBM_BLOCK_TABLE_IDX; i <= COPY_COUNTS_IDX; ++i) {
        auto desc = context_->GetInputDesc(i);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT32,
            OPS_LOG_E(context_->GetNodeName(), "block tables and copy_counts must be int32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheOffloadCopyTiling::DoOpTiling()
{
    if (batchSize_ == 0) {
        tilingData_.set_usedCoreNum(0);
        tilingData_.set_totalPairSlots(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t totalPairSlots = batchSize_ * copyCap_;
    int64_t usedCoreNum = totalPairSlots < coreNum_ ? totalPairSlots : coreNum_;
    tilingData_.set_totalPairSlots(totalPairSlots);
    tilingData_.set_usedCoreNum(usedCoreNum);
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheOffloadCopyTiling::PostTiling()
{
    context_->SetTilingKey(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get tilingdata nullptr."),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheOffloadCopyTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetDtypeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4KvcacheOffloadCopy(gert::TilingContext* context)
{
    KvcacheOffloadCopyTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4KvcacheOffloadCopy(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NanovllmKvcacheOffloadCopy)
    .Tiling(Tiling4KvcacheOffloadCopy)
    .TilingParse<KvcacheOffloadCopyCompileInfo>(TilingPrepare4KvcacheOffloadCopy);

} // namespace optiling
