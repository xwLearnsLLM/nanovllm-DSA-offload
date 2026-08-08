/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "a5_fused_li_manage_mtp_tiling.h"
#include <algorithm>
#include "../op_kernel/a5_fused_li_manage_mtp_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus A5FusedLiManageMtpTiling::GetNpuInfo(LIMtpA5TilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("A5FusedLiManageMtp", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpTiling::GetTensorInfo(LIMtpA5TilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.cacheSlots.desc = context_->GetInputDesc(CACHE_SLOTS_INDEX);
    op.cacheSlots.shape = context_->GetInputShape(CACHE_SLOTS_INDEX);
    op.actualSeqLengthsQuery.desc = context_->GetInputDesc(ACTUAL_SEQ_Q_INDEX);
    op.actualSeqLengthsQuery.tensor = context_->GetInputTensor(ACTUAL_SEQ_Q_INDEX);
    op.actualSeqLengthsKey.desc = context_->GetInputDesc(ACTUAL_SEQ_K_INDEX);
    op.actualSeqLengthsKey.tensor = context_->GetInputTensor(ACTUAL_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INDEX);
    op.topkIndexOut.desc = context_->GetOutputDesc(TOPK_INDEX);
    op.topkIndexOut.shape = context_->GetOutputShape(TOPK_INDEX);
    op.topkSlotsOut.desc = context_->GetOutputDesc(TOPK_SLOTS_INDEX);
    op.topkSlotsOut.shape = context_->GetOutputShape(TOPK_SLOTS_INDEX);
    op.missIndexOut.desc = context_->GetOutputDesc(MISS_INDEX);
    op.missIndexOut.shape = context_->GetOutputShape(MISS_INDEX);
    op.missSlotsOut.desc = context_->GetOutputDesc(MISS_SLOTS_INDEX);
    op.missSlotsOut.shape = context_->GetOutputShape(MISS_SLOTS_INDEX);
    op.missCountOut.desc = context_->GetOutputDesc(MISS_COUNT_INDEX);
    op.missCountOut.shape = context_->GetOutputShape(MISS_COUNT_INDEX);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc == nullptr || op.cacheSlots.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsQuery.desc == nullptr || op.actualSeqLengthsQuery.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_query desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsKey.desc == nullptr || op.actualSeqLengthsKey.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/tensor is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc == nullptr || op.topkIndexOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_index desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkSlotsOut.desc == nullptr || op.topkSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missIndexOut.desc == nullptr || op.missIndexOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_index desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missSlotsOut.desc == nullptr || op.missSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missCountOut.desc == nullptr || op.missCountOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_count desc/shape is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpTiling::CheckDtype(const LIMtpA5TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    ge::DataType qType = op.query.desc->GetDataType();
    ge::DataType kType = op.key.desc->GetDataType();
    ge::DataType wType = op.weights.desc->GetDataType();
    OPS_ERR_IF(qType != kType || qType != wType,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must match."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qType != ge::DT_FLOAT16 && qType != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must be fp16 or bf16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_slots dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsQuery.desc->GetDataType() != ge::DT_INT32 ||
                   op.actualSeqLengthsKey.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "actual sequence length tensors must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "block_table dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.topkSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missIndexOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missCountOut.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "all outputs must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpTiling::CheckShape(LIMtpA5TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &cacheShape = op.cacheSlots.shape->GetStorageShape();
    const auto &seqQShape = op.actualSeqLengthsQuery.tensor->GetStorageShape();
    const auto &seqKShape = op.actualSeqLengthsKey.tensor->GetStorageShape();
    const auto &blockShape = op.blockTable.tensor->GetStorageShape();
    const auto &indexOutShape = op.topkIndexOut.shape->GetStorageShape();
    const auto &slotsOutShape = op.topkSlotsOut.shape->GetStorageShape();
    const auto &missIndexOutShape = op.missIndexOut.shape->GetStorageShape();
    const auto &missSlotsOutShape = op.missSlotsOut.shape->GetStorageShape();
    const auto &missCountOutShape = op.missCountOut.shape->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be packed TND [T, N1, 128], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights must be [T, N1], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots must be [B, 262144]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqQShape.GetDimNum() != DIM_NUM_ONE || seqKShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "actual sequence length tensors must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDimNum() != DIM_NUM_THREE || slotsOutShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must be [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missIndexOutShape.GetDimNum() != DIM_NUM_TWO ||
                   missSlotsOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "miss_index/miss_slots must be [B, 8192]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "miss_count must be [B]."), return ge::GRAPH_FAILED);

    tilingInfo.tSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.bSize = static_cast<uint32_t>(cacheShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(1));
    tilingInfo.n2Size = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_TWO));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = tilingInfo.blockSize * tilingInfo.maxBlockNumPerBatch;

    OPS_ERR_IF(tilingInfo.bSize == 0 || tilingInfo.tSize == 0,
               OPS_LOG_E(tilingInfo.opName, "B and packed T must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqQShape.GetShapeSize() != tilingInfo.bSize ||
                   seqKShape.GetShapeSize() != tilingInfo.bSize ||
                   blockShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName,
                         "cache_slots, both sequence-length tensors, and block_table must share B."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDim(0) != tilingInfo.bSize || cacheShape.GetDim(1) != CACHE_SLOTS_SIZE,
               OPS_LOG_E(tilingInfo.opName, "cache_slots must have shape [B, 262144]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize == 0 || tilingInfo.blockSize > 1024 || tilingInfo.blockSize % 16 != 0,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be a multiple of 16 in (0, 1024]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.s2Size > CACHE_SLOTS_SIZE,
               OPS_LOG_E(tilingInfo.opName, "maxBlockNumPerBatch * blockSize must be <= 262144."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n2Size != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != DECODE_G_SIZE_32 && tilingInfo.n1Size != DECODE_G_SIZE_64,
               OPS_LOG_E(tilingInfo.opName, "decode query N1 must be 32 or 64."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.tSize || wShape.GetDim(1) != tilingInfo.n1Size,
               OPS_LOG_E(tilingInfo.opName, "weights must match query [T, N1]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.tSize < tilingInfo.bSize ||
                   tilingInfo.tSize > tilingInfo.bSize * MTP_MAX_QUERY_COUNT,
               OPS_LOG_E(tilingInfo.opName, "packed T must be in [B, 4 * B]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDim(0) != tilingInfo.tSize || indexOutShape.GetDim(1) != DECODE_N2 ||
                   indexOutShape.GetDim(2) != DECODE_SPARSE_COUNT ||
                   slotsOutShape.GetDim(0) != tilingInfo.tSize || slotsOutShape.GetDim(1) != DECODE_N2 ||
                   slotsOutShape.GetDim(2) != DECODE_SPARSE_COUNT,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must have shape [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missIndexOutShape.GetDim(0) != tilingInfo.bSize ||
                   missIndexOutShape.GetDim(1) != MTP_CACHE_SIZE ||
                   missSlotsOutShape.GetDim(0) != tilingInfo.bSize ||
                   missSlotsOutShape.GetDim(1) != MTP_CACHE_SIZE,
               OPS_LOG_E(tilingInfo.opName, "miss_index/miss_slots must have shape [B, 8192]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName, "miss_count must have shape [B]."),
               return ge::GRAPH_FAILED);

    tilingInfo.inputQType = op.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpTiling::ParseAndCheck(LIMtpA5TilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpTiling::DoTiling(LIMtpA5TilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingInfo->usedCoreNum = aicNum;
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t S1_BASE_SIZE = 4;
    constexpr uint32_t S2_BASE_SIZE = 128;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += S1_BASE_SIZE *
        ((tilingInfo->s2Size + S2_BASE_SIZE - 1) / S2_BASE_SIZE) *
        S2_BASE_SIZE * sizeof(uint16_t) * aicNum;
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_n2Size(DECODE_N2);
    tilingData_.set_gSize(tilingInfo->n1Size);
    tilingData_.set_s1Size(MTP_MAX_QUERY_COUNT);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_sparseCount(DECODE_SPARSE_COUNT);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_usedCoreNum(aicNum);
    tilingData_.set_sparseMode(3);
    tilingData_.set_preTokens(INT64_MAX);
    tilingData_.set_nextTokens(INT64_MAX);
    tilingData_.set_returnValue(0);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(tilingInfo->inputQType));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForA5FusedLiManageMtp(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForA5FusedLiManageMtp(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("A5FusedLiManageMtp", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    LIMtpA5TilingInfo liInfo;
    A5FusedLiManageMtpTiling liTiling(context);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(A5FusedLiManageMtp)
    .Tiling(TilingForA5FusedLiManageMtp)
    .TilingParse<LIMtpA5CompileInfo>(TilingPrepareForA5FusedLiManageMtp);

} // namespace optiling
