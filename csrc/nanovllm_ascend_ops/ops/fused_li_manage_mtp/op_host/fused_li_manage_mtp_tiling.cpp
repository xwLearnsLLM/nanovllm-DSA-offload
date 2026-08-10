/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "fused_li_manage_mtp_tiling.h"
#include <algorithm>
#include "../op_kernel/fused_li_manage_mtp_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus LIUMtpTiling::GetPlatform(LIUMtpTilingInfo &info) const
{
    info.opName = context_->GetNodeName();
    OPS_ERR_IF(info.opName == nullptr,
               OPS_LOG_E("NanovllmFusedLiManageMtp", "node name is nullptr."),
               return ge::GRAPH_FAILED);
    info.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(info.platformInfo == nullptr,
               OPS_LOG_E(info.opName, "platform info is nullptr."),
               return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(info.platformInfo);
    OPS_ERR_IF(platform.GetCoreNumAic() == 0 || platform.GetCoreNumAiv() == 0,
               OPS_LOG_E(info.opName, "AI Core count is zero."),
               return ge::GRAPH_FAILED);
    info.socVersion = platform.GetSocVersion();
    OPS_ERR_IF(info.socVersion != platform_ascendc::SocVersion::ASCEND910B &&
                   info.socVersion != platform_ascendc::SocVersion::ASCEND910_93,
               OPS_LOG_E(info.opName, "unsupported SoC version %d.",
                         static_cast<int32_t>(info.socVersion)),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr ||
                   context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(info.opName, "workspace or tiling buffer is nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIUMtpTiling::GetTensors(LIUMtpTilingInfo &info) const
{
    auto &t = info.tensors;
#define LOAD_INPUT(field, index)                                                \
    do {                                                                         \
        t.field.desc = context_->GetInputDesc(index);                            \
        t.field.shape = context_->GetInputShape(index);                          \
        OPS_ERR_IF(t.field.desc == nullptr || t.field.shape == nullptr,          \
                   OPS_LOG_E(info.opName, #field " input is nullptr."),          \
                   return ge::GRAPH_FAILED);                                    \
    } while (0)
#define LOAD_OUTPUT(field, index)                                               \
    do {                                                                         \
        t.field.desc = context_->GetOutputDesc(index);                           \
        t.field.shape = context_->GetOutputShape(index);                         \
        OPS_ERR_IF(t.field.desc == nullptr || t.field.shape == nullptr,          \
                   OPS_LOG_E(info.opName, #field " output is nullptr."),         \
                   return ge::GRAPH_FAILED);                                    \
    } while (0)
    LOAD_INPUT(query, MTP_QUERY_INDEX);
    LOAD_INPUT(key, MTP_KEY_INDEX);
    LOAD_INPUT(weights, MTP_WEIGHTS_INDEX);
    LOAD_INPUT(reqPoolEntries, MTP_REQ_POOL_INDEX);
    LOAD_INPUT(cacheSlots, MTP_CACHE_SLOTS_INDEX);
    LOAD_INPUT(cacheTokens, MTP_CACHE_TOKENS_INDEX);
    LOAD_INPUT(candidateLens, MTP_CANDIDATE_LENS_INDEX);
    LOAD_INPUT(blockTable, MTP_BLOCK_TABLE_INDEX);
    LOAD_OUTPUT(topkSlots, MTP_TOPK_SLOTS_OUT);
    LOAD_OUTPUT(topkSource, MTP_TOPK_SOURCE_OUT);
    LOAD_OUTPUT(missSource, MTP_MISS_SOURCE_OUT);
    LOAD_OUTPUT(missSlots, MTP_MISS_SLOTS_OUT);
    LOAD_OUTPUT(missCounts, MTP_MISS_COUNTS_OUT);
    LOAD_OUTPUT(cacheSlotsOut, MTP_CACHE_SLOTS_OUT);
#undef LOAD_INPUT
#undef LOAD_OUTPUT
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIUMtpTiling::CheckDtypes(const LIUMtpTilingInfo &info) const
{
    const auto &t = info.tensors;
    ge::DataType queryType = t.query.desc->GetDataType();
    OPS_ERR_IF(queryType != ge::DT_BF16 && queryType != ge::DT_FLOAT16,
               OPS_LOG_E(info.opName, "query must be bf16 or fp16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(t.key.desc->GetDataType() != queryType ||
                   t.weights.desc->GetDataType() != queryType,
               OPS_LOG_E(info.opName, "query/key/weights dtype must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(t.reqPoolEntries.desc->GetDataType() != ge::DT_INT32 ||
                   t.cacheSlots.desc->GetDataType() != ge::DT_INT32 ||
                   t.cacheTokens.desc->GetDataType() != ge::DT_INT32 ||
                   t.candidateLens.desc->GetDataType() != ge::DT_INT32 ||
                   t.blockTable.desc->GetDataType() != ge::DT_INT32 ||
                   t.topkSlots.desc->GetDataType() != ge::DT_INT32 ||
                   t.topkSource.desc->GetDataType() != ge::DT_INT32 ||
                   t.missSource.desc->GetDataType() != ge::DT_INT32 ||
                   t.missSlots.desc->GetDataType() != ge::DT_INT32 ||
                   t.missCounts.desc->GetDataType() != ge::DT_INT32 ||
                   t.cacheSlotsOut.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(info.opName, "all metadata tensors must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIUMtpTiling::CheckShapes(LIUMtpTilingInfo &info) const
{
    const auto &t = info.tensors;
    const auto &q = t.query.shape->GetStorageShape();
    const auto &k = t.key.shape->GetStorageShape();
    const auto &w = t.weights.shape->GetStorageShape();
    const auto &req = t.reqPoolEntries.shape->GetStorageShape();
    const auto &cache = t.cacheSlots.shape->GetStorageShape();
    const auto &cacheTokens = t.cacheTokens.shape->GetStorageShape();
    const auto &lens = t.candidateLens.shape->GetStorageShape();
    const auto &blocks = t.blockTable.shape->GetStorageShape();
    const auto &topkSlots = t.topkSlots.shape->GetStorageShape();
    const auto &topkSource = t.topkSource.shape->GetStorageShape();
    const auto &missSource = t.missSource.shape->GetStorageShape();
    const auto &missSlots = t.missSlots.shape->GetStorageShape();
    const auto &missCounts = t.missCounts.shape->GetStorageShape();
    const auto &cacheOut = t.cacheSlotsOut.shape->GetStorageShape();

    OPS_ERR_IF(q.GetDimNum() != 3 || k.GetDimNum() != 4 ||
                   w.GetDimNum() != 2 || req.GetDimNum() != 1 ||
                   cache.GetDimNum() != 2 || cacheTokens.GetDimNum() != 1 ||
                   lens.GetDimNum() != 1 || blocks.GetDimNum() != 2,
               OPS_LOG_E(info.opName, "invalid MTP LIM input ranks."),
               return ge::GRAPH_FAILED);
    info.tokenRows = static_cast<uint32_t>(q.GetDim(0));
    info.batchSize = static_cast<uint32_t>(req.GetDim(0));
    info.poolSize = static_cast<uint32_t>(cache.GetDim(0));
    info.sourceCapacity = static_cast<uint32_t>(cache.GetDim(1));
    info.blockSize = static_cast<uint32_t>(k.GetDim(1));
    info.maxBlocks = static_cast<uint32_t>(blocks.GetDim(1));

    OPS_ERR_IF(info.batchSize == 0 || info.poolSize == 0 ||
                   info.maxBlocks == 0 || info.tokenRows != info.batchSize * MTP_QUERY_COUNT,
               OPS_LOG_E(info.opName, "require T=4*B and non-empty pool/table."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(q.GetDim(1) != MTP_HEADS || q.GetDim(2) != MTP_HEAD_DIM ||
                   w.GetDim(0) != info.tokenRows || w.GetDim(1) != MTP_HEADS,
               OPS_LOG_E(info.opName, "query must be [4B,32,128] and weights [4B,32]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(k.GetDim(0) == 0 || k.GetDim(1) != MTP_BLOCK_SIZE ||
                   k.GetDim(2) != MTP_KEY_HEADS || k.GetDim(3) != MTP_HEAD_DIM,
               OPS_LOG_E(info.opName, "key must be [blocks,128,1,128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheTokens.GetDim(0) != info.batchSize ||
                   lens.GetDim(0) != info.batchSize ||
                   blocks.GetDim(0) != info.batchSize ||
                   info.maxBlocks > (1U << 11) ||
                   info.sourceCapacity != info.maxBlocks * MTP_BLOCK_SIZE ||
                   info.sourceCapacity > (1U << 18),
               OPS_LOG_E(info.opName, "invalid request metadata or source capacity."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(topkSlots.GetDimNum() != 3 ||
                   topkSlots.GetDim(0) != info.tokenRows ||
                   topkSlots.GetDim(1) != 1 || topkSlots.GetDim(2) != MTP_TOPK,
               OPS_LOG_E(info.opName, "topk_slots must be [4B,1,2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(topkSource.GetDimNum() != 3 ||
                   topkSource.GetDim(0) != info.tokenRows ||
                   topkSource.GetDim(1) != 1 ||
                   topkSource.GetDim(2) != MTP_TOPK,
               OPS_LOG_E(info.opName,
                         "topk_source_ids must be [4B,1,2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missSource.GetDimNum() != 2 || missSlots.GetDimNum() != 2 ||
                   missSource.GetDim(0) != info.batchSize ||
                   missSlots.GetDim(0) != info.batchSize ||
                   missSource.GetDim(1) != MTP_UNION_CAPACITY ||
                   missSlots.GetDim(1) != MTP_UNION_CAPACITY ||
                   missCounts.GetDimNum() != 1 ||
                   missCounts.GetDim(0) != info.batchSize,
               OPS_LOG_E(info.opName, "union miss outputs must be [B,8192]/[B]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheOut.GetDimNum() != 2 ||
                   cacheOut.GetDim(0) != cache.GetDim(0) ||
                   cacheOut.GetDim(1) != cache.GetDim(1),
               OPS_LOG_E(info.opName, "cache_slots_out must alias the pool shape."),
               return ge::GRAPH_FAILED);
    info.queryType = t.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIUMtpTiling::ParseAndCheck(LIUMtpTilingInfo &info)
{
    if (GetPlatform(info) != ge::GRAPH_SUCCESS ||
        GetTensors(info) != ge::GRAPH_SUCCESS ||
        CheckDtypes(info) != ge::GRAPH_SUCCESS ||
        CheckShapes(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIUMtpTiling::DoTiling(LIUMtpTilingInfo *info)
{
    auto platform = platform_ascendc::PlatformAscendC(info->platformInfo);
    uint32_t aivNum = platform.GetCoreNumAiv();
    uint32_t aicNum = platform.GetCoreNumAic();
    uint32_t requestCores = std::min(info->batchSize, aicNum);
    uint32_t requestedAiv = std::min(aivNum, requestCores * 2U);
    uint32_t blockDim = platform.CalcTschBlockDim(requestedAiv, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    info->usedCoreNum = blockDim;

    constexpr uint64_t MTP_MM_RESULT_SLOTS = 4;
    constexpr uint64_t SCORE_CHUNK = 512;
    uint64_t workspaceSize = platform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<uint64_t>(blockDim) * MTP_MM_RESULT_SLOTS *
                     MTP_HEADS * SCORE_CHUNK * sizeof(float);
    uint64_t scoreStride =
        (static_cast<uint64_t>(info->sourceCapacity) + SCORE_CHUNK - 1U) /
        SCORE_CHUNK * SCORE_CHUNK;
    workspaceSize += static_cast<uint64_t>(info->batchSize) * scoreStride *
                     sizeof(float);
    workspaceSize += static_cast<uint64_t>(info->batchSize) * MTP_QUERY_COUNT *
                     MTP_TOPK * sizeof(int32_t);
    workspaceSize += static_cast<uint64_t>(info->batchSize) * MTP_QUERY_COUNT *
                     sizeof(float);
    // Four persistent (score,payload) TopK rows allow chunk-first execution,
    // which lets the Cube reuse each K chunk across all four MTP queries.
    workspaceSize += static_cast<uint64_t>(info->batchSize) * MTP_QUERY_COUNT *
                     MTP_TOPK * 2U * sizeof(float);
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(info->batchSize);
    tilingData_.set_s2Size(info->sourceCapacity);
    tilingData_.set_usedCoreNum(info->usedCoreNum);
    tilingData_.set_blockSize(info->blockSize);
    tilingData_.set_maxBlockNumPerBatch(info->maxBlocks);
    tilingData_.set_poolSize(info->poolSize);
    tilingData_.set_n1Size(MTP_HEADS);
    tilingData_.set_cacheSlotsSize(info->sourceCapacity);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    uint32_t key = GET_TPL_TILING_KEY(static_cast<uint32_t>(info->queryType));
    context_->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareMtpTiling(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForNanovllmFusedLiManageMtp(
    gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("NanovllmFusedLiManageMtp",
                                           "TilingContext is null."),
               return ge::GRAPH_FAILED);
    LIUMtpTilingInfo info;
    LIUMtpTiling tiling(context);
    if (tiling.ParseAndCheck(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tiling.DoTiling(&info);
}

IMPL_OP_OPTILING(NanovllmFusedLiManageMtp)
    .Tiling(TilingForNanovllmFusedLiManageMtp)
    .TilingParse<LIUMtpCompileInfo>(PrepareMtpTiling);

} // namespace optiling
