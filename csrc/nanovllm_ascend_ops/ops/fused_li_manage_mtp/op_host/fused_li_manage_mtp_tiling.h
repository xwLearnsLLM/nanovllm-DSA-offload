/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_TILING_H_
#define NANOVLLM_FUSED_LI_MANAGE_MTP_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct MtpRequiredTensor {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *shape = nullptr;
};

constexpr uint32_t MTP_QUERY_INDEX = 0;
constexpr uint32_t MTP_KEY_INDEX = 1;
constexpr uint32_t MTP_WEIGHTS_INDEX = 2;
constexpr uint32_t MTP_REQ_POOL_INDEX = 3;
constexpr uint32_t MTP_CACHE_SLOTS_INDEX = 4;
constexpr uint32_t MTP_CACHE_TOKENS_INDEX = 5;
constexpr uint32_t MTP_CANDIDATE_LENS_INDEX = 6;
constexpr uint32_t MTP_BLOCK_TABLE_INDEX = 7;

constexpr uint32_t MTP_TOPK_SLOTS_OUT = 0;
constexpr uint32_t MTP_TOPK_SOURCE_OUT = 1;
constexpr uint32_t MTP_MISS_SOURCE_OUT = 2;
constexpr uint32_t MTP_MISS_SLOTS_OUT = 3;
constexpr uint32_t MTP_MISS_COUNTS_OUT = 4;
constexpr uint32_t MTP_CACHE_SLOTS_OUT = 5;

constexpr uint32_t MTP_QUERY_COUNT = 4;
constexpr uint32_t MTP_HEADS_MIN = 32;
constexpr uint32_t MTP_HEADS_MAX = 64;
constexpr uint32_t MTP_KEY_HEADS = 1;
constexpr uint32_t MTP_HEAD_DIM = 128;
constexpr uint32_t MTP_BLOCK_SIZE = 128;
constexpr uint32_t MTP_TOPK = 2048;
constexpr uint32_t MTP_UNION_CAPACITY = 8192;

BEGIN_TILING_DATA_DEF(LIUMtpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NanovllmFusedLiManageMtp, LIUMtpTilingData)

struct LIUMtpCompileInfo {};

struct LIUMtpTensors {
    MtpRequiredTensor query;
    MtpRequiredTensor key;
    MtpRequiredTensor weights;
    MtpRequiredTensor reqPoolEntries;
    MtpRequiredTensor cacheSlots;
    MtpRequiredTensor cacheTokens;
    MtpRequiredTensor candidateLens;
    MtpRequiredTensor blockTable;
    MtpRequiredTensor topkSlots;
    MtpRequiredTensor topkSource;
    MtpRequiredTensor missSource;
    MtpRequiredTensor missSlots;
    MtpRequiredTensor missCounts;
    MtpRequiredTensor cacheSlotsOut;
};

class LIUMtpTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion =
        platform_ascendc::SocVersion::ASCEND910B;
    LIUMtpTensors tensors;
    uint32_t batchSize = 0;
    uint32_t tokenRows = 0;
    uint32_t queryHeads = 0;
    uint32_t sourceCapacity = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlocks = 0;
    uint32_t poolSize = 0;
    uint32_t usedCoreNum = 0;
    ge::DataType queryType = ge::DT_FLOAT16;
};

class LIUMtpTiling {
public:
    explicit LIUMtpTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus ParseAndCheck(LIUMtpTilingInfo &info);
    ge::graphStatus DoTiling(LIUMtpTilingInfo *info);

private:
    ge::graphStatus GetPlatform(LIUMtpTilingInfo &info) const;
    ge::graphStatus GetTensors(LIUMtpTilingInfo &info) const;
    ge::graphStatus CheckDtypes(const LIUMtpTilingInfo &info) const;
    ge::graphStatus CheckShapes(LIUMtpTilingInfo &info) const;

    gert::TilingContext *context_ = nullptr;
    LIUMtpTilingData tilingData_;
};

} // namespace optiling
#endif
