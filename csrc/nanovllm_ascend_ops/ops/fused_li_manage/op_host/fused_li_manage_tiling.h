/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_LI_MANAGE_TILING_H_
#define FUSED_LI_MANAGE_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct HMRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct HMTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 3;
constexpr uint32_t CACHE_SLOTS_INDEX = 4;
constexpr uint32_t CACHE_TOKENS_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6;
constexpr uint32_t BLOCK_TABLE_INDEX = 7;
constexpr uint32_t TOPK_INDEX = 0;
constexpr uint32_t TOPK_SLOTS_INDEX = 1;
constexpr uint32_t MISS_COUNT_INDEX = 2;
constexpr uint32_t CACHE_SLOTS_OUT_INDEX = 3;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = 1;
constexpr uint32_t DECODE_HEAD_DIM = 128;
constexpr uint32_t DECODE_SPARSE_COUNT = 2048;
constexpr uint32_t DECODE_OUTPUT_CAPACITY = 2048;

BEGIN_TILING_DATA_DEF(FusedLiManageTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
TILING_DATA_FIELD_DEF(uint32_t, scheduleMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NanovllmFusedLiManage, FusedLiManageTilingData)

struct FusedLiManageCompileInfo {};

struct FusedLiManageParaInfo {
    HMRequiredParaInfo query = {nullptr, nullptr};
    HMRequiredParaInfo key = {nullptr, nullptr};
    HMRequiredParaInfo weights = {nullptr, nullptr};
    HMTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    HMRequiredParaInfo cacheSlots = {nullptr, nullptr};
    HMTensorParaInfo cacheTokens = {nullptr, nullptr};
    HMTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    HMTensorParaInfo blockTable = {nullptr, nullptr};
    HMRequiredParaInfo topkIndexOut = {nullptr, nullptr};
    HMRequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    HMRequiredParaInfo missCountOut = {nullptr, nullptr};
    HMRequiredParaInfo cacheSlotsOut = {nullptr, nullptr};
};

class FusedLiManageTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    FusedLiManageParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = 32;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t cacheSlotsSize = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class FusedLiManageTiling {
public:
    explicit FusedLiManageTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(FusedLiManageTilingInfo &tilingInfo);
    ge::graphStatus DoTiling(FusedLiManageTilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(FusedLiManageTilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(FusedLiManageTilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const FusedLiManageTilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(FusedLiManageTilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    FusedLiManageTilingData tilingData_;
};

} // namespace optiling
#endif // FUSED_LI_MANAGE_TILING_H_

