/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef a5_fused_li_manage_mtp_TILING_H_
#define a5_fused_li_manage_mtp_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct URequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct UTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t CACHE_SLOTS_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr uint32_t TOPK_INDEX = 0;
constexpr uint32_t TOPK_SLOTS_INDEX = 1;
constexpr uint32_t MISS_INDEX = 2;
constexpr uint32_t MISS_SLOTS_INDEX = 3;
constexpr uint32_t MISS_COUNT_INDEX = 4;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = 1;
constexpr uint32_t DECODE_G_SIZE_32 = 32;
constexpr uint32_t DECODE_G_SIZE_64 = 64;
constexpr uint32_t DECODE_HEAD_DIM = 128;
constexpr uint32_t DECODE_SPARSE_COUNT = 2048;
constexpr uint32_t CACHE_SLOTS_SIZE = 262144;
constexpr uint32_t MTP_MAX_QUERY_COUNT = 4;
constexpr uint32_t MTP_CACHE_SIZE = 8192;

BEGIN_TILING_DATA_DEF(LIMtpA5TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(A5FusedLiManageMtp, LIMtpA5TilingData)

struct LIMtpA5CompileInfo {};

struct LIMtpA5ParaInfo {
    URequiredParaInfo query = {nullptr, nullptr};
    URequiredParaInfo key = {nullptr, nullptr};
    URequiredParaInfo weights = {nullptr, nullptr};
    URequiredParaInfo cacheSlots = {nullptr, nullptr};
    UTensorParaInfo actualSeqLengthsQuery = {nullptr, nullptr};
    UTensorParaInfo actualSeqLengthsKey = {nullptr, nullptr};
    UTensorParaInfo blockTable = {nullptr, nullptr};
    URequiredParaInfo topkIndexOut = {nullptr, nullptr};
    URequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    URequiredParaInfo missIndexOut = {nullptr, nullptr};
    URequiredParaInfo missSlotsOut = {nullptr, nullptr};
    URequiredParaInfo missCountOut = {nullptr, nullptr};
};

class LIMtpA5TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LIMtpA5ParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t tSize = 0;
    uint32_t n1Size = DECODE_G_SIZE_64;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class A5FusedLiManageMtpTiling {
public:
    explicit A5FusedLiManageMtpTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(LIMtpA5TilingInfo &tilingInfo);
    ge::graphStatus DoTiling(LIMtpA5TilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(LIMtpA5TilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(LIMtpA5TilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const LIMtpA5TilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(LIMtpA5TilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    LIMtpA5TilingData tilingData_;
};

} // namespace optiling
#endif // a5_fused_li_manage_mtp_TILING_H_
