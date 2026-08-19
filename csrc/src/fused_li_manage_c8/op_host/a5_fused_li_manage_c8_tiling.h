/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef a5_fused_li_manage_c8_TILING_H_
#define a5_fused_li_manage_c8_TILING_H_

#include "a5_sfa_shared/ops_log_compat.h"
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
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 6;
constexpr uint32_t CACHE_SLOTS_INDEX = 7;
constexpr uint32_t CACHE_TOKENS_INDEX = 8;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 9;
constexpr uint32_t BLOCK_TABLE_INDEX = 10;
constexpr uint32_t SOURCE_IDS_INDEX = 0;
constexpr uint32_t DESTINATION_SLOTS_INDEX = 1;
constexpr uint32_t MISS_COUNTS_INDEX = 2;
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
constexpr uint32_t DECODE_BLOCK_SIZE = 128;
constexpr uint32_t MAX_CACHE_SLOTS_SIZE = 1U << 18;
constexpr uint32_t KEY_STRIDE0 = DECODE_BLOCK_SIZE * DECODE_HEAD_DIM;
constexpr uint32_t KEY_DEQUANT_SCALE_STRIDE0 = DECODE_BLOCK_SIZE;

// 与 BF16 版 A5FusedLiManage 完全同构：官方 QLITilingData 字段
// （顺序与类型是 kernel 侧内存布局契约）+ fused 缓存管理的
// poolSize / cacheSlotsSize。
BEGIN_TILING_DATA_DEF(LIC8TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, tSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(A5FusedLiManageC8, LIC8TilingData)

struct LIC8CompileInfo {};

struct LIC8ParaInfo {
    URequiredParaInfo query = {nullptr, nullptr};
    URequiredParaInfo key = {nullptr, nullptr};
    URequiredParaInfo weights = {nullptr, nullptr};
    URequiredParaInfo queryDequantScale = {nullptr, nullptr};
    URequiredParaInfo keyDequantScale = {nullptr, nullptr};
    UTensorParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    UTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    URequiredParaInfo cacheSlots = {nullptr, nullptr};
    UTensorParaInfo cacheTokens = {nullptr, nullptr};
    UTensorParaInfo actualSeqLengthsK = {nullptr, nullptr};
    UTensorParaInfo blockTable = {nullptr, nullptr};
    URequiredParaInfo sourceIdsOut = {nullptr, nullptr};
    URequiredParaInfo destinationSlotsOut = {nullptr, nullptr};
    URequiredParaInfo missCountsOut = {nullptr, nullptr};
    URequiredParaInfo cacheSlotsOut = {nullptr, nullptr};
};

class LIC8TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LIC8ParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t tSize = 0;
    uint32_t n1Size = 0;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t cacheSlotsSize = 0;
};

class A5FusedLiManageC8Tiling {
public:
    explicit A5FusedLiManageC8Tiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(LIC8TilingInfo &tilingInfo);
    ge::graphStatus DoTiling(LIC8TilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(LIC8TilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(LIC8TilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const LIC8TilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(LIC8TilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    LIC8TilingData tilingData_;
};

} // namespace optiling
#endif // a5_fused_li_manage_c8_TILING_H_
