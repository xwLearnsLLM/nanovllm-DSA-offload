/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef KVCACHE_OFFLOAD_COPY_TILING_H_
#define KVCACHE_OFFLOAD_COPY_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvcacheOffloadCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, totalPairSlots);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, copyCap);
TILING_DATA_FIELD_DEF(int64_t, blockBytes);
TILING_DATA_FIELD_DEF(int64_t, hbmBlockCount);
TILING_DATA_FIELD_DEF(int64_t, dramBlockCount);
TILING_DATA_FIELD_DEF(int64_t, tileBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NanovllmKvcacheOffloadCopy, KvcacheOffloadCopyTilingData)

struct KvcacheOffloadCopyCompileInfo {
};

class KvcacheOffloadCopyTiling {
public:
    explicit KvcacheOffloadCopyTiling(gert::TilingContext* context) : context_(context) {}
    ~KvcacheOffloadCopyTiling() {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus GetDtypeInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    KvcacheOffloadCopyTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t batchSize_ = 0;
    int64_t copyCap_ = 0;
    int64_t blockBytes_ = 0;
};

} // namespace optiling

#endif // KVCACHE_OFFLOAD_COPY_TILING_H_
