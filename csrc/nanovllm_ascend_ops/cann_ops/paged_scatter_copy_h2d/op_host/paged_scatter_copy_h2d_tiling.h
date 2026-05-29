#ifndef PAGED_SCATTER_COPY_H2D_TILING_H_
#define PAGED_SCATTER_COPY_H2D_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t KROPE_SRC_INDEX = 0;
constexpr uint32_t KNOPE_SRC_INDEX = 1;
constexpr uint32_t NPU_BLOCK_TABLE_INDEX = 2;
constexpr uint32_t CPU_BLOCK_TABLE_INDEX = 3;
constexpr uint32_t NPU_DST_TOKEN_INDEX = 4;
constexpr uint32_t CPU_SRC_TOKEN_INDEX = 5;
constexpr uint32_t COPY_COUNTS_INDEX = 6;

constexpr uint32_t ATTR_KROPE_UNIT_BYTES = 0;
constexpr uint32_t ATTR_KNOPE_UNIT_BYTES = 1;
constexpr uint32_t ATTR_BLOCK_SIZE = 2;

BEGIN_TILING_DATA_DEF(PagedScatterCopyH2dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, tokenCountPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, npuBlockTableWidth)
TILING_DATA_FIELD_DEF(uint32_t, cpuBlockTableWidth)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, kropeUnitBytes)
TILING_DATA_FIELD_DEF(uint32_t, knopeUnitBytes)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(PagedScatterCopyH2d, PagedScatterCopyH2dTilingData)

} // namespace optiling

#endif // PAGED_SCATTER_COPY_H2D_TILING_H_
