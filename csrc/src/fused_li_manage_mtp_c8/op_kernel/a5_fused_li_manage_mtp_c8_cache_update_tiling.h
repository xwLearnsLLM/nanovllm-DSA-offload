#ifndef A5_FUSED_LI_MANAGE_MTP_C8_CACHE_UPDATE_TILING_H
#define A5_FUSED_LI_MANAGE_MTP_C8_CACHE_UPDATE_TILING_H

#include <cstdint>

struct A5FusedLiManageMtpC8CacheUpdateTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t packedQueryCount;
    uint32_t poolSize;
    uint32_t sourceCapacity;
};

#endif
