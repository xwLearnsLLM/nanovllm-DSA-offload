#ifndef DSA_INDEX_UPDATE_TILING_DATA_H_
#define DSA_INDEX_UPDATE_TILING_DATA_H_

#include <cstdint>

constexpr int32_t DSA_INDEX_UPDATE_MAX_K = 128;

struct DsaIndexUpdateTilingData {
    int64_t batchSize = 0;
    int64_t maxSeqLen = 0;
    int64_t maxSelectedLen = 0;
    int64_t poolCapacity = 0;
    int64_t maxOutputLen = 0;
    int64_t k = 0;
    int64_t usedCoreNum = 1;
    int64_t coreNumPerBatch = 1;
};

#endif // DSA_INDEX_UPDATE_TILING_DATA_H_
