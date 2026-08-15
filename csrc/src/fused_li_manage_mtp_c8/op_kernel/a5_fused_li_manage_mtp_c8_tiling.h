#ifndef A5_FUSED_LI_MANAGE_MTP_C8_TILING_H
#define A5_FUSED_LI_MANAGE_MTP_C8_TILING_H

#include <cstdint>

// Only static geometry is tiled. Query counts, request-pool entries, cache
// budgets and candidate lengths remain device tensors for graph replay.
struct A5FusedLiManageMtpC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t packedQueryCount;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t indexHeads;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxCandidateLen;
    uint32_t keyStride;
    uint32_t scaleStride;
    uint32_t scoreWorkspaceStride;
};

#endif
