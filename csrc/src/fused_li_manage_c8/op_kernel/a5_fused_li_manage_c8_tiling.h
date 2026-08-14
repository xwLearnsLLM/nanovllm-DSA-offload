#ifndef A5_FUSED_LI_MANAGE_C8_TILING_H
#define A5_FUSED_LI_MANAGE_C8_TILING_H

#include <cstdint>

// Static geometry only. Request-dependent lengths, budgets and pool entries
// remain device tensors so full-decode-only graph replay can refresh them.
struct A5FusedLiManageC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
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
