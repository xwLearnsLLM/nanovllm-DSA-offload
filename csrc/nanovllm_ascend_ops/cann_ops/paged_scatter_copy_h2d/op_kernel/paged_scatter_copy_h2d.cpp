#include "kernel_operator.h"

class PagedScatterCopyH2dKernel {
public:
    __aicore__ inline PagedScatterCopyH2dKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ uint8_t *kropeSrc,
                                __gm__ uint8_t *knopeSrc,
                                __gm__ uint8_t *npuBlockTable,
                                __gm__ uint8_t *cpuBlockTable,
                                __gm__ uint8_t *npuDstTokenIndex,
                                __gm__ uint8_t *cpuSrcTokenIndex,
                                __gm__ uint8_t *copyCounts,
                                __gm__ uint8_t *kropeDst,
                                __gm__ uint8_t *knopeDst,
                                __gm__ uint8_t *workspace,
                                __gm__ uint8_t *tiling)
    {
        (void)workspace;
        GET_TILING_DATA(tilingData, tiling);
        tilingData_ = tilingData;
        kropeSrc_ = kropeSrc;
        knopeSrc_ = knopeSrc;
        kropeDst_ = kropeDst;
        knopeDst_ = knopeDst;

        const uint32_t batchSize = tilingData_.batchSize;
        const uint32_t tokenCount = tilingData_.tokenCountPerBatch;
        npuBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(npuBlockTable),
                                       batchSize * tilingData_.npuBlockTableWidth);
        cpuBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cpuBlockTable),
                                       batchSize * tilingData_.cpuBlockTableWidth);
        npuDstTokenIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(npuDstTokenIndex),
                                          batchSize * tokenCount);
        cpuSrcTokenIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cpuSrcTokenIndex),
                                          batchSize * tokenCount);
        copyCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(copyCounts), batchSize);

        usedCoreNum_ = tilingData_.usedCoreNum == 0 ? AscendC::GetBlockNum() : tilingData_.usedCoreNum;
        if (usedCoreNum_ == 0) {
            usedCoreNum_ = 1;
        }
        const uint32_t maxUnitBytes =
            tilingData_.kropeUnitBytes > tilingData_.knopeUnitBytes ? tilingData_.kropeUnitBytes : tilingData_.knopeUnitBytes;
        pipe_->InitBuffer(copyBuf_, maxUnitBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t transferCount = tilingData_.batchSize * tilingData_.tokenCountPerBatch;
        if (coreIdx >= usedCoreNum_ || transferCount == 0 || tilingData_.blockSize == 0) {
            return;
        }

        AscendC::LocalTensor<uint8_t> local = copyBuf_.Get<uint8_t>();
        for (uint32_t tokenTaskIdx = coreIdx; tokenTaskIdx < transferCount; tokenTaskIdx += usedCoreNum_) {
            const uint32_t batchIdx = tokenTaskIdx / tilingData_.tokenCountPerBatch;
            const uint32_t tokenIdxInBatch = tokenTaskIdx % tilingData_.tokenCountPerBatch;
            const int32_t copyCount = copyCounts_.GetValue(batchIdx);
            if (copyCount <= 0 || tokenIdxInBatch >= static_cast<uint32_t>(copyCount)) {
                continue;
            }

            const int32_t dstToken = npuDstTokenIndex_.GetValue(tokenTaskIdx);
            const int32_t srcToken = cpuSrcTokenIndex_.GetValue(tokenTaskIdx);
            if (dstToken < 0 || srcToken < 0) {
                continue;
            }

            const uint32_t dstLogicalBlock = static_cast<uint32_t>(dstToken / tilingData_.blockSize);
            const uint32_t srcLogicalBlock = static_cast<uint32_t>(srcToken / tilingData_.blockSize);
            if (dstLogicalBlock >= tilingData_.npuBlockTableWidth ||
                srcLogicalBlock >= tilingData_.cpuBlockTableWidth) {
                continue;
            }

            const int32_t dstBlock = npuBlockTable_.GetValue(
                batchIdx * tilingData_.npuBlockTableWidth + dstLogicalBlock);
            const int32_t srcBlock = cpuBlockTable_.GetValue(
                batchIdx * tilingData_.cpuBlockTableWidth + srcLogicalBlock);
            if (dstBlock <= 0 || srcBlock <= 0) {
                continue;
            }

            const uint32_t dstOffset = static_cast<uint32_t>(dstToken % tilingData_.blockSize);
            const uint32_t srcOffset = static_cast<uint32_t>(srcToken % tilingData_.blockSize);
            const uint64_t dstSlot = static_cast<uint64_t>(dstBlock) * tilingData_.blockSize + dstOffset;
            const uint64_t srcSlot = static_cast<uint64_t>(srcBlock) * tilingData_.blockSize + srcOffset;

            CopyOne(local, kropeSrc_, kropeDst_, srcSlot, dstSlot, tilingData_.kropeUnitBytes);
            CopyOne(local, knopeSrc_, knopeDst_, srcSlot, dstSlot, tilingData_.knopeUnitBytes);
        }
    }

private:
    __aicore__ inline void CopyOne(AscendC::LocalTensor<uint8_t> local,
                                   __gm__ uint8_t *srcBase,
                                   __gm__ uint8_t *dstBase,
                                   uint64_t srcSlot,
                                   uint64_t dstSlot,
                                   uint32_t unitBytes)
    {
        AscendC::GlobalTensor<uint8_t> srcTensor;
        AscendC::GlobalTensor<uint8_t> dstTensor;
        srcTensor.SetGlobalBuffer(srcBase + srcSlot * unitBytes);
        dstTensor.SetGlobalBuffer(dstBase + dstSlot * unitBytes);

        AscendC::DataCopy(local, srcTensor, unitBytes);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(dstTensor, local, unitBytes);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECIN> copyBuf_;
    AscendC::GlobalTensor<int32_t> npuBlockTable_;
    AscendC::GlobalTensor<int32_t> cpuBlockTable_;
    AscendC::GlobalTensor<int32_t> npuDstTokenIndex_;
    AscendC::GlobalTensor<int32_t> cpuSrcTokenIndex_;
    AscendC::GlobalTensor<int32_t> copyCounts_;
    PagedScatterCopyH2dTilingData tilingData_;
    uint32_t usedCoreNum_ = 1;
    __gm__ uint8_t *kropeSrc_ = nullptr;
    __gm__ uint8_t *knopeSrc_ = nullptr;
    __gm__ uint8_t *kropeDst_ = nullptr;
    __gm__ uint8_t *knopeDst_ = nullptr;
};

extern "C" __global__ __aicore__ void paged_scatter_copy_h2d(
    __gm__ uint8_t *kropeSrc,
    __gm__ uint8_t *knopeSrc,
    __gm__ uint8_t *npuBlockTable,
    __gm__ uint8_t *cpuBlockTable,
    __gm__ uint8_t *npuDstTokenIndex,
    __gm__ uint8_t *cpuSrcTokenIndex,
    __gm__ uint8_t *copyCounts,
    __gm__ uint8_t *kropeDst,
    __gm__ uint8_t *knopeDst,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AscendC::AIC) {
        return;
    }

    AscendC::TPipe pipe;
    PagedScatterCopyH2dKernel op(&pipe);
    op.Init(kropeSrc, knopeSrc, npuBlockTable, cpuBlockTable, npuDstTokenIndex, cpuSrcTokenIndex,
            copyCounts, kropeDst, knopeDst, workspace, tiling);
    op.Process();
}
