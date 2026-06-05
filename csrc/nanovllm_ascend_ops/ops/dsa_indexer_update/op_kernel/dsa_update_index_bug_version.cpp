#include "kernel_operator.h"                 // AscendC kernel, tensor, pipe, and vector-sort APIs.
#include "kernel_tiling/kernel_tiling.h"     // AscendC tiling-data access macros.
#include "dsa_update_index_tiling_data.h"    // Kernel-side tiling struct and max-k constant.

namespace {                                  // Keep helper symbols private to this translation unit.

constexpr uint16_t DSA_BF16_LOW_SENTINEL_RAW = 0xFF80;     // Raw BF16 -inf, used by scalar fallback path.
constexpr int32_t DSA_STAGE1_SORT_BLOCK = 1024;            // Number of candidates sorted by one Sort call.
constexpr int32_t DSA_STAGE1_SORT_REPEAT = DSA_STAGE1_SORT_BLOCK / 32; // Sort repeat count, 32 elems/repeat.
constexpr int32_t DSA_ACCUM_SORT_BLOCKS = 4;               // Merge up to four sorted 1024-blocks before topK merge.
constexpr int32_t DSA_VALUE_AND_INDEX_NUM = 2;             // Interleaved pair layout: [value, index].
constexpr int32_t DSA_PAIR_FLOATS = DSA_UPDATE_INDEX_MAX_K * DSA_VALUE_AND_INDEX_NUM; // One topK pair list.
constexpr int32_t DSA_ACCUM_SORT_ELEMS = DSA_STAGE1_SORT_BLOCK * DSA_ACCUM_SORT_BLOCKS; // 4096 candidates.
constexpr int32_t DSA_MERGE_PAIR_FLOATS =
    (DSA_ACCUM_SORT_ELEMS + DSA_UPDATE_INDEX_MAX_K) * DSA_VALUE_AND_INDEX_NUM; // 4096 + 128 pairs.
constexpr int32_t DSA_WORKSPACE_LIST_NUM = 2;                // Per core partial lists: bottom and top.
constexpr int32_t DSA_WORKSPACE_BOTTOM_LIST = 0;             // Workspace list id for demote candidates.
constexpr int32_t DSA_WORKSPACE_TOP_LIST = 1;                // Workspace list id for promote candidates.
constexpr int32_t DSA_MRG_QUE_0 = 0;                         // MrgSort source queue 0.
constexpr int32_t DSA_MRG_QUE_1 = 1;                         // MrgSort source queue 1.
constexpr int32_t DSA_MRG_QUE_2 = 2;                         // MrgSort source queue 2.
constexpr int32_t DSA_MRG_QUE_3 = 3;                         // MrgSort source queue 3.
constexpr float DSA_SORT_NEG_INF = -3.4028234663852886e38F;  // Float invalid candidate sentinel.
constexpr uint32_t DSA_BF16_SORT_KEY_MASK = 0xFFFFU;         // BF16 sortable key fits in 16 bits.
constexpr uint32_t DSA_FLOAT_ONE_RAW = 0x3F800000U;          // Raw float bits for 1.0f.
constexpr uint32_t DSA_SORT_KEY_TO_MANTISSA_SHIFT = 7U;      // Put 16-bit key into float mantissa.
constexpr uint32_t DSA_SORT_KEY_MIN = 0U;                    // Scalar topK initialization key.
constexpr uint32_t DSA_SORT_KEY_MAX = 0xFFFFFFFFU;           // Scalar bottomK initialization key.

class KernelDsaUpdateIndex {                                // One kernel object per launched AIV block.
public:
    __aicore__ inline KernelDsaUpdateIndex() {}

    __aicore__ inline void Init(GM_ADDR score, GM_ADDR selectedIdx, GM_ADDR promoteIdx,
        GM_ADDR demoteIdx, GM_ADDR seqLen, GM_ADDR selectedLen, GM_ADDR partialWorkspace,
        const DsaUpdateIndexTilingData* tiling, AscendC::TPipe* pipe)
    {
        scoreGm_.SetGlobalBuffer((__gm__ uint16_t*)score);
        selectedIdxGm_.SetGlobalBuffer((__gm__ int32_t*)selectedIdx);
        promoteIdxGm_.SetGlobalBuffer((__gm__ int32_t*)promoteIdx);
        demoteIdxGm_.SetGlobalBuffer((__gm__ int32_t*)demoteIdx);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t*)seqLen);
        selectedLenGm_.SetGlobalBuffer((__gm__ int32_t*)selectedLen);
        partialGm_.SetGlobalBuffer((__gm__ float*)partialWorkspace);
        partialIdxGm_.SetGlobalBuffer((__gm__ int32_t*)partialWorkspace);

        batchSize_ = tiling->batchSize;
        maxSeqLen_ = tiling->maxSeqLen;
        maxSelectedLen_ = tiling->maxSelectedLen;
        k_ = static_cast<int32_t>(tiling->k);
        usedCoreNum_ = tiling->usedCoreNum;
        coreNumPerBatch_ = tiling->coreNumPerBatch;

        pipe->InitBuffer(valueBuf_, DSA_STAGE1_SORT_BLOCK * sizeof(float));
        pipe->InitBuffer(indexBuf_, DSA_STAGE1_SORT_BLOCK * sizeof(uint32_t));
        pipe->InitBuffer(accumBuf_, DSA_MERGE_PAIR_FLOATS * sizeof(float));
        pipe->InitBuffer(tmpBuf_, DSA_MERGE_PAIR_FLOATS * sizeof(float));
        pipe->InitBuffer(topBuf_, DSA_PAIR_FLOATS * sizeof(float));
        pipe->InitBuffer(bottomBuf_, DSA_PAIR_FLOATS * sizeof(float));
        pipe->InitBuffer(scoreRawBuf_, DSA_STAGE1_SORT_BLOCK * sizeof(uint16_t));
        pipe->InitBuffer(indexTemplateBuf_, DSA_STAGE1_SORT_BLOCK * sizeof(uint32_t));

        AscendC::LocalTensor<uint32_t> tmplInit = indexTemplateBuf_.Get<uint32_t>();
        for (int32_t i = 0; i < DSA_STAGE1_SORT_BLOCK; ++i) {
            tmplInit.SetValue(i, static_cast<uint32_t>(i));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        if (k_ < DSA_UPDATE_INDEX_MAX_K || coreNumPerBatch_ <= 1) {
            for (int64_t batch = blockIdx; batch < batchSize_; batch += usedCoreNum_) {
                ProcessBatchSingleCore(batch);
            }
            return;
        }

        const int64_t batch = blockIdx / coreNumPerBatch_;
        const int32_t coreInBatch = static_cast<int32_t>(blockIdx - batch * coreNumPerBatch_);
        ProcessBatchMultiCore(batch, coreInBatch, blockIdx);
    }

private:
    /**
     * Convert raw BF16 bits into an unsigned key whose order matches numeric order.
     */
    __aicore__ inline uint32_t Bf16RawToSortKey(uint16_t raw)
    {
        if ((raw & 0x7FFFU) == 0U) {
            return 0x8000U;
        }
        if ((raw & 0x8000U) != 0U) {
            return static_cast<uint32_t>(static_cast<uint16_t>(~raw));
        }
        return static_cast<uint32_t>(raw ^ 0x8000U);
    }

    __aicore__ inline int32_t MinInt32(int32_t lhs, int32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void SplitRange(int32_t total, int32_t partIdx, int32_t partNum,
        int32_t* start, int32_t* end)
    {
        const int64_t total64 = static_cast<int64_t>(total);
        *start = static_cast<int32_t>((total64 * partIdx) / partNum);
        *end = static_cast<int32_t>((total64 * (partIdx + 1)) / partNum);
    }

    __aicore__ inline void SplitSortBlockRange(int32_t total, int32_t partIdx, int32_t partNum,
        int32_t* start, int32_t* end)
    {
        const int32_t blockNum = (total + DSA_STAGE1_SORT_BLOCK - 1) / DSA_STAGE1_SORT_BLOCK;
        const int32_t startBlock = static_cast<int32_t>((static_cast<int64_t>(blockNum) * partIdx) / partNum);
        const int32_t endBlock = static_cast<int32_t>((static_cast<int64_t>(blockNum) * (partIdx + 1)) / partNum);
        *start = startBlock * DSA_STAGE1_SORT_BLOCK;
        const int32_t rawEnd = endBlock * DSA_STAGE1_SORT_BLOCK;
        *end = rawEnd < total ? rawEnd : total;
    }

    __aicore__ inline bool IsBetterForBottom(uint32_t scoreKey, int32_t localIdx, uint32_t refKey, int32_t refIdx)
    {
        return (scoreKey < refKey) || ((scoreKey == refKey) && (localIdx < refIdx));
    }

    __aicore__ inline bool IsBetterForTop(uint32_t scoreKey, int32_t globalIdx, uint32_t refKey, int32_t refIdx)
    {
        return (scoreKey > refKey) || ((scoreKey == refKey) && (globalIdx < refIdx));
    }

    __aicore__ inline void InsertBottomScalar(uint32_t scoreKey, int32_t localIdx,
        uint32_t* bottomKeys, int32_t* bottomIdx)
    {
        int32_t insertPos = -1;
        for (int32_t pos = 0; pos < k_; ++pos) {
            if (IsBetterForBottom(scoreKey, localIdx, bottomKeys[pos], bottomIdx[pos])) {
                insertPos = pos;
                break;
            }
        }
        if (insertPos < 0) {
            return;
        }
        for (int32_t pos = k_ - 1; pos > insertPos; --pos) {
            bottomKeys[pos] = bottomKeys[pos - 1];
            bottomIdx[pos] = bottomIdx[pos - 1];
        }
        bottomKeys[insertPos] = scoreKey;
        bottomIdx[insertPos] = localIdx;
    }

    __aicore__ inline void InsertTopScalar(uint32_t scoreKey, int32_t globalIdx,
        uint32_t* topKeys, int32_t* topIdx)
    {
        int32_t insertPos = -1;
        for (int32_t pos = 0; pos < k_; ++pos) {
            if (IsBetterForTop(scoreKey, globalIdx, topKeys[pos], topIdx[pos])) {
                insertPos = pos;
                break;
            }
        }
        if (insertPos < 0) {
            return;
        }
        for (int32_t pos = k_ - 1; pos > insertPos; --pos) {
            topKeys[pos] = topKeys[pos - 1];
            topIdx[pos] = topIdx[pos - 1];
        }
        topKeys[insertPos] = scoreKey;
        topIdx[insertPos] = globalIdx;
    }

    __aicore__ inline uint32_t ScoreKeyToSortFloatBits(uint32_t scoreKey)
    {
        return DSA_FLOAT_ONE_RAW |
               ((scoreKey & DSA_BF16_SORT_KEY_MASK) << DSA_SORT_KEY_TO_MANTISSA_SHIFT);
    }

    __aicore__ inline uint32_t ReverseScoreKeyToSortFloatBits(uint32_t scoreKey)
    {
        const uint32_t reversedKey = DSA_BF16_SORT_KEY_MASK - (scoreKey & DSA_BF16_SORT_KEY_MASK);
        return DSA_FLOAT_ONE_RAW | (reversedKey << DSA_SORT_KEY_TO_MANTISSA_SHIFT);
    }

    __aicore__ inline void InitPairBuf(AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf)
    {
        AscendC::LocalTensor<float> pairLocal = pairBuf.Get<float>();
        AscendC::LocalTensor<uint32_t> pairIndexLocal = pairLocal.template ReinterpretCast<uint32_t>();
        for (int32_t i = 0; i < DSA_UPDATE_INDEX_MAX_K; ++i) {
            pairLocal.SetValue(i * DSA_VALUE_AND_INDEX_NUM, DSA_SORT_NEG_INF);
            pairIndexLocal.SetValue(i * DSA_VALUE_AND_INDEX_NUM + 1, 0U);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClearSortInput()
    {
        AscendC::LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        AscendC::LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        AscendC::Duplicate(valueLocal, DSA_SORT_NEG_INF, DSA_STAGE1_SORT_BLOCK);
        AscendC::Duplicate(indexLocal.template ReinterpretCast<int32_t>(), 0, DSA_STAGE1_SORT_BLOCK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortCurrentBlockToAccum(int32_t accumBlockIdx)
    {
        AscendC::LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        AscendC::LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        AscendC::LocalTensor<float> accumLocal = accumBuf_.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        AscendC::Sort<float, true>(
            accumLocal[accumBlockIdx * DSA_STAGE1_SORT_BLOCK * DSA_VALUE_AND_INDEX_NUM],
            valueLocal, indexLocal, tmpLocal, DSA_STAGE1_SORT_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeAccumListsToTmp(int32_t listCount, int32_t listLen)
    {
        AscendC::LocalTensor<float> accumLocal = accumBuf_.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        if (listCount == 1) {
            AscendC::DataCopy(tmpLocal, accumLocal, listLen * DSA_VALUE_AND_INDEX_NUM);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::MrgSort4Info params;
        params.elementLengths[DSA_MRG_QUE_0] = listLen;
        params.elementLengths[DSA_MRG_QUE_1] = listLen;
        params.elementLengths[DSA_MRG_QUE_2] = listLen;
        params.elementLengths[DSA_MRG_QUE_3] = listLen;
        params.ifExhaustedSuspension = false;
        params.repeatTimes = 1;
        if (listCount == 2) {
            params.validBit = 0b0011;
        } else if (listCount == 3) {
            params.validBit = 0b0111;
        } else {
            params.validBit = 0b1111;
        }

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = accumLocal[0];
        srcList.src2 = accumLocal[listLen * DSA_VALUE_AND_INDEX_NUM];
        srcList.src3 = accumLocal[listLen * DSA_VALUE_AND_INDEX_NUM * 2];
        srcList.src4 = accumLocal[listLen * DSA_VALUE_AND_INDEX_NUM * 3];
        AscendC::MrgSort<float>(tmpLocal, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeTmpListIntoPair(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf, int32_t srcLen)
    {
        AscendC::LocalTensor<float> pairLocal = pairBuf.Get<float>();
        AscendC::LocalTensor<float> srcLocal = tmpBuf_.Get<float>();
        AscendC::LocalTensor<float> outLocal = accumBuf_.Get<float>();

        AscendC::MrgSort4Info params;
        params.elementLengths[DSA_MRG_QUE_0] = DSA_UPDATE_INDEX_MAX_K;
        params.elementLengths[DSA_MRG_QUE_1] = srcLen;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b0011;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = pairLocal;
        srcList.src2 = srcLocal;
        AscendC::MrgSort<float>(outLocal, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(pairLocal, outLocal, DSA_PAIR_FLOATS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeAccumListsIntoPair(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf, int32_t listCount, int32_t listLen)
    {
        if (listCount <= 0) {
            return;
        }
        MergeAccumListsToTmp(listCount, listLen);
        MergeTmpListIntoPair(pairBuf, listCount * listLen);
    }

    __aicore__ inline bool IsBetterPair(float value, int32_t idx, float refValue, int32_t refIdx)
    {
        return (value > refValue) || ((value == refValue) && (idx < refIdx));
    }

    __aicore__ inline void InsertPairScalar(float value, int32_t idx, float* bestValues, int32_t* bestIdx)
    {
        int32_t insertPos = -1;
        for (int32_t pos = 0; pos < DSA_UPDATE_INDEX_MAX_K; ++pos) {
            if (IsBetterPair(value, idx, bestValues[pos], bestIdx[pos])) {
                insertPos = pos;
                break;
            }
        }
        if (insertPos < 0) {
            return;
        }
        for (int32_t pos = DSA_UPDATE_INDEX_MAX_K - 1; pos > insertPos; --pos) {
            bestValues[pos] = bestValues[pos - 1];
            bestIdx[pos] = bestIdx[pos - 1];
        }
        bestValues[insertPos] = value;
        bestIdx[insertPos] = idx;
    }

    __aicore__ inline int32_t GetPairIndex(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf, int32_t pos)
    {
        AscendC::LocalTensor<float> pairLocal = pairBuf.Get<float>();
        AscendC::LocalTensor<int32_t> pairIndexLocal = pairLocal.template ReinterpretCast<int32_t>();
        return pairIndexLocal.GetValue(pos * DSA_VALUE_AND_INDEX_NUM + 1);
    }

    __aicore__ inline void StorePairToWorkspace(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf, int64_t blockIdx, int32_t listId)
    {
        AscendC::LocalTensor<float> pairLocal = pairBuf.Get<float>();
        AscendC::DataCopy(partialGm_[PartialOffset(blockIdx, listId)], pairLocal, DSA_PAIR_FLOATS);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int64_t PartialOffset(int64_t blockIdx, int32_t listId)
    {
        return (blockIdx * DSA_WORKSPACE_LIST_NUM + listId) * static_cast<int64_t>(DSA_PAIR_FLOATS);
    }

    __aicore__ inline void ReduceWorkspacePartialsIntoPair(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& pairBuf, int64_t groupBlockStart, int32_t listId)
    {
        float bestValues[DSA_UPDATE_INDEX_MAX_K];
        int32_t bestIdx[DSA_UPDATE_INDEX_MAX_K];
        for (int32_t i = 0; i < DSA_UPDATE_INDEX_MAX_K; ++i) {
            bestValues[i] = DSA_SORT_NEG_INF;
            bestIdx[i] = 0;
        }

        for (int32_t core = 0; core < coreNumPerBatch_; ++core) {
            const int64_t srcOffset = PartialOffset(groupBlockStart + core, listId);
            for (int32_t pos = 0; pos < DSA_UPDATE_INDEX_MAX_K; ++pos) {
                const int64_t pairOffset = srcOffset + pos * DSA_VALUE_AND_INDEX_NUM;
                const float value = partialGm_.GetValue(pairOffset);
                const int32_t idx = partialIdxGm_.GetValue(pairOffset + 1);
                InsertPairScalar(value, idx, bestValues, bestIdx);
            }
        }

        AscendC::LocalTensor<float> pairLocal = pairBuf.Get<float>();
        AscendC::LocalTensor<int32_t> pairIndexLocal = pairLocal.template ReinterpretCast<int32_t>();
        for (int32_t i = 0; i < DSA_UPDATE_INDEX_MAX_K; ++i) {
            pairLocal.SetValue(i * DSA_VALUE_AND_INDEX_NUM, bestValues[i]);
            pairIndexLocal.SetValue(i * DSA_VALUE_AND_INDEX_NUM + 1, bestIdx[i]);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * Find bottom-K demote candidates among selected positions (by local index).
     * Does NOT write to GM — masking is done in the top path in UB.
     */
    __aicore__ inline void ProcessBottomCandidatesRange(
        int64_t scoreBase, int64_t selectedBase, int32_t localStart, int32_t localEnd)
    {
        InitPairBuf(bottomBuf_);
        AscendC::LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        AscendC::LocalTensor<uint32_t> valueBitsLocal = valueLocal.template ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();

        int32_t accumCount = 0;
        for (int32_t base = localStart; base < localEnd; base += DSA_STAGE1_SORT_BLOCK) {
            const int32_t chunk = MinInt32(DSA_STAGE1_SORT_BLOCK, localEnd - base);
            if (chunk < DSA_STAGE1_SORT_BLOCK) {
                ClearSortInput();
            }
            AscendC::LocalTensor<uint32_t> tmplLocal = indexTemplateBuf_.Get<uint32_t>();
            AscendC::Duplicate(indexLocal, static_cast<uint32_t>(base), chunk);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<int32_t> indexInt = indexLocal.template ReinterpretCast<int32_t>();
            AscendC::LocalTensor<int32_t> tmplInt = tmplLocal.template ReinterpretCast<int32_t>();
            AscendC::Add(indexInt, indexInt, tmplInt, chunk);
            AscendC::PipeBarrier<PIPE_V>();

            for (int32_t i = 0; i < chunk; ++i) {
                const int32_t localIdx = base + i;
                const int32_t globalIdx = selectedIdxGm_.GetValue(selectedBase + localIdx);
                const uint32_t scoreKey = Bf16RawToSortKey(scoreGm_.GetValue(scoreBase + globalIdx));
                valueBitsLocal.SetValue(i, ReverseScoreKeyToSortFloatBits(scoreKey));
            }
            AscendC::PipeBarrier<PIPE_V>();
            SortCurrentBlockToAccum(accumCount);
            ++accumCount;
            if (accumCount == DSA_ACCUM_SORT_BLOCKS) {
                MergeAccumListsIntoPair(bottomBuf_, accumCount, DSA_STAGE1_SORT_BLOCK);
                accumCount = 0;
            }
        }
        MergeAccumListsIntoPair(bottomBuf_, accumCount, DSA_STAGE1_SORT_BLOCK);
    }

    /**
     * Find top-K promote candidates in [globalStart, globalEnd).
     *
     * Masking is done on raw BF16 data in scoreRawBuf_ BEFORE Cast, so that
     * scalar (mask) and vector (Cast) pipelines write to different UB buffers.
     */
    __aicore__ inline void ProcessTopCandidatesRange(
        int64_t scoreBase, int64_t selectedBase, int32_t m, int32_t globalStart, int32_t globalEnd)
    {
        InitPairBuf(topBuf_);
        AscendC::LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        AscendC::LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        AscendC::LocalTensor<uint16_t> scoreRawLocal = scoreRawBuf_.Get<uint16_t>();
        AscendC::LocalTensor<bfloat16_t> scoreBf16Local = scoreRawLocal.template ReinterpretCast<bfloat16_t>();

        int32_t selectedCursor = 0;
        int32_t accumCount = 0;
        for (int32_t base = globalStart; base < globalEnd; base += DSA_STAGE1_SORT_BLOCK) {
            const int32_t chunk = MinInt32(DSA_STAGE1_SORT_BLOCK, globalEnd - base);
            if (chunk < DSA_STAGE1_SORT_BLOCK) {
                ClearSortInput();
            }
            AscendC::DataCopy(scoreRawLocal, scoreGm_[scoreBase + base], chunk);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Fast path assumes selected_idx is monotonic in token-id order. This is
            // intentionally restored for performance comparison; dynamic DSA updates
            // can make selected_idx unordered, so this path may select duplicate tokens.
            while (selectedCursor < m) {
                const int32_t globalIdx = selectedIdxGm_.GetValue(selectedBase + selectedCursor);
                if (globalIdx < base) {
                    ++selectedCursor;
                    continue;
                }
                if (globalIdx >= base + chunk) {
                    break;
                }
                scoreRawLocal.SetValue(globalIdx - base, DSA_BF16_LOW_SENTINEL_RAW);
                ++selectedCursor;
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Cast<float, bfloat16_t>(valueLocal, scoreBf16Local, AscendC::RoundMode::CAST_NONE, chunk);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<uint32_t> tmplLocal = indexTemplateBuf_.Get<uint32_t>();
            AscendC::Duplicate(indexLocal, static_cast<uint32_t>(base), chunk);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<int32_t> indexInt = indexLocal.template ReinterpretCast<int32_t>();
            AscendC::LocalTensor<int32_t> tmplInt = tmplLocal.template ReinterpretCast<int32_t>();
            AscendC::Add(indexInt, indexInt, tmplInt, chunk);
            AscendC::PipeBarrier<PIPE_V>();
            SortCurrentBlockToAccum(accumCount);
            ++accumCount;
            if (accumCount == DSA_ACCUM_SORT_BLOCKS) {
                MergeAccumListsIntoPair(topBuf_, accumCount, DSA_STAGE1_SORT_BLOCK);
                accumCount = 0;
            }
        }
        MergeAccumListsIntoPair(topBuf_, accumCount, DSA_STAGE1_SORT_BLOCK);
    }

    /**
     * Single-core batch path.
     *
     * Bottom path finds demote candidates. Top path reads all scores via
     * DataCopy + Cast and masks selected positions in UB.
     */
    __aicore__ inline void ProcessBatchSingleCore(int64_t batch)
    {
        const int32_t n = seqLenGm_.GetValue(batch);
        const int32_t m = selectedLenGm_.GetValue(batch);
        const int64_t scoreBase = batch * maxSeqLen_;
        const int64_t selectedBase = batch * maxSelectedLen_;
        const int64_t outBase = batch * static_cast<int64_t>(k_);

        if (k_ < DSA_UPDATE_INDEX_MAX_K) {
            ProcessBatchScalar(n, m, scoreBase, selectedBase, outBase);
            return;
        }

        ProcessBottomCandidatesRange(scoreBase, selectedBase, 0, m);
        ProcessTopCandidatesRange(scoreBase, selectedBase, m, 0, n);
        WriteFinalOutputs(selectedBase, outBase);
    }

    /**
     * Multi-core k=128 batch path.
     */
    __aicore__ inline void ProcessBatchMultiCore(int64_t batch, int32_t coreInBatch, int64_t blockIdx)
    {
        const int32_t n = seqLenGm_.GetValue(batch);
        const int32_t m = selectedLenGm_.GetValue(batch);
        const int64_t scoreBase = batch * maxSeqLen_;
        const int64_t selectedBase = batch * maxSelectedLen_;
        const int64_t outBase = batch * static_cast<int64_t>(k_);
        const int64_t groupBlockStart = batch * coreNumPerBatch_;

        int32_t selectedStart = 0;
        int32_t selectedEnd = 0;
        SplitRange(m, coreInBatch, coreNumPerBatch_, &selectedStart, &selectedEnd);
        ProcessBottomCandidatesRange(scoreBase, selectedBase, selectedStart, selectedEnd);
        StorePairToWorkspace(bottomBuf_, blockIdx, DSA_WORKSPACE_BOTTOM_LIST);

        AscendC::SyncAll();

        int32_t globalStart = 0;
        int32_t globalEnd = 0;
        SplitRange(n, coreInBatch, coreNumPerBatch_, &globalStart, &globalEnd);
        ProcessTopCandidatesRange(scoreBase, selectedBase, m, globalStart, globalEnd);
        StorePairToWorkspace(topBuf_, blockIdx, DSA_WORKSPACE_TOP_LIST);

        AscendC::SyncAll();

        if (coreInBatch == 0) {
            ReduceWorkspacePartialsIntoPair(bottomBuf_, groupBlockStart, DSA_WORKSPACE_BOTTOM_LIST);
            ReduceWorkspacePartialsIntoPair(topBuf_, groupBlockStart, DSA_WORKSPACE_TOP_LIST);
            WriteFinalOutputs(selectedBase, outBase);
        }
    }

    __aicore__ inline void WriteFinalOutputs(int64_t selectedBase, int64_t outBase)
    {
        for (int32_t i = 0; i < k_; ++i) {
            const int32_t bottomIdx = GetPairIndex(bottomBuf_, i);
            const int32_t topIdx = GetPairIndex(topBuf_, i);
            demoteIdxGm_.SetValue(outBase + i, bottomIdx);
            promoteIdxGm_.SetValue(outBase + i, topIdx);
            selectedIdxGm_.SetValue(selectedBase + bottomIdx, topIdx);
        }
    }

    __aicore__ inline void ProcessBatchScalar(int32_t n, int32_t m,
        int64_t scoreBase, int64_t selectedBase, int64_t outBase)
    {
        uint32_t bottomKeys[DSA_UPDATE_INDEX_MAX_K];
        int32_t bottomIdx[DSA_UPDATE_INDEX_MAX_K];
        uint32_t topKeys[DSA_UPDATE_INDEX_MAX_K];
        int32_t topIdx[DSA_UPDATE_INDEX_MAX_K];

        for (int32_t i = 0; i < k_; ++i) {
            bottomKeys[i] = DSA_SORT_KEY_MAX;
            bottomIdx[i] = 0;
            topKeys[i] = DSA_SORT_KEY_MIN;
            topIdx[i] = 0;
        }

        for (int32_t localIdx = 0; localIdx < m; ++localIdx) {
            const int32_t globalIdx = selectedIdxGm_.GetValue(selectedBase + localIdx);
            const uint32_t scoreKey = Bf16RawToSortKey(scoreGm_.GetValue(scoreBase + globalIdx));
            InsertBottomScalar(scoreKey, localIdx, bottomKeys, bottomIdx);
            scoreGm_.SetValue(scoreBase + globalIdx, DSA_BF16_LOW_SENTINEL_RAW);
        }

        for (int32_t globalIdx = 0; globalIdx < n; ++globalIdx) {
            const uint32_t scoreKey = Bf16RawToSortKey(scoreGm_.GetValue(scoreBase + globalIdx));
            InsertTopScalar(scoreKey, globalIdx, topKeys, topIdx);
        }

        for (int32_t i = 0; i < k_; ++i) {
            demoteIdxGm_.SetValue(outBase + i, bottomIdx[i]);
            promoteIdxGm_.SetValue(outBase + i, topIdx[i]);
            selectedIdxGm_.SetValue(selectedBase + bottomIdx[i], topIdx[i]);
        }
    }

private:
    AscendC::GlobalTensor<uint16_t> scoreGm_;
    AscendC::GlobalTensor<int32_t> selectedIdxGm_;
    AscendC::GlobalTensor<int32_t> promoteIdxGm_;
    AscendC::GlobalTensor<int32_t> demoteIdxGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> selectedLenGm_;
    AscendC::GlobalTensor<float> partialGm_;
    AscendC::GlobalTensor<int32_t> partialIdxGm_;

    int64_t batchSize_ = 0;
    int64_t maxSeqLen_ = 0;
    int64_t maxSelectedLen_ = 0;
    int32_t k_ = 0;
    int64_t usedCoreNum_ = 1;
    int32_t coreNumPerBatch_ = 1;

    AscendC::TBuf<AscendC::TPosition::VECCALC> valueBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> topBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bottomBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexTemplateBuf_;
};

} // namespace

extern "C" __global__ __aicore__ void dsa_update_index(GM_ADDR score, GM_ADDR selectedIdx,
    GM_ADDR seqLen, GM_ADDR selectedLen, GM_ADDR promoteIdx, GM_ADDR demoteIdx,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DsaUpdateIndexTilingData);
    GET_TILING_DATA_WITH_STRUCT(DsaUpdateIndexTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    AscendC::TPipe pipe;
    KernelDsaUpdateIndex op;
    op.Init(score, selectedIdx, promoteIdx, demoteIdx, seqLen, selectedLen, userWorkspace, &tilingData, &pipe);
    op.Process();
}
