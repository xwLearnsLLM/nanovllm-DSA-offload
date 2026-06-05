#include "kernel_operator.h"
#include "../../../common/kernels/types.h"

namespace {

constexpr uint32_t DSA_INDEXER_PROJECT_MAX_HEAD_DIM = 256;
constexpr uint32_t DSA_INDEXER_PROJECT_MAX_HEADS = 128;

template <typename scalar_t>
class DsaIndexerProjectPostKernel {
public:
    __aicore__ inline DsaIndexerProjectPostKernel() {}

    __aicore__ inline void Init(
        GM_ADDR qIn,
        GM_ADDR kIn,
        GM_ADDR weightsIn,
        GM_ADDR cos,
        GM_ADDR sin,
        GM_ADDR qOut,
        GM_ADDR kOut,
        GM_ADDR weightsOut,
        uint32_t numTokens,
        uint32_t nHead,
        uint32_t headDim,
        uint32_t ropeDim,
        float scoreScale,
        AscendC::TPipe* pipe)
    {
        qInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(qIn));
        kInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kIn));
        weightsInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightsIn));
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cos));
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(sin));
        qOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(qOut));
        kOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kOut));
        weightsOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(weightsOut));
        numTokens_ = numTokens;
        nHead_ = nHead;
        headDim_ = headDim;
        ropeDim_ = ropeDim;
        ropeHalf_ = ropeDim / 2;
        scoreScale_ = scoreScale;
        qRows_ = numTokens_ * nHead_;
        kRows_ = numTokens_;
        weightRows_ = numTokens_;
        totalRows_ = qRows_ + kRows_ + weightRows_;

        pipe->InitBuffer(rowBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(scalar_t));
        pipe->InitBuffer(cosBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(scalar_t));
        pipe->InitBuffer(sinBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(scalar_t));
        pipe->InitBuffer(x0Buf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(x1Buf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(cosFloatBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(sinFloatBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(tmpBuf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(out0Buf_, DSA_INDEXER_PROJECT_MAX_HEAD_DIM * sizeof(float));
        pipe->InitBuffer(weightsFloatBuf_, DSA_INDEXER_PROJECT_MAX_HEADS * sizeof(float));
        pipe->InitBuffer(weightsOutBuf_, DSA_INDEXER_PROJECT_MAX_HEADS * sizeof(scalar_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockId = AscendC::GetBlockIdx();
        const uint32_t stride = AscendC::GetBlockNum();
        for (uint32_t row = blockId; row < totalRows_; row += stride) {
            if (row < qRows_) {
                ProcessRopeRow(qInGm_, qOutGm_, row, row / nHead_);
            } else if (row < qRows_ + kRows_) {
                const uint32_t kRow = row - qRows_;
                ProcessRopeRow(kInGm_, kOutGm_, kRow, kRow);
            } else {
                ProcessWeightsRow(row - qRows_ - kRows_);
            }
        }
    }

private:
    __aicore__ inline void ProcessRopeRow(
        AscendC::GlobalTensor<scalar_t>& inGm,
        AscendC::GlobalTensor<scalar_t>& outGm,
        uint32_t row,
        uint32_t token)
    {
        AscendC::LocalTensor<scalar_t> rowLocal = rowBuf_.Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> cosLocal = cosBuf_.Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> sinLocal = sinBuf_.Get<scalar_t>();
        AscendC::LocalTensor<float> x0 = x0Buf_.Get<float>();
        AscendC::LocalTensor<float> x1 = x1Buf_.Get<float>();
        AscendC::LocalTensor<float> cosFloat = cosFloatBuf_.Get<float>();
        AscendC::LocalTensor<float> sinFloat = sinFloatBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf_.Get<float>();
        AscendC::LocalTensor<float> out0 = out0Buf_.Get<float>();

        const uint64_t rowBase = static_cast<uint64_t>(row) * headDim_;
        const uint64_t cosBase = static_cast<uint64_t>(token) * ropeDim_;
        AscendC::DataCopy(rowLocal, inGm[rowBase], headDim_);
        // cos/sin are passed as full rope_dim tensors. Real model usually stores
        // duplicated halves for NeoX, but the op must still honor both halves.
        AscendC::DataCopy(cosLocal, cosGm_[cosBase], ropeDim_);
        AscendC::DataCopy(sinLocal, sinGm_[cosBase], ropeDim_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Cast(x0, rowLocal, AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::Cast(x1, rowLocal[ropeHalf_], AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::Cast(cosFloat, cosLocal, AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::Cast(sinFloat, sinLocal, AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();

        // NeoX RoPE: first half = x0*cos - x1*sin, second half = x1*cos + x0*sin.
        AscendC::Mul(out0, x0, cosFloat, ropeHalf_);
        AscendC::Mul(tmp, x1, sinFloat, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(out0, out0, tmp, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rowLocal, out0, AscendC::RoundMode::CAST_RINT, ropeHalf_);

        AscendC::Cast(cosFloat, cosLocal[ropeHalf_], AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::Cast(sinFloat, sinLocal[ropeHalf_], AscendC::RoundMode::CAST_NONE, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(out0, x1, cosFloat, ropeHalf_);
        AscendC::Mul(tmp, x0, sinFloat, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(out0, out0, tmp, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rowLocal[ropeHalf_], out0, AscendC::RoundMode::CAST_RINT, ropeHalf_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(outGm[rowBase], rowLocal, headDim_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void ProcessWeightsRow(uint32_t row)
    {
        AscendC::LocalTensor<float> weightsFloat = weightsFloatBuf_.Get<float>();
        AscendC::LocalTensor<scalar_t> weightsOut = weightsOutBuf_.Get<scalar_t>();
        const uint64_t base = static_cast<uint64_t>(row) * nHead_;
        AscendC::DataCopy(weightsFloat, weightsInGm_[base], nHead_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Muls(weightsFloat, weightsFloat, scoreScale_, nHead_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(weightsOut, weightsFloat, AscendC::RoundMode::CAST_RINT, nHead_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(weightsOutGm_[base], weightsOut, nHead_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    AscendC::GlobalTensor<scalar_t> qInGm_;
    AscendC::GlobalTensor<scalar_t> kInGm_;
    AscendC::GlobalTensor<float> weightsInGm_;
    AscendC::GlobalTensor<scalar_t> cosGm_;
    AscendC::GlobalTensor<scalar_t> sinGm_;
    AscendC::GlobalTensor<scalar_t> qOutGm_;
    AscendC::GlobalTensor<scalar_t> kOutGm_;
    AscendC::GlobalTensor<scalar_t> weightsOutGm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x0Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x1Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cosFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sinFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> out0Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightsFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightsOutBuf_;
    uint32_t numTokens_ = 0;
    uint32_t nHead_ = 0;
    uint32_t headDim_ = 0;
    uint32_t ropeDim_ = 0;
    uint32_t ropeHalf_ = 0;
    float scoreScale_ = 1.0f;
    uint32_t qRows_ = 0;
    uint32_t kRows_ = 0;
    uint32_t weightRows_ = 0;
    uint32_t totalRows_ = 0;
};

template <typename scalar_t>
__aicore__ inline void RunDsaIndexerProjectPost(
    GM_ADDR qIn,
    GM_ADDR kIn,
    GM_ADDR weightsIn,
    GM_ADDR cos,
    GM_ADDR sin,
    GM_ADDR qOut,
    GM_ADDR kOut,
    GM_ADDR weightsOut,
    uint32_t numTokens,
    uint32_t nHead,
    uint32_t headDim,
    uint32_t ropeDim,
    float scoreScale)
{
    AscendC::TPipe pipe;
    DsaIndexerProjectPostKernel<scalar_t> op;
    op.Init(qIn, kIn, weightsIn, cos, sin, qOut, kOut, weightsOut, numTokens, nHead, headDim, ropeDim, scoreScale, &pipe);
    op.Process();
}

} // namespace

extern "C" __global__ __aicore__ void dsa_indexer_project_post_half(
    GM_ADDR qIn,
    GM_ADDR kIn,
    GM_ADDR weightsIn,
    GM_ADDR cos,
    GM_ADDR sin,
    GM_ADDR qOut,
    GM_ADDR kOut,
    GM_ADDR weightsOut,
    uint32_t numTokens,
    uint32_t nHead,
    uint32_t headDim,
    uint32_t ropeDim,
    float scoreScale)
{
    RunDsaIndexerProjectPost<half>(qIn, kIn, weightsIn, cos, sin, qOut, kOut, weightsOut,
        numTokens, nHead, headDim, ropeDim, scoreScale);
}

#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
extern "C" __global__ __aicore__ void dsa_indexer_project_post_bfloat16_t(
    GM_ADDR qIn,
    GM_ADDR kIn,
    GM_ADDR weightsIn,
    GM_ADDR cos,
    GM_ADDR sin,
    GM_ADDR qOut,
    GM_ADDR kOut,
    GM_ADDR weightsOut,
    uint32_t numTokens,
    uint32_t nHead,
    uint32_t headDim,
    uint32_t ropeDim,
    float scoreScale)
{
    RunDsaIndexerProjectPost<bfloat16_t>(qIn, kIn, weightsIn, cos, sin, qOut, kOut, weightsOut,
        numTokens, nHead, headDim, ropeDim, scoreScale);
}
#endif

namespace vllm_ascend {

extern void dsa_indexer_project_post_impl(
    AscendType type,
    void* stream,
    void* q_in,
    void* k_in,
    void* weights_in,
    void* cos,
    void* sin,
    void* q_out,
    void* k_out,
    void* weights_out,
    uint32_t num_tokens,
    uint32_t n_head,
    uint32_t head_dim,
    uint32_t rope_dim,
    float score_scale,
    uint32_t block_dim)
{
    if (type == AscendType::FP16) {
        dsa_indexer_project_post_half<<<block_dim, nullptr, stream>>>(q_in, k_in, weights_in, cos, sin, q_out,
            k_out, weights_out, num_tokens, n_head, head_dim, rope_dim, score_scale);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        dsa_indexer_project_post_bfloat16_t<<<block_dim, nullptr, stream>>>(q_in, k_in, weights_in, cos, sin, q_out,
            k_out, weights_out, num_tokens, n_head, head_dim, rope_dim, score_scale);
#endif
    }
}

} // namespace vllm_ascend
