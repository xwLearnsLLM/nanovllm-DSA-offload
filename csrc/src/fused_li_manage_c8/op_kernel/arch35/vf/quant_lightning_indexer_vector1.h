/**
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file quant_lightning_indexer_vector1.h
 * \brief
 */
#ifndef quant_lightning_indexer_VECTOR1_H
#define quant_lightning_indexer_VECTOR1_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO      = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK  = 0x7FC00000;
    static constexpr UInt ALL_ONE   = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO      = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK  = 0x7FC0;
    static constexpr UInt ALL_ONE   = 0xFFFF;
};


template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOnes;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOnes,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}


template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey,
                                               AscendC::MicroAPI::RegTensor<FloatT>& inVal,
                                               FloatSortConstCtx<FloatT>& ctx,
                                               AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto& inBits = (AscendC::MicroAPI::RegTensor<UInt>&)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.allOnes, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOnes, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey0,
                                                 AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey1,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal0,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal1,
                                                 FloatSortConstCtx<FloatT>& ctx,
                                                 AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp[2];
    AscendC::MicroAPI::RegTensor<UInt> regMask[2];
    AscendC::MicroAPI::MaskReg regSelectNan[2];
    AscendC::MicroAPI::MaskReg regSelectSign[2];

    auto& inBits0 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal0;
    auto& inBits1 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal1;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey0, ctx.allOnes, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.allOnes, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.allOnes, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.allOnes, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}


template <typename T, size_t N>
__simd_callee__ inline void DuplicateZero(AscendC::MicroAPI::RegTensor<T> (&regArray)[N],
                                          AscendC::MicroAPI::MaskReg& mask)
{
    static_assert(N <= 4, "N must be <= 4");
    //                ,          fatal error: error in backend: Unsupported Inst must be hoisted.
    if constexpr (N >= 1) {
        AscendC::MicroAPI::Duplicate(regArray[0], static_cast<T>(0), mask);
    }
    if constexpr (N >= 2) {
        AscendC::MicroAPI::Duplicate(regArray[1], static_cast<T>(0), mask);
    }
    if constexpr (N >= 3) {
        AscendC::MicroAPI::Duplicate(regArray[2], static_cast<T>(0), mask);
    }
    if constexpr (N >= 4) {
        AscendC::MicroAPI::Duplicate(regArray[3], static_cast<T>(0), mask);
    }
}


template <typename T, size_t N, bool ApplyRelu = true>
__simd_callee__ inline void WeightedAccum(AscendC::MicroAPI::RegTensor<T> (&accum)[N],
                                          AscendC::MicroAPI::RegTensor<T> (&input)[N],
                                          AscendC::MicroAPI::RegTensor<T>& weight,
                                          AscendC::MicroAPI::MaskReg& mask)
{
    static_assert(N <= 2, "N must be <= 2");
    // ---- Relu block ----
    if constexpr (ApplyRelu) {
        if constexpr (N >= 1) {
            AscendC::MicroAPI::Relu(input[0], input[0], mask);
        }
        if constexpr (N >= 2) {
            AscendC::MicroAPI::Relu(input[1], input[1], mask);
        }
    }
    // ---- MulAdd block ----
    if constexpr (N >= 1) {
        AscendC::MicroAPI::MulAddDst(accum[0], input[0], weight, mask);
    }
    if constexpr (N >= 2) {
        AscendC::MicroAPI::MulAddDst(accum[1], input[1], weight, mask);
    }
}


__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float>& dst,
                                          AscendC::MicroAPI::RegTensor<float>& src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
    AscendC::MicroAPI::Duplicate(brcGatherIndex, laneIdx);
    AscendC::MicroAPI::Gather(dst, src, brcGatherIndex);
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float>& dst,
                                          __local_mem__ float* src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ]
                                             const LocalTensor<float> &qk_,       // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<float> &weight_,   // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,   // qScale [G]          [64    ]
                                             const int gSize)                     // G 64
{
    auto weight = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto qk = (__local_mem__ float*)qk_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;

        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum0[2];
        AscendC::MicroAPI::RegTensor<float> regSum1[2];
        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                       MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW, weight);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale);
        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        MicroAPI::LoadAlign<float>(regKScale[0], kScale);
        MicroAPI::LoadAlign<float>(regKScale[1], kScale + 64);

        // unroll2
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
            MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride   128,                   bank   
            MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
            BroadcastLane(regwBrc, regW, i);
            WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

            MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
            MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
            BroadcastLane(regwBrc, regW, i + 1);
            WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
        }

        AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
        AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

        AscendC::MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
        // interleave cast ==> regSum[1] high regSum[0] low
        AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

        AscendC::MicroAPI::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        // normal store
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}


// bfloat16_t in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ]
                                             const LocalTensor<bfloat16_t> &qk_,  // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,           // unused for bfloat16
                                             const LocalTensor<float> &weight_,   // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,   // qScale [G]          [64    ]
                                             const int gSize)                     // G 64
{
    auto weight = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto qk = (__local_mem__ bfloat16_t*)qk_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regQK[4];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regQKB16[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc[2];
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[2];

        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);


        using CastTrait = AscendC::MicroAPI::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD  = {AscendC::MicroAPI::RegLayo ^     k w  qQ               M    |        M     mL 	   t        l      t(                                                    }   M    M      (                                                    1    Q               M    |        M     l   t          l      t(                                                    }   M    M      (                                                         M                               ) (                    }}     }   }|              |   A        (          M         }}     }   }|         M    |   A        (          M         }}     }   }|         M    |   A        (                }}     }   }|          |   A        (                 }}     }   }|       }      |   A        ((                                   M      (          M         M         M    M      (                       M      (        M    M             (                          M      ((    }}Y}M=A}|(     (               5    A$  I  Q                 	  l t (               5    A$  I  Q                E, l t (               5    A$  I  Q                E, l t (               5    A$  I  Q                ]l t ((               5    A$  I  Q                EM    l t (               5    A$  I  Q                -M    l t (               5    A$  I  Q                M   l t (               5    A$  I  Q                M   l t (               5    A$  5   I                     5    A$       5                  5    A$  5   A        10    (               5    A$  5   I                     5    A$       5            }          5    A$  5   A        10    ((            M                  }          (        %       M                            ((                         5    A$     Q         Q      Q   }Y8    5    A$  I  1       iI<  5    A$  M  5     9=}MP (                                                                       5    A$  5   5    5     5I%9  I    5     MQ}I=U9  (                         5    A$     Q         Q      Q   }=    5    A$  I  1       =9  5    A$  M  5     9=}MP (                                                                      5    A$  5   5    5     iI=%9  I    5     MQ}I=U9  ((               5    A$  1                  ]l t           (               5    A$  1                  ]l t           (               5    A$  1                  EM    l t   M       (               5    A$  1                  EM    l t   M       (               5    A$  5      ]l t     ]l t     EM    l t            (               5    A$  5      ]l t     ]l t     EM    l t            (              ]l w  9           B#    R (               5    A$  M                       5    A$  M         %MQ}9=I4              ]l t            (               5    A$  1    5  	          5    A$  5  Q     Y}MQ=I         5    A$  5  Q     Y}1=    (                i       M               (                i       M               ((        5    A$  1                  -M    l t   M       (        5    A$  1                  -M    l t   M            ((                   }             }                  }    M            (            5    A$  1                  E, l t                 (            5    A$  1                  E, l t                    Y1M       (            5    A$  1                  E, l t                 (            5    A$  1                  E, l t                    Y1M       (                   B#    R     V   O     n    (            	        1        	  l t     ]l t     (               ]      ^           3 R 1        v  >C >Y         <(            	        1        	  l t              (                   5    A$  I       E, l t     E, l t            (                   5    A$  I       E, l t     E, l t            (                   5    A$  I       E, l t     E, l t            (                   5    A$  I       E, l t     E, l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (         ((                 M            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            (((                                                      (               5    A$  I  Q              }      M  	  l t (               5    A$  I  Q            }      =  l t (               5    A$   %             M   l t     M   l t     M   l t     M   l t  (               5    A$   %             M   l t     M   l t     M   l t     M   l t  (               5    A$              }              Q      Q   }=     M  	  l t     M   l t            (               5    A$              }              Q      Q   }=     M  	  l t     M   l t            (               5    A$              }              Q      Q   }Y8     M  	  l t     M   l t            (               5    A$              }              Q      Q   }Y8     M  	  l t     M   l t            ((            ` Q M       -           }      =  l t     =  l t     M  	  l t     M  	  l t                    (               5    A$  M               }          5    A$  M         %MQ}9=I4           =  l t            (               5    A$  M               }          5    A$  M         %MQ}9=I4           =  l t            (     ) (((        ]L   (                         )}}      }|             5  ]       I     M          1    Q            }       |              l   L 	   t     l      t(                                                    }     M      (                                                    1    Q              }      |         y   l     L 	   t  l      t(                                                    }    Y1M      (                                                    }    M             M         (                                                    1    Q                    |              l   t          l      t(                                                    }        M      (                                                    1    Q               M    |        M     mL 	   t        l      t(                                                    }   M    M      (                                                    1    Q               M    |        M     l   t          l      t(                                                    }   M    M      (                                                         M                               ) (                    }}     }   }|              |   A        (          M         }}     }   }|         M    |   A        (          M         }}     }   }|         M    |   A        (                }}     }   }|         }     |   A        (                 }}     }   }|       }      |   A        ((                                   M      (          M         M         M    M      (                       M      (        M    M             (                          M      ((    }}Y}M=A}|(     (               5    A$  I  Q                 	  l t (               5    A$  I  Q                E, l t (               5    A$  I  Q                E, l t (               5    A$  I  Q                ]l t (               5    A$  I  Q              }      E-  l t ((               5    A$  I  Q                EM    l t (               5    A$  I  Q                -M    l t (               5    A$  I  Q                M   l t (               5    A$  I  Q                M   l t (               5    A$  5   I                     5    A$       5                  5    A$  5   A        10    (               5    A$  5   I                     5    A$       5            }          5    A$  5   A        10    ((            M                  }          (        %       M                            ((                 Q              5    A$     Q     (                            Q         Q         }Y8           5    A$  I  1       iI<         5    A$  M  5     U9-9=]8 (                                                                   5    A$  5   5    5     iI=%9  I    5     U9-9=]9  (                            Q         Q         }=            5    A$  I  1       =9         5    A$  M  5     U9-9=]8 (                                                                   5    A$  5   5    5     iI=%9  I    5     U9-9=]9  ((                         5    A$     Q         Q      Q   }Y8    5    A$  I  1       iI<  5    A$  M  5     9=}MP (                                                                       5    A$  5   5    5     5I%9  I    5     MQ}I=U9  (                         5    A$     Q         Q      Q   }=    5    A$  I  1       =9  5    A$  M  5     9=}MP (                                                                      5    A$  5   5    5     iI=%9  I    5     MQ}I=U9  ((               5    A$  1                  ]l t           (               5    A$  1                  ]l t           (               5    A$  1                  EM    l t   M       (               5    A$  1                  EM    l t   M       (               5    A$  5      ]l t     ]l t     EM    l t            (               5    A$  5      ]l t     ]l t     EM    l t            (               g  w  [  3    c f  >      w  <(               5    A$  M                       5    A$  M         %MQ}9=I4              ]l t            (               5    A$  M                       5    A$  M         %MQ}9=I4              ]l t            (                i       M               (                i       M               ((                          (        5    A$  1               5    A$  1        %MQ}%9Q1Y}       -M    l t     -M    l t   M       ((                   }             }                  }    M            (                   5    A$  1                }      E-  l t                     I  M      b        3   B  r            ((                   5    A$  1                }      E-  l t                     I  M      b        3   B  r            ((                   5    A$  1                      5    A$  1        %MQ}	I}        	  l t               (                   5    A$  1                      5    A$  1        %MQ}	I}        	  l t               (                              (                   5    A$                     }       Q         }Y8     E, l t     E-  l t            (                   5    A$                     }       Q         }=     E, l t     E-  l t            (                   5    A$                     }       Q         }Y8     E, l t     E-  l t            (                   5    A$                     }       Q         }=     E, l t     E-  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (                   5    A$  5          M   l t     E, l t      	  l t            (         ((                 M            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            (               5    A$  5      M   l t     M   l t     -M    l t            ((                                                      (               5    A$  I  Q              }      M  	  l t (               5    A$  I  Q            }      =  l t (               5    A$              }              Q      Q   }=     M  	  l t     M   l t            (               5    A$              }              Q      Q   }=     M  	  l t     M   l t            (               5    A$              }              Q      Q   }Y8     M  	  l t     M   l t            (               5    A$              }              Q      Q   }Y8     M  	  l t     M   l t            ((            ` Q M       -           }      =  l t     =  l t     M  	  l t     M  	  l t                    (               5    A$  M               }          5    A$  M         %MQ}9=I4           =  l t            (               5    A$  M               }          5    A$  M         %MQ}9=I4           =  l t            (     ) (()                  E-}P           M=I}P )}}      }|             	    5  ]       I     M         1    Q      M=I}P      |              mL 	   t     l      t(                                                        }     M      (                                                        1    Q      E-}P     |              y   m  L 	   t  l      t(                                                        }    Y1M      (                                                        }    M      (                                                        1    Q                    |              mt          l      t(                                                        }        M      (                                                        1    Q               M    |        M     mL 	   t     l      t(                                                        }   M    M      (                                                        1    Q               M    |        M     mt          l      t(                                                        }   M    M      (                                                             M                               (                                                                  ) (        j >  R  2  g     7      B;    R  "C    : (                                   (               (     (                     (        5  ]       I     M       |     M      (                                 |    Y1M         M      (                                     |        M      (                                M    |   M    M      (                                M    |   M    M      (                                M     (            (        5  ]       I     M      |    |    Y1M             |   M    |   M    |   M     (     ) () ((     ￿￿￿
