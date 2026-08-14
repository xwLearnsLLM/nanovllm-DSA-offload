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
* \file vf_top_k.h
* \brief
*/

#ifndef VF_TOP_K_H
#define VF_TOP_K_H

namespace topkb32 {
template<typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    //                cout0 0-127 cout1 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    // 32bit    16bit
    MicroAPI::RegTensor<uint32_t> vreg0U16;
    // 32bit    16bit
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg0, pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg0, pregB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* nkValueBuf, __ubuf__ uint32_t* histogramsBuf, uint32_t bottomK)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx0;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Duplicate(btmK, bottomK);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx0;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx0, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx0Buf, sqzIdx0, alignIdx0);
    }
    MicroAPI::StoreUnAlignPost(idx0Buf, alignIdx0);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev0;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg0, idx0, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev0, idx0, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev0, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template<typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* idx0Buf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    //                0-127 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    // 0x000000fc -> 0xfcfcfcfc
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (MicroAPI::RegTensor<uint8_t>&)idx0, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg1, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg1, pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

// kValue      bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t* idx1Buf, __ubuf__ uint32_t* nkValueBuf,  __ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx1;

    MicroAPI::RegTensor<uint32_t> btmK1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK1, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx1;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK1, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx1, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx1Buf, sqzIdx1, alignIdx1);
    }
    MicroAPI::StoreUnAlignPost(idx1Buf, alignIdx1);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev1;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg1, idx1, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev1, idx1, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev1, idxPrev1, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev1, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg1);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK1, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template<typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    //                0-127 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    // 0x000000fc -> 0xfcfcfcfc
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);
        MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t>&)vreg1U16, (MicroAPI::RegTensor<uint8_t>&)vreg3U16);

        MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t>&)idx0, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t>&)idx1, pregB8);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::And(pregEQ, pregEQ0, pregEQ       k w  P  %MQ}	I}             	    ((    5    A$  I  Q            }        T   (    5    A$  I  Q            }        T   (    5    A$  I  Q            }        T   (    5    A$  I  Q            }        T   ((    5    A$  I  Q           }         (    5    A$  I  Q           }         (    5    A$  I  Q           }         (    5    A$  I  Q           }         ((                     5    A$     Q     MQ}QI%Q}U%9P  }Q=U%9P  }Y8    5    A$  I  1       iI< (                5    A$  M  5     U9-9=]8  5    A$  5   5    5     iI=%9  I    5     U9-9=]9  ((                     5    A$     Q     MQ}QI%Q}U%9P  }Q=U%9P  }=    5    A$  I  1       =9 (                5    A$  M  5     U9-9=]8  5    A$  5   5    5     iI=%9  I    5     U9-9=]9  ((               }               1           (        5    A$  1              }   5    A$  1        %MQ}%9Q1Y}         T         T         	              (        5    A$  1              }   5    A$  1        %MQ}%9Q1Y}         T         T         	                      ((        5    A$   %                         5    A$  I  Q           }         T     5    A$  I  Q           }         T    (        5    A$   %                         5    A$  I  Q           }         T     5    A$  I  Q           }         T    ((        5    A$  5   I       D    5    A$       5         }   5    A$  5   A        10    (        5    A$  5   I       D    5    A$       5         }   5    A$  5   A        10    (        5    A$  5   I       D    5    A$       5         }   5    A$  5   A        10    (        5    A$              }   5A5=  D      D           5    A$  I  Q           }                 (        5    A$              }   5A5=  D      D           5    A$  I  Q           }                 (        5    A$              }   5A5=  D      D           5    A$  I  Q           }                 ((        5    A$  5   I       D       5    A$       5         }   5    A$  5   A        10    (        5    A$  5   I       E     5    A$       5         }   5    A$  5   A        10    (        5    A$         D          D       D          (        5    A$         E        D          D          ((        5    A$  !               }         }   5    A$  !         	  Q     	%8  (                             5    A$  !         Q     U5U1Q                    E    (        5    A$  !               }         }   5    A$  !         	  Q     	%8  (                             5    A$  !         Q     U5U1Q                    E    (     ((    5    A$            }         }   MQ}QI%Q}U%9P  }Q=U%9P  }Y8       T                      (    5    A$            }         }   MQ}QI%Q}U%9P  }Q=U%9P  }=       T  =                   (    5    A$            }         }   MQ}QI%Q}U%9P  }Q=U%9P  }Y8       T                      (    5    A$            }         }   MQ}QI%Q}U%9P  }Q=U%9P  }=       T  =                   ((    5    A$  M               }   5    A$  M         %MQ}%9Q1Y}              	         T            T  =            (    5    A$  M               }   5    A$  M         %MQ}%9Q1Y}              	               T            T  =            ) ()}}    }  }|         -  Y%    }}    }|       }    Y      }}    }|       }             	    }}    }|       }       	    }}    }|       }       	    }}    }|       }       	    }}    }|       }       	   ) (    5    A$  5   I            5    A$       5          }   5    A$  5   A        10    ((    5    A$      M          M      A      I    H    ((    5    A$  U      I    M          %    ((    5    A$  I  Q            }      ,  (    5    A$  1              }   5    A$  1        %MQ}9=I4     ,    Y      ((               }                    }             (        5    A$  I  Q           }       (        5    A$  I  Q            }        (        5    A$  I  Q            }      %    ((        5    A$  5   I          5    A$       5          }   5    A$  5   A        10    ((        5    A$                     (        5    A$  1              }   5    A$  1        %MQ}9=I4                  	             (        5    A$               }   5A5=                   ,           (        5    A$  M             }   5    A$       5   5     MQ=I}I     %      5    A$  I  Q            }               (        5    A$  M    U            }   5    A$  A   1        A=MQ}5=}UAQ      	       %          %     (     (    5    A$  M    U     A        	         %     ((    5    A$  1    5  	          5    A$  5  Q     Y}MQ=I         5    A$  5  Q     Y}1=    ((    5    A$  I  Q            }        (    5    A$  I  Q            }        (    5    A$  I  Q            }        (    5    A$  I  Q            }        (    5    A$  1              }   5    A$  1        %MQ}	I}              	    (    5    A$  1              }   5    A$  1        %MQ}	I}              	    (    5    A$  1              }   5    A$  1        %MQ}	I}              	    (    5    A$  1              }   5    A$  1        %MQ}	I}              	    ((    5    A$  M    1                       }              (    5    A$  M    1                       }              (    5    A$  M    1                       }             ((       (    5    A$                               (    5    A$                               (    5    A$                               ((    5    A$  M               }   5    A$  M         %MQ}9=I4   Y                    ) ()}}    }  }|         %  Q=     Y%    }}    }|       }         %  	    }}    }|       }        	          }       %    }}    }|       }    Y            }    1    ) (    5    A$  5   I            5    A$       5          }   5    A$  5   A        10    ((    5    A$      M          M      A      I    H    ((    5    A$  U      I    M          %   ((    5    A$  I  Q            }      Y     (    5    A$  1              }   5    A$  1        %MQ}9=I4     Y       Y      ((    5    A$  I  Q            }       %     ((               }                    }     1            (        5    A$  I  Q           }       (        5    A$                  %             ((        5    A$  1              }   5    A$  1        %MQ}9=I4      %           	             ((        5    A$  5   I       P   5    A$       5          }   5    A$  5   A        10    ((        5    A$  I  Q            }      %  =   (        5    A$               }   5A5=  P      P      %         Y              ((        5    A$  M             }   5    A$       5   5     MQ=I}I     %  =     5    A$  I  Q            }             P  (        5    A$  M    U            }   5    A$  A   1        A=MQ}5=}UAQ        %  	       %  =         %    (     (    5    A$  M    U     A          %  	         %    ) ()}}    }  }|         %  E=     Y%    }}    }|       }         %  	    }}    }|       }        	          }       %    }}    }|       }    Y     ) (    5    A$  5   I            5    A$       5          }   5    A$  5   A        10    ((    5    A$  U      I    M          %   ((    5    A$  I  Q            }      Y     (    5    A$  1              }   5    A$  1        %MQ}9=I4     Y       Y      ((    5    A$  I  Q            }       %     ((    5    A$  I  Q           }       (    5    A$                  %    ((    5    A$  1              }   5    A$  1        %MQ}9=I4      %           	    ((    5    A$  5   I       D   5    A$       5          }   5    A$  5   A        10    ((    5    A$  I  Q            }      %  =   (    5    A$               }   5A5=  D      D      %         Y              ((    5    A$  M             }   5    A$       5   5     MQ=I}I     %  =     5    A$  I  Q            }             D  (    5    A$  M    U            }   5    A$  A   1        A=MQ}5=}UAQ        %  	       %  =         %    (    5    A$  M    U     A          %  	         %    ) ()}}    }  }|         Y    Q=     Y%    }}    }|       }         Y    	    }}    }|       }        	    }}    }|       }    Y            }    1    ) (    5    A$  5   I            5    A$       5          }   5    A$  5   A        10    ((    5    A$      M          M      A      I    H    ((    5    A$  U      I    M          Y     ((    5    A$  I  Q            }      Y     (    5    A$  1              }   5    A$  1        %MQ}9=I4     Y       Y      ((    5    A$  I  Q            }       %     ((               }                    }     1            (        5    A$  1              }   5    A$  1        %MQ}9=I4      %           	             ((        5    A$  5   I       P   5    A$       5          }   5    A$  5   A        10    ((        5    A$  I  Q            }      Y    =   (        5    A$               }   5A5=  P      P      %         Y              ((        5    A$  M             }   5    A$       5   5     MQ=I}I     Y    =        %          P  (        5    A$  M    U            }   5    A$  A   1        A=MQ}5=}UAQ        Y    	       Y    =         Y      (     (    5    A$  M    U     A          Y    	         Y      ) ()}}    }  }|         Y    E=     Y%    }}    }|       }         Y    	    }}    }|       }        	    }}    }|       }    Y     ) (    5    A$  5   I            5    A$       5          }   5    A$  5   A        10    ((    5    A$  U      I    M          Y     ((    5    A$  I  Q            }      Y     (    5    A$  1              }   5    A$  1        %MQ}9=I4     Y       Y      ((    5    A$  I  Q            }       %     ((    5    A$  1              }   5    A$  1        %MQ}9=I4      %           	    ((    5    A$  5   I       D   5    A$       5          }   5    A$  5   A        10    ((    5    A$  I  Q            }      Y    =   (    5    A$               }   5A5=  D      D      %         Y              ((    5    A$  M             }   5    A$       5   5     MQ=I}I     Y    =        %          D  (    5    A$  M    U            }   5    A$  A   1        A=MQ}5=}UAQ        Y    	       Y    =         Y      (    5    A$  M    U     A          Y    	         Y      ) ()}}      }|             1 Q  -Y       1    Q            }          %  1     (                                      1    Q            }          Y    1     (                                      1    Q            }         1     (                                      1    Q            }       %  1     (                                      1    Q            }       Y    1     (                                      1    Q            }              1     (                                      1    Q            }        1     (                                      1    Q            }        1     (                                      1    Q            }        1     (                                      1    Q            }        1     (                                      1    Q            }      Y    1     (                                      }     , (                                      }    M  1   ) (    }}    }|       }         %  	      }}    }|       }         %  1       A        (    }}    }|       }         Y    	      }}    }|       }         Y    1       A        (    }}    }|       }        	      }}    }|       }        1       A        (    }}    }|       }      %  	      }}    }|       }      %  1       A        (    }}    }|       }      Y    	      }}    }|       }      Y    1       A        (    }}    }|       }             	      }}    }|       }             1       A        (    }}    }|       }       	      }}    }|       }       1       A        (    }}    }|       }       	      }}    }|       }       1       A        (    }}    }|       }       	      }}    }|       }       1       A        (    }}    }|       }       	      }}    }|       }       1       A        (    }}    }|       }     Y    	      }}    }|       }     Y    1       A        ((          }        ,     M  1        ,     (          }       %       (                     ((                }        M           (                }        M           ((          }            1   9        M  1           M                  M     (          }       1   9        M  1           M                   M      (          }      1   9         ,                ((                     (    !             Y%          }             	         	              1   9          (           Q     	  Y%        	      Y    	              	          ,  (    !         M     Y%          }             	         	        	              1   9          (       M     Q     	  Y%        	      Y    	      Y    	              	    (    !         Q    Y%          }             	         	        	        	              1   9          (       Q    Q     	  Y%        	      Y    	      Y    	              	    (    !         1   Y%          }             	         	        	        	        	              1   9          (       -  Y%      Y    	              	        	        	        	        	    ((             (         O       9        j       (       Y    Q=     Y%          Y    	         	      Y    	         1   9    (               O &7 ?        9        j   r K &   +    c f     j r' V#  _ * V (         }    Y    9              M          M      A      I    H    (              &   g r    O    '  9        j V  <(         }        Y    9        ,      Y    9                  }    (              }                  1   9          (             }    Y    9  A  1               M          M      A      I    H    (                Y    9  A  1        Y    9                   }            Y    9     (                  R       ~  &   '  9           j    ,(               Y    E=     Y%          Y    	         	               Y    	    (                (                  (         (     ((         O       9        j     (       %  Q=     Y%          %  	         	           }         Y    	         1   9    (             O &7 ?        9        j   r K &   +    c f     j r' V#  _ * V (         }    %  9              M          M      A      I    H    (         }        %  9        ,      %  9                  }    (              }                  1   9          (             }    %  9  A  1               M          M      A      I    H    (                %  9  A  1        %  9                   }            %  9     (                  R       ~  &   '  9           j    ,(                 %            (               %  E=     Y%          %  	         	                  %      Y    	    (                (                  (         (     ) ) (      ￿￿￿
