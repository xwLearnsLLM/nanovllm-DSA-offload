/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qk_score_tiling.cpp
 * \brief
 */

#include "qk_score_tiling.h"
#include "../op_kernel/qk_score_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
namespace {
constexpr uint32_t M_BASE_SIZE = 512;
constexpr uint32_t S2_BASE_SIZE_DEFAULT = 512;
constexpr uint32_t S2_BASE_SIZE_DECODE = 1024;
constexpr uint32_t DECODE_S2_1024_THRESHOLD = 65536;
constexpr uint32_t DECODE_LONG_BATCH_1024_THRESHOLD = 32;

uint32_t SelectS2BaseMode(const QkScoreTilingInfo *tilingInfo)
{
    bool isTndOneTokenPerSeq = tilingInfo->inputQLayout == DataLayout::TND && tilingInfo->s1Size == tilingInfo->bSize;
    bool isLargeDecode = (tilingInfo->s2Size >= DECODE_S2_1024_THRESHOLD &&
                          tilingInfo->bSize >= DECODE_LONG_BATCH_1024_THRESHOLD);
    return isTndOneTokenPerSeq && isLargeDecode ? QK_S2_BASE_1024 : QK_S2_BASE_512;
}

uint32_t GetS2BaseSizeFromMode(uint32_t s2BaseMode)
{
    return s2BaseMode == QK_S2_BASE_1024 ? S2_BASE_SIZE_DECODE : S2_BASE_SIZE_DEFAULT;
}
} // namespace

ge::graphStatus QkScoreInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layOut == nullptr, OPS_LOG_E(opName_, "attr layout_query is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.layOutKey == nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.scoreCount == nullptr, OPS_LOG_E(opName_, "attr score_count is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("QkScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr, OPS_LOG_E(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OPS_LOG_E(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr, OPS_LOG_E(opName_, "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QkScoreInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void QkScoreInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void QkScoreInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(QK_SCORE);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(QK_SCORE);
}

ge::graphStatus QkScoreInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.scoreCount = attrs->GetAttrPointer<int32_t>(ATTR_SCORE_COUNT_INDEX);
    opParamInfo_.outputDType = attrs->GetStr(ATTR_OUTPUT_DTYPE_INDEX);

    if (opParamInfo_.layOut != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.scoreCount != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "score output length is:%d", *opParamInfo_.scoreCount);
    }
    if (opParamInfo_.outputDType != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "output dtype is:%s", opParamInfo_.outputDType);
    }
    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");

    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND")
        && (std::string(opParamInfo_.layOut) != std::string(opParamInfo_.layOutKey))),
        OPS_LOG_E(opName_, "under non-PA conditions, layout_query and layout_key should be equal."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND") && (std::string(opParamInfo_.layOutKey) != "BSND")
        && (std::string(opParamInfo_.layOutKey) != "TND")),
        OPS_LOG_E(opName_, "input attr layout_key only supported PA_BSND, BSND or TND"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(((std::string(opParamInfo_.layOut) != "BSND") && (std::string(opParamInfo_.layOut) != "TND")),
               OPS_LOG_E(opName_, "input attr layout_query only supported BSND or TND."), return ge::GRAPH_FAILED);
    OPS_ERR_IF((*opParamInfo_.scoreCount <= 0),
               OPS_LOG_E(opName_, "input attr score_count must be greater than 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_) && (inputKType_ == weightsType_);
    OPS_ERR_IF(!inDTypeAllEqual,
               OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be the same."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
               OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((outputType_ != ge::DT_FLOAT) && (outputType_ != ge::DT_BF16)),
               OPS_LOG_E(opName_, "The data type of the output scores must be float32 or bfloat16."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetQueryKeyAndOutLayout()
{
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"TND", DataLayout::TND},
        {"PA_BSND", DataLayout::BnBsND}
    };

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OPS_LOG_E(opName_, "key layout only supported PA_BSND, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            opParamInfo_.actualSeqLengths.tensor == nullptr,
            OPS_LOG_E(opName_, "key layout only supported PA_BSND, input actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OPS_LOG_E(opName_, "when layout_key is TND, input actual_seq_lengths_key must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
               opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"),
                   return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
                   opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"),
               return ge::GRAPH_FAILED);
    if (qLayout_ == DataLayout::TND) {
        OPS_ERR_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                   OPS_LOG_E(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                   opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(opName_, "input actual_seq_lengths_query data type only support int32"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kLayout_ != DataLayout::BnBsND && opParamInfo_.blockTable.tensor != nullptr,
                   OPS_LOG_E(opName_, "when key layout is not PA_BSND, input block_table must be null"),
                   return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::CheckShapeDim()
{
    OPS_ERR_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                   (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
    }
    if (kLayout_ == DataLayout::TND) {
        kExpectShapeDim = DIM_NUM_THREE;
    }
    OPS_ERR_IF(kShapeDim != kExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of key's shape should be %u, but now is %u", kExpectShapeDim, kShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of query's shape should be %u, but now is %u",
                qExpectShapeDim, qShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(outShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of scores's shape should be %u, but now is %u",
                qExpectShapeDim, outShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(weightsShapeDim == qExpectShapeDim - 1),
               OPS_LOG_E(opName_, "the dim num of weights's shape should be %u, but now is %u", qExpectShapeDim - 1,
                weightsShapeDim),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    OPS_LOG_I(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OPS_LOG_E(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OPS_LOG_I(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OPS_ERR_IF(n2Size_ != 1, OPS_LOG_E(opName_, "key shape[%u] is numhead, only support 1.", n2Index),
    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OPS_LOG_E(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    OPS_ERR_IF(gSize_ != 64, OPS_LOG_E(opName_, "N1 is %u, N2 is %u, N1 divided by N2 must equal 64.",
        n1Size_, n2Size_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetBatchSize()
{
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QkScoreInfoParser::GetHeadDim()
{
    uint32_t dIndex = DIM_IDX_TWO;
    switch (qLayout_) {
        case DataLayout::TND:
            // TND: [Total, N, D] -> D is the 2nd dimension
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND: [Batch, SeqLen, N, D] -> D is the 3rd dimension
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OPS_LOG_E(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OPS_ERR_IF(headDim_ != HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "input query's last dim head_dim only support 128."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OPS_ERR_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
               OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OPS_ERR_IF((blockCount_ == 0),
                OPS_LOG_E(opName_, "input key's block_count cannot be 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    OPS_ERR_IF((*opParamInfo_.scoreCount % blockSize_ != 0),
               OPS_LOG_E(opName_, "score_count must be a multiple of block_size under PA_BSND, got score_count %d, block_size %d.",
                         *opParamInfo_.scoreCount, blockSize_),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((*opParamInfo_.scoreCount > static_cast<int32_t>(maxBlockNumPerBatch_ * blockSize_)),
               OPS_LOG_E(opName_, "score_count %d exceeds physical block_table capacity %u.",
                         *opParamInfo_.scoreCount, maxBlockNumPerBatch_ * blockSize_),
               return ge::GRAPH_FAILED);
    s2Size_ = *opParamInfo_.scoreCount;
    OPS_LOG_I(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::GetS2Size()
{
    if (kLayout_ == DataLayout::BnBsND) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::ValidateInputShapesMatchQTnd()
{
    // -----------------------check BatchSize-------------------
    if (kLayout_ == DataLayout::TND) {
        OPS_ERR_IF(
        (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
            OPS_LOG_E(opName_,
                "TND case input actual_seq_lengths_query, actual_seq_lengths_key are %u, %ld respectively, they must be same.",
                bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize()),
            return ge::GRAPH_FAILED);
    } else { // kLayout_ PA_BSND
        OPS_ERR_IF(
        (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_),
            OPS_LOG_E(
                opName_,
                "TND case input actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are %u, %ld, %ld respectively, they must be same.",
                bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize(),
                opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
    }
    // -----------------------check T-------------------
    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                   (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                OPS_LOG_E(opName_, "TND case input query, weights, scores dim 0 are %u, %ld, %ld respectively, they must be same.",
                    qTsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::ValidateInputShapesMatchQBsnd()
{
    // -----------------------check BatchSize-------------------
    if (kLayout_ == DataLayout::BnBsND) {
        OPS_ERR_IF((opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_key, block_table dim 0 are %u, %ld, %ld respectively, they must be same.",
                    bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize(),
                    opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::BSND) {
        OPS_ERR_IF(opParamInfo_.key.shape->GetStorageShape().GetDim(0) != bSize_,
                OPS_LOG_E(opName_, "BSND case input query, key dim 0 are %u, %ld respectively, they must be same.",
                    bSize_, opParamInfo_.key.shape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((opParamInfo_.actualSeqLengths.tensor != nullptr) &&
                    (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_key dim 0 are %u, %ld respectively, they must be same.",
                    bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_),
                OPS_LOG_E(opName_, "BSND case input query, weight, scores dim 0 are %u, %ld, %ld respectively, they must be same.",
                    bSize_, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF((opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
                   (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_query dim 0 are %u, %ld respectively, they must be same.",
                    bSize_, opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);
    // -----------------------check S1-------------------
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                   (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                OPS_LOG_E(opName_, "BSND case input query, weight, scores dim 1 are %u, %ld, %ld, they must be same.",
                    s1Size_, opParamInfo_.weights.shape->GetStorageShape().GetDim(1),
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreInfoParser::ValidateInputShapesMatch()
{
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        if (ValidateInputShapesMatchQTnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (ValidateInputShapesMatchQBsnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
               OPS_LOG_E(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OPS_ERR_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
               OPS_LOG_E(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OPS_ERR_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
               OPS_LOG_E(opName_, "input query and output scores shape n2 dim must be same."),
               return ge::GRAPH_FAILED);
    // -----------------------check score output length-------------------
    const int64_t outputStride = opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1);
    OPS_ERR_IF((outputStride < *opParamInfo_.scoreCount),
               OPS_LOG_E(opName_, "output scores shape last dim must be >= attr score_count, got %ld and %d.",
                         outputStride, *opParamInfo_.scoreCount),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((static_cast<int64_t>(*opParamInfo_.scoreCount) != s2Size_),
               OPS_LOG_E(opName_, "attr score_count must equal logical key length %ld, but got %d.",
                         s2Size_, *opParamInfo_.scoreCount),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QkScoreInfoParser::GenerateInfo(QkScoreTilingInfo &qkInfo)
{
    qkInfo.opName = opName_;
    qkInfo.platformInfo = platformInfo_;
    qkInfo.opParamInfo = opParamInfo_;
    qkInfo.socVersion = socVersion_;

    qkInfo.bSize = bSize_;
    qkInfo.n1Size = n1Size_;
    qkInfo.n2Size = n2Size_;
    qkInfo.s1Size = s1Size_;
    qkInfo.s2Size = s2Size_;
    qkInfo.gSize = gSize_;

    qkInfo.inputQType = inputQType_;
    qkInfo.inputKType = inputKType_;
    qkInfo.outputType = outputType_;

    qkInfo.blockSize = blockSize_;
    qkInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    qkInfo.pageAttentionFlag = layOutKeyStr == "PA_BSND" ? true : false;
    qkInfo.scoreCount = *opParamInfo_.scoreCount;
    uint32_t outN2Dim = qLayout_ == DataLayout::TND ? DIM_IDX_ONE : DIM_IDX_TWO;
    qkInfo.outputStride = opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1);

    qkInfo.inputQLayout = qLayout_;
    qkInfo.inputKLayout = kLayout_;
}

ge::graphStatus QkScoreInfoParser::ParseAndCheck(QkScoreTilingInfo &qkInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(qkInfo);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForQkScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkScoreTiling::DoTiling(QkScoreTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    uint32_t s2BaseMode = SelectS2BaseMode(tilingInfo);
    uint32_t s2BaseSize = GetS2BaseSizeFromMode(s2BaseMode);
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t mm1ResSize = M_BASE_SIZE * s2BaseSize;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_scoreCount(tilingInfo->scoreCount);
    tilingData_.set_outputStride(tilingInfo->outputStride);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout,
                           s2BaseMode);
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForQkScore(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("QkScore", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    QkScoreTilingInfo qkInfo;
    QkScoreInfoParser parser(context);
    if (parser.ParseAndCheck(qkInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QkScoreTiling qkTiling(context);
    return qkTiling.DoTiling(&qkInfo);
}

IMPL_OP_OPTILING(QkScore)
    .Tiling(TilingForQkScore)
    .TilingParse<QkScoreCompileInfo>(TilingPrepareForQkScore);

} // namespace optiling
