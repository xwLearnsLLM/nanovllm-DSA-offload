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
 * \file qk_score_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SCORE_COUNT_INDEX = 2;

static ge::graphStatus InferShapeQkScore(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QkScore", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    gert::Shape *outShape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutQueryPtr, return ge::GRAPH_FAILED);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutKeyPtr, return ge::GRAPH_FAILED);
    const int64_t *scoreCount = attrs->GetInt(ATTR_SCORE_COUNT_INDEX);
    OPS_LOG_E_IF_NULL(context, scoreCount, return ge::GRAPH_FAILED);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OPS_ERR_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OPS_LOG_E(context, "The attr layout_query should be TND or BSND, but got %s.", inputLayoutQueryPtrStr.c_str()),
        return ge::GRAPH_FAILED);

    int32_t keyNDimIndex = (inputLayoutKeyPtrStr == "TND") ? 1 : 2;
    outShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        OPS_ERR_IF(
            queryShape->GetDimNum() != 4,
            OPS_LOG_E(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        outShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        outShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        outShape->SetDim(2, keyShape->GetDim(keyNDimIndex)); // 2:Dim N2
        outShape->SetDim(3, *scoreCount);                    // 3:Dim S2 scores
    } else {
        OPS_ERR_IF(
            queryShape->GetDimNum() != 3,
            OPS_LOG_E(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        outShape->SetDim(0, queryShape->GetDim(0));                      // 0:Dim T
        outShape->SetDim(1, keyShape->GetDim(keyNDimIndex));             // 1:Dim N2
        outShape->SetDim(2, *scoreCount);                                // 2:Dim S2 scores
    }
    OPS_LOG_D(context->GetNodeName(), "QkScore InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQkScore(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QkScore", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "Enter QkScore InferDataType impl.");
    ge::DataType outputType = ge::DT_FLOAT;
    context->SetOutputDataType(0, outputType);
    OPS_LOG_D(context->GetNodeName(), "QkScore InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkScore)
    .InferShape(InferShapeQkScore)
    .InferDataType(InferDataTypeQkScore);
} // namespace ops
