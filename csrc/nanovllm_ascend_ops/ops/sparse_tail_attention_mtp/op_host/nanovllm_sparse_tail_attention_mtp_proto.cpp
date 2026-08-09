/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
constexpr size_t MTP_QUERY_INPUT_INDEX = 0;

ge::graphStatus InferShapeNanovllmSparseTailAttentionMtp(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_LOG_E("NanovllmSparseTailAttentionMtp", "InferShapeContext is nullptr"),
        return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(MTP_QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED)
    *outputShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeNanovllmSparseTailAttentionMtp(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_LOG_E("NanovllmSparseTailAttentionMtp", "InferDataTypeContext is nullptr"),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(MTP_QUERY_INPUT_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(NanovllmSparseTailAttentionMtp)
    .InferShape(InferShapeNanovllmSparseTailAttentionMtp)
    .InferDataType(InferDataTypeNanovllmSparseTailAttentionMtp);
} // namespace ops
