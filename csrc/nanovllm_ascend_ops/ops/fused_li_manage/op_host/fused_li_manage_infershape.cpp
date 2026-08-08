/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t CACHE_SLOTS_INDEX = 4;
constexpr int64_t DECODE_OUTPUT_CAPACITY = 2048;

static ge::graphStatus InferShapeNanovllmFusedLiManage(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("NanovllmFusedLiManage", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    gert::Shape *indexOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, indexOutShape, return ge::GRAPH_FAILED);
    gert::Shape *slotsOutShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, slotsOutShape, return ge::GRAPH_FAILED);
    gert::Shape *missCountOutShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL(context, missCountOutShape, return ge::GRAPH_FAILED);
    gert::Shape *cacheSlotsOutShape = context->GetOutputShape(3);
    OPS_LOG_E_IF_NULL(context, cacheSlotsOutShape, return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape->GetDimNum() != 3,
               OPS_LOG_E(context, "query must be TND [B, N1, D], rank should be 3 but got %zu.",
                         queryShape->GetDimNum()),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyShape->GetDimNum() != 4,
               OPS_LOG_E(context, "key must be PA_BSND [num_blocks, block_size, N2, D], rank should be 4 but got %zu.",
                         keyShape->GetDimNum()),
               return ge::GRAPH_FAILED);

    indexOutShape->SetDimNum(3);
    indexOutShape->SetDim(0, queryShape->GetDim(0));
    indexOutShape->SetDim(1, keyShape->GetDim(2));
    indexOutShape->SetDim(2, DECODE_OUTPUT_CAPACITY);

    slotsOutShape->SetDimNum(3);
    slotsOutShape->SetDim(0, queryShape->GetDim(0));
    slotsOutShape->SetDim(1, keyShape->GetDim(2));
    slotsOutShape->SetDim(2, DECODE_OUTPUT_CAPACITY);

    missCountOutShape->SetDimNum(1);
    missCountOutShape->SetDim(0, queryShape->GetDim(0));

    const gert::Shape *cacheSlotsShape = context->GetInputShape(CACHE_SLOTS_INDEX);
    OPS_LOG_E_IF_NULL(context, cacheSlotsShape, return ge::GRAPH_FAILED);
    *cacheSlotsOutShape = *cacheSlotsShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNanovllmFusedLiManage(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("NanovllmFusedLiManage", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NanovllmFusedLiManage)
    .InferShape(InferShapeNanovllmFusedLiManage)
    .InferDataType(InferDataTypeNanovllmFusedLiManage);
} // namespace ops

