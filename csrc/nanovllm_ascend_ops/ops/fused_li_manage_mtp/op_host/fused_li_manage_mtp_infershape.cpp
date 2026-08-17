/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
constexpr uint32_t QUERY_INDEX = 2;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 10;
constexpr uint32_t CACHE_STATE_INDEX = 11;
constexpr uint32_t CACHE_SLOTS_INDEX = 12;
constexpr int64_t TOPK = 2048;
constexpr int64_t UNION_CAPACITY = 8192;

static ge::graphStatus InferShapeNanovllmFusedLiManageMtp(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("NanovllmFusedLiManageMtp",
                         "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *query = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *req = context->GetInputShape(REQ_POOL_ENTRIES_INDEX);
    const gert::Shape *cache = context->GetInputShape(CACHE_SLOTS_INDEX);
    const gert::Shape *cacheState = context->GetInputShape(CACHE_STATE_INDEX);
    OPS_LOG_E_IF_NULL(context, query, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, req, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cache, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheState, return ge::GRAPH_FAILED);
    OPS_ERR_IF(query->GetDimNum() != 3 || req->GetDimNum() != 1 ||
                   cache->GetDimNum() != 2,
               OPS_LOG_E(context, "invalid MTP LIM input ranks."),
               return ge::GRAPH_FAILED);

    gert::Shape *topkSlots = context->GetOutputShape(0);
    gert::Shape *topkSource = context->GetOutputShape(1);
    gert::Shape *missSource = context->GetOutputShape(2);
    gert::Shape *missSlots = context->GetOutputShape(3);
    gert::Shape *missCounts = context->GetOutputShape(4);
    gert::Shape *cacheStateOut = context->GetOutputShape(5);
    gert::Shape *cacheOut = context->GetOutputShape(6);
    OPS_LOG_E_IF_NULL(context, topkSlots, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, topkSource, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missSource, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missSlots, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missCounts, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheOut, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheStateOut, return ge::GRAPH_FAILED);

    topkSlots->SetDimNum(3);
    topkSlots->SetDim(0, query->GetDim(0));
    topkSlots->SetDim(1, 1);
    topkSlots->SetDim(2, TOPK);
    *topkSource = *topkSlots;
    missSource->SetDimNum(2);
    missSource->SetDim(0, req->GetDim(0));
    missSource->SetDim(1, UNION_CAPACITY);
    *missSlots = *missSource;
    missCounts->SetDimNum(1);
    missCounts->SetDim(0, req->GetDim(0));
    *cacheStateOut = *cacheState;
    *cacheOut = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNanovllmFusedLiManageMtp(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("NanovllmFusedLiManageMtp",
                         "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    for (uint32_t output = 0; output < 7; ++output) {
        context->SetOutputDataType(output, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NanovllmFusedLiManageMtp)
    .InferShape(InferShapeNanovllmFusedLiManageMtp)
    .InferDataType(InferDataTypeNanovllmFusedLiManageMtp);
} // namespace ops
