/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr int32_t INPUT_DRAM_KV_CACHE = 1;

static ge::graphStatus InferShape4KvcacheOffloadCopy(gert::InferShapeContext* context)
{
    const gert::Shape* dramKvShape = context->GetInputShape(INPUT_DRAM_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, dramKvShape, return ge::GRAPH_FAILED);
    *context->GetOutputShape(0) = *dramKvShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4KvcacheOffloadCopy(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(INPUT_DRAM_KV_CACHE));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NanovllmKvcacheOffloadCopy)
    .InferShape(InferShape4KvcacheOffloadCopy)
    .InferDataType(InferDtype4KvcacheOffloadCopy);
} // namespace ops
