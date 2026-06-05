#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeDsaUpdateIndex(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("DsaUpdateIndex", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape* scoreShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, scoreShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(scoreShape->GetDimNum() != 2,
               OPS_LOG_E(context, "DsaUpdateIndex: score must be rank 2."),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL(context, kPtr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*kPtr <= 0,
               OPS_LOG_E(context, "DsaUpdateIndex: k must be positive, got %ld.", *kPtr),
               return ge::GRAPH_FAILED);

    gert::Shape* promoteShape = context->GetOutputShape(0);
    gert::Shape* demoteShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, promoteShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, demoteShape, return ge::GRAPH_FAILED);

    const int64_t batchSize = scoreShape->GetDim(0);
    promoteShape->SetDimNum(2);
    promoteShape->SetDim(0, batchSize);
    promoteShape->SetDim(1, *kPtr);
    demoteShape->SetDimNum(2);
    demoteShape->SetDim(0, batchSize);
    demoteShape->SetDim(1, *kPtr);

    OPS_LOG_D(context->GetNodeName(), "DsaUpdateIndex InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDsaUpdateIndex(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("DsaUpdateIndex", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaUpdateIndex)
    .InferShape(InferShapeDsaUpdateIndex)
    .InferDataType(InferDataTypeDsaUpdateIndex);
}  // namespace ops
