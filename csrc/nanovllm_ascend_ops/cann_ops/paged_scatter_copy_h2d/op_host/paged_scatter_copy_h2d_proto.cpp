#include "error/ops_error.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapePagedScatterCopyH2d(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PagedScatterCopyH2d", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr,
               OPS_LOG_E("PagedScatterCopyH2d", "Output shape is nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePagedScatterCopyH2d(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PagedScatterCopyH2d", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PagedScatterCopyH2d)
    .InferShape(InferShapePagedScatterCopyH2d)
    .InferDataType(InferDataTypePagedScatterCopyH2d);
} // namespace ops
