#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

namespace ops {

static ge::graphStatus InferShape4DsaIndexUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* scoreShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);
    OP_CHECK_IF(scoreShape->GetDimNum() != 2,
        OP_LOGE(context, "DsaIndexUpdate: score must be rank 2."),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    OP_CHECK_IF(*kPtr <= 0,
        OP_LOGE(context, "DsaIndexUpdate: max_copy_tokens must be positive, got %ld.", *kPtr),
        return ge::GRAPH_FAILED);

    gert::Shape* promoteShape = context->GetOutputShape(0);
    gert::Shape* demoteShape = context->GetOutputShape(1);
    gert::Shape* copyCountsShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, promoteShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, demoteShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, copyCountsShape);

    const int64_t batchSize = scoreShape->GetDim(0);
    promoteShape->SetDimNum(2);
    promoteShape->SetDim(0, batchSize);
    promoteShape->SetDim(1, *kPtr);
    demoteShape->SetDimNum(2);
    demoteShape->SetDim(0, batchSize);
    demoteShape->SetDim(1, *kPtr);
    copyCountsShape->SetDimNum(1);
    copyCountsShape->SetDim(0, batchSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaIndexUpdate).InferShape(InferShape4DsaIndexUpdate);

} // namespace ops
