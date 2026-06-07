/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_SELECTION_K_ROPE = 0;
constexpr size_t INPUT_IDX_SELECTION_KV_CACHE = 1;
constexpr size_t INPUT_IDX_SELECTION_KV_BLOCK_STATUS = 3;
const int32_t INDEX_INPUT_0 = 0;
const int32_t INDEX_INPUT_1 = 1;
const int32_t INDEX_INPUT_3 = 3;
const int32_t INDEX_OUTPUT_0 = 0;
const int32_t INDEX_OUTPUT_1 = 1;
const int32_t INDEX_OUTPUT_2 = 2;

static ge::graphStatus InferShape4GatherSelectionKvCache(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4GatherSelectionKvCache.");

    const gert::Shape* selectionKRopeShape = context->GetInputShape(INDEX_INPUT_0);
    OPS_LOG_E_IF_NULL(context, selectionKRopeShape, return ge::GRAPH_FAILED);
    auto selectionKRopeInplaceShape = context->GetOutputShape(INDEX_OUTPUT_0);
    *selectionKRopeInplaceShape = *selectionKRopeShape;

    const gert::Shape* selectionKvCacheShape = context->GetInputShape(INDEX_INPUT_1);
    OPS_LOG_E_IF_NULL(context, selectionKvCacheShape, return ge::GRAPH_FAILED);
    auto selectionKvCacheInplaceShape = context->GetOutputShape(INDEX_OUTPUT_1);
    *selectionKvCacheInplaceShape = *selectionKvCacheShape;

    const gert::Shape* selectionKvBlockStatusShape = context->GetInputShape(INDEX_INPUT_3);
    OPS_LOG_E_IF_NULL(context, selectionKvBlockStatusShape, return ge::GRAPH_FAILED);
    auto selectionKvBlockStatusInplaceShape = context->GetOutputShape(INDEX_OUTPUT_2);
    *selectionKvBlockStatusInplaceShape = *selectionKvBlockStatusShape;

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4GatherSelectionKvCache");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GatherSelectionKvCache(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherSelectionKvCache enter");

    const auto selectionKRopeDtype = context->GetInputDataType(INPUT_IDX_SELECTION_K_ROPE);
    const auto selectionKvCacheDtype = context->GetInputDataType(INPUT_IDX_SELECTION_KV_CACHE);
    const auto selectionKvBlockStatusDtype = context->GetInputDataType(INPUT_IDX_SELECTION_KV_BLOCK_STATUS);
    context->SetOutputDataType(INDEX_OUTPUT_0, selectionKRopeDtype);
    context->SetOutputDataType(INDEX_OUTPUT_1, selectionKvCacheDtype);
    context->SetOutputDataType(INDEX_OUTPUT_2, selectionKvBlockStatusDtype);

    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherSelectionKvCache end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherSelectionKvCache)
    .InferShape(InferShape4GatherSelectionKvCache)
    .InferDataType(InferDtype4GatherSelectionKvCache);
}  // namespace ops
