#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int64_t kRopeDim = 64;

ge::graphStatus InferStage1Shape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *query = context->GetInputShape(0);
    if (query->GetDimNum() != 3 || query->GetDim(2) < kRopeDim) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape(
        {query->GetDim(0), query->GetDim(1), query->GetDim(2) - kRopeDim});
    *context->GetOutputShape(1) =
        gert::Shape({1, query->GetDim(0), query->GetDim(1)});
    *context->GetOutputShape(2) = *context->GetOutputShape(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferStage1Dtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferStage2Shape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *query = context->GetInputShape(0);
    if (query->GetDimNum() != 3 || query->GetDim(2) < kRopeDim) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape(
        {query->GetDim(0), query->GetDim(1), query->GetDim(2) - kRopeDim});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferStage2Dtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

void AddCommonAttrs(OpDef &op)
{
    op.Attr("scale_value").AttrType(REQUIRED).Float(1.0);
    op.Attr("key_quant_mode").AttrType(REQUIRED).Int(2);
    op.Attr("value_quant_mode").AttrType(REQUIRED).Int(2);
    op.Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
    op.Attr("layout_query").AttrType(OPTIONAL).String("TND");
    op.Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
    op.Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
    op.Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    op.Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    op.Attr("attention_mode").AttrType(OPTIONAL).Int(2);
    op.Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(1);
    op.Attr("tile_size").AttrType(OPTIONAL).Int(128);
    op.Attr("rope_head_dim").AttrType(OPTIONAL).Int(kRopeDim);
    op.Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
}

void AddCommonInputs(OpDef &op)
{
    const std::vector<ge::DataType> q = {ge::DT_BF16, ge::DT_FLOAT16};
    const std::vector<ge::DataType> kv = {ge::DT_INT8, ge::DT_INT8};
    const std::vector<ge::DataType> i32 = {ge::DT_INT32, ge::DT_INT32};
    const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
    const std::vector<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
    op.Input("query").ParamType(REQUIRED).DataType(q).Format(nd).AutoContiguous();
    op.Input("key").ParamType(REQUIRED).DataType(kv).Format(nd).AutoContiguous();
    op.Input("value").ParamType(REQUIRED).Follow("key").AutoContiguous();
    op.Input("sparse_indices").ParamType(REQUIRED).DataType(i32).Format(nd).AutoContiguous();
    op.Input("key_dequant_scale").ParamType(OPTIONAL).DataType(fp32).Format(nd).AutoContiguous();
    op.Input("value_dequant_scale").ParamType(OPTIONAL).DataType(fp32).Format(nd).AutoContiguous();
    op.Input("block_table").ParamType(OPTIONAL).DataType(i32).Format(nd).AutoContiguous();
    op.Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType(i32).Format(nd).AutoContiguous();
    op.Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataType(i32).Format(nd).AutoContiguous();
    op.Input("miss_counts").ParamType(REQUIRED).DataType(i32).Format(nd).AutoContiguous();
}

void AddA5Config(OpDef &op, bool stage1)
{
    const std::vector<ge::DataType> q = {
        ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
    const std::vector<ge::DataType> kv = {
        ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8,
        ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
    const std::vector<ge::DataType> fp32(6, ge::DT_FLOAT);
    OpAICoreConfig config;
    config.Input("query").ParamType(REQUIRED).DataType(q).FormatList({ge::FORMAT_ND});
    config.Input("key").ParamType(REQUIRED).DataType(kv).FormatList({ge::FORMAT_ND});
    config.Input("value").ParamType(REQUIRED).DataType(kv).FormatList({ge::FORMAT_ND});
    config.Input("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    config.Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    config.Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    config.Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    config.Input("miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    for (const char *name : {"key_dequant_scale", "value_dequant_scale"}) {
        config.Input(name).ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    }
    if (stage1) {
        config.Output("partial_out").ParamType(REQUIRED).DataType(fp32).FormatList({ge::FORMAT_ND});
        config.Output("softmax_max").ParamType(REQUIRED).DataType(fp32).FormatList({ge::FORMAT_ND});
        config.Output("softmax_sum").ParamType(REQUIRED).DataType(fp32).FormatList({ge::FORMAT_ND});
    } else {
        for (const char *name : {"previous_p", "previous_m", "previous_l"}) {
            config.Input(name).ParamType(REQUIRED).DataType(fp32).FormatList({ge::FORMAT_ND});
        }
        config.Output("attention_out").ParamType(REQUIRED).DataType(q).FormatList({ge::FORMAT_ND});
    }
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    op.AICore().AddConfig("ascend950", config);
}
} // namespace

class A5SparseTailAttentionC8Stage1 : public OpDef {
public:
    explicit A5SparseTailAttentionC8Stage1(const char *name) : OpDef(name)
    {
        AddCommonInputs(*this);
        const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Output("partial_out").ParamType(REQUIRED).DataType(fp32).Format(nd);
        this->Output("softmax_max").ParamType(REQUIRED).DataType(fp32).Format(nd);
        this->Output("softmax_sum").ParamType(REQUIRED).DataType(fp32).Format(nd);
        AddCommonAttrs(*this);
        AddA5Config(*this, true);
    }
};

class A5SparseTailAttentionC8Stage2 : public OpDef {
public:
    explicit A5SparseTailAttentionC8Stage2(const char *name) : OpDef(name)
    {
        AddCommonInputs(*this);
        const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::DataType> q = {ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("previous_p").ParamType(REQUIRED).DataType(fp32).Format(nd).AutoContiguous();
        this->Input("previous_m").ParamType(REQUIRED).DataType(fp32).Format(nd).AutoContiguous();
        this->Input("previous_l").ParamType(REQUIRED).DataType(fp32).Format(nd).AutoContiguous();
        this->Output("attention_out").ParamType(REQUIRED).DataType(q).Format(nd);
        AddCommonAttrs(*this);
        AddA5Config(*this, false);
    }
};

OP_ADD(A5SparseTailAttentionC8Stage1);
IMPL_OP_INFERSHAPE(A5SparseTailAttentionC8Stage1)
    .InferShape(InferStage1Shape).InferDataType(InferStage1Dtype);
OP_ADD(A5SparseTailAttentionC8Stage2);
IMPL_OP_INFERSHAPE(A5SparseTailAttentionC8Stage2)
    .InferShape(InferStage2Shape).InferDataType(InferStage2Dtype);
} // namespace ops
