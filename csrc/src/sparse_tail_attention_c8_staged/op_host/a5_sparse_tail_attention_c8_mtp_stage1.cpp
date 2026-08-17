/**
 * Repository-local C8 QSFA state operator.
 *
 * The compute and tiling implementation is vendored from CANN
 * ops-transformer 9.1.0.  Unlike the upstream normalized-output ABI, this
 * operator writes the unnormalized FP32 flash-attention state P/M/L.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t QUERY = 0;
constexpr int64_t ROPE_DIM = 64;
}

namespace ops {
static ge::graphStatus InferStateShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *query = context->GetInputShape(QUERY);
    if (query->GetDimNum() != 3 || query->GetDim(2) < ROPE_DIM) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape(
        {query->GetDim(0), query->GetDim(1), query->GetDim(2) - ROPE_DIM});
    *context->GetOutputShape(1) =
        gert::Shape({1, query->GetDim(0), query->GetDim(1)});
    *context->GetOutputShape(2) = *context->GetOutputShape(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferStateDtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferAttentionShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *query = context->GetInputShape(QUERY);
    if (query->GetDimNum() != 3 || query->GetDim(2) < ROPE_DIM) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape(
        {query->GetDim(0), query->GetDim(1), query->GetDim(2) - ROPE_DIM});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferAttentionDtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(QUERY));
    return ge::GRAPH_SUCCESS;
}

#ifndef A5_C8_STAGE2_HOST_ONLY
class A5SparseTailAttentionC8MtpStage1 : public OpDef {
public:
    explicit A5SparseTailAttentionC8MtpStage1(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> baseQueryTypes = {
            ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> baseKvTypes = {
            ge::DT_INT8, ge::DT_INT8};
        const std::vector<ge::DataType> baseIntTypes = {
            ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::DataType> baseFp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> baseFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataType(baseQueryTypes).Format(baseFormats);
        this->Input("key").ParamType(REQUIRED).DataType(baseKvTypes).Format(baseFormats);
        this->Input("value").ParamType(REQUIRED).Follow("key");
        this->Input("sparse_indices").ParamType(REQUIRED).DataType(baseIntTypes).Format(baseFormats);
        this->Input("key_dequant_scale").ParamType(OPTIONAL).DataType(baseFp32Types).Format(baseFormats);
        this->Input("value_dequant_scale").ParamType(OPTIONAL).DataType(baseFp32Types).Format(baseFormats);
        this->Input("block_table").ParamType(OPTIONAL).DataType(baseIntTypes).Format(baseFormats);
        this->Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType(baseIntTypes).Format(baseFormats);
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataType(baseIntTypes).Format(baseFormats);
        this->Input("miss_counts").ParamType(REQUIRED).DataType(baseIntTypes).Format(baseFormats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(baseIntTypes).Format(baseFormats);
        this->Output("partial_out").ParamType(REQUIRED).DataType(baseFp32Types).Format(baseFormats);
        this->Output("softmax_max").ParamType(REQUIRED).DataType(baseFp32Types).Format(baseFormats);
        this->Output("softmax_sum").ParamType(REQUIRED).DataType(baseFp32Types).Format(baseFormats);

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(2);
        this->Attr("value_quant_mode").AttrType(REQUIRED).Int(2);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128);
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int(64);
        this->Attr("stage_mode").AttrType(OPTIONAL).Int(1);

        const std::vector<ge::DataType> a5QueryTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> a5KvTypes = {
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8,
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
        const std::vector<ge::DataType> a5Fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        OpAICoreConfig config;
        config.Input("query")
            .ParamType(REQUIRED)
            .DataType(a5QueryTypes)
            .FormatList({ge::FORMAT_ND});
        config.Input("key")
            .ParamType(REQUIRED)
            .DataType(a5KvTypes)
            .FormatList({ge::FORMAT_ND});
        config.Input("value")
            .ParamType(REQUIRED)
            .DataType(a5KvTypes)
            .FormatList({ge::FORMAT_ND});
        config.Input("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("key_dequant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        config.Input("value_dequant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        config.Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("cache_tokens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Output("partial_out").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.Output("softmax_max").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.Output("softmax_sum").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
#endif

#ifdef A5_C8_STAGE2_HOST_ONLY
class A5SparseTailAttentionC8MtpStage2 : public OpDef {
public:
    explicit A5SparseTailAttentionC8MtpStage2(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> queryTypes = {
            ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> kvTypes = {
            ge::DT_INT8, ge::DT_INT8};
        const std::vector<ge::DataType> intTypes = {
            ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataType(queryTypes).Format(formats);
        this->Input("key").ParamType(REQUIRED).DataType(kvTypes).Format(formats);
        this->Input("value").ParamType(REQUIRED).Follow("key");
        this->Input("sparse_indices").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("key_dequant_scale").ParamType(OPTIONAL).DataType(fp32Types).Format(formats);
        this->Input("value_dequant_scale").ParamType(OPTIONAL).DataType(fp32Types).Format(formats);
        this->Input("block_table").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("miss_counts").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("previous_p").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Input("previous_m").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Input("previous_l").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Output("attention_out").ParamType(REQUIRED).DataType(queryTypes).Format(formats);

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(2);
        this->Attr("value_quant_mode").AttrType(REQUIRED).Int(2);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128);
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int(64);
        this->Attr("stage_mode").AttrType(OPTIONAL).Int(2);

        const std::vector<ge::DataType> a5QueryTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> a5KvTypes = {
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8,
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
        const std::vector<ge::DataType> a5Fp32Types(6, ge::DT_FLOAT);
        OpAICoreConfig config;
        config.Input("query").ParamType(REQUIRED).DataType(a5QueryTypes).FormatList({ge::FORMAT_ND});
        config.Input("key").ParamType(REQUIRED).DataType(a5KvTypes).FormatList({ge::FORMAT_ND});
        config.Input("value").ParamType(REQUIRED).DataType(a5KvTypes).FormatList({ge::FORMAT_ND});
        config.Input("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("key_dequant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        config.Input("value_dequant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        config.Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("cache_tokens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        config.Input("previous_p").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.Input("previous_m").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.Input("previous_l").ParamType(REQUIRED).DataType(a5Fp32Types).FormatList({ge::FORMAT_ND});
        config.Output("attention_out").ParamType(REQUIRED).DataType(a5QueryTypes).FormatList({ge::FORMAT_ND});
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
#endif

#ifndef A5_C8_STAGE2_HOST_ONLY
OP_ADD(A5SparseTailAttentionC8MtpStage1);
IMPL_OP_INFERSHAPE(A5SparseTailAttentionC8MtpStage1)
    .InferShape(InferStateShape)
    .InferDataType(InferStateDtype);
#else
OP_ADD(A5SparseTailAttentionC8MtpStage2);
IMPL_OP_INFERSHAPE(A5SparseTailAttentionC8MtpStage2)
    .InferShape(InferAttentionShape)
    .InferDataType(InferAttentionDtype);
#endif
} // namespace ops
