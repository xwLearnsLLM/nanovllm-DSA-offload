/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "register/op_def_registry.h"

namespace ops {
class NanovllmFusedLiManageMtp : public OpDef {
public:
    explicit NanovllmFusedLiManageMtp(const char *name) : OpDef(name)
    {
        this->Input("weights").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_dequant_scale").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("block_table").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths_key").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("offload_seq_lengths_key").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("req_valid").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("req_pool_entries").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cache_state").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cache_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("topk_source_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("topk_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("topk_miss_counts").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_source_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_destination_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_counts").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("cache_state_out").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("cache_slots_out").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(NanovllmFusedLiManageMtp);
} // namespace ops
