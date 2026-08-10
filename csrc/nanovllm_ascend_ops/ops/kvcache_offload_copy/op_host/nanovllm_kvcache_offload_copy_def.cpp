/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "register/op_def_registry.h"

namespace ops {
class NanovllmKvcacheOffloadCopy : public OpDef {
public:
    explicit NanovllmKvcacheOffloadCopy(const char* name) : OpDef(name)
    {
        this->Input("hbm_kv_cache").ParamType(REQUIRED).DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_kv_cache").ParamType(REQUIRED).DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("hbm_block_table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_block_table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("copy_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();

        // Reusing the input name makes the generated ACLNN interface in-place.
        this->Output("dram_kv_cache").ParamType(REQUIRED).DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(NanovllmKvcacheOffloadCopy);
} // namespace ops
