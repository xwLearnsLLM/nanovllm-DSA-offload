/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef TEMPLATE_TILING_KEY_LI_C8_DECODE_UPDATE_H_
#define TEMPLATE_TILING_KEY_LI_C8_DECODE_UPDATE_H_

#include "ascendc/host_api/tiling/template_argument.h"

// msopgen 的 json 类型表不支持 float8_e4m3fn，ops.json 中以 uint8 顶替
// （fp8 与 uint8 同为 1 字节存储）。DT 模板值取 ge::DT_UINT8(=4)，
// 运行时真实的 fp8 dtype 由 OpDef(ge::DT_FLOAT8_E4M3FN) 声明校验，
// kernel 内部按 fp8 语义处理数据。
#define LI_C8_TPL_UINT8 4

ASCENDC_TPL_ARGS_DECL(A5FusedLiManageC8,
                      ASCENDC_TPL_DTYPE_DECL(DT, LI_C8_TPL_UINT8));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_C8_TPL_UINT8)), );

#endif
