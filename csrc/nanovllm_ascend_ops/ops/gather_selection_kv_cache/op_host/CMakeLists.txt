# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.

add_ops_compile_options(
        OP_NAME GatherSelectionKvCache
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(gather_selection_kv_cache_depends transformer/attention/gather_selection_kv_cache PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        gather_selection_kv_cache_def.cpp
)

target_sources(optiling PRIVATE
        gather_selection_kv_cache_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            gather_selection_kv_cache_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        gather_selection_kv_cache_proto.cpp
)
