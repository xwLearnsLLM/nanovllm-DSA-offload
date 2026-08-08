add_ops_compile_options(
        OP_NAME NanovllmFusedLiManage
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

# CANN derives aclnnInner_nanovllm_fused_li_manage.cpp from this stem.
target_sources(op_host_aclnnInner PRIVATE
        nanovllm_fused_li_manage_def.cpp
)

target_sources(optiling PRIVATE
        fused_li_manage_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        fused_li_manage_infershape.cpp
)

target_sources(opapi PRIVATE
        op_api/aclnn_fused_li_manage.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
)

install(FILES op_api/aclnn_fused_li_manage.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)

