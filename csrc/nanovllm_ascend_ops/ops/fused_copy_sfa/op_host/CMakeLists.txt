add_ops_compile_options(
        OP_NAME NanovllmFusedCopySfa
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                --op_relocatable_kernel_binary=true
)

# CANN derives the ACLNN adapter from this definition stem.
target_sources(op_host_aclnn PRIVATE
        nanovllm_fused_copy_sfa_def.cpp
)

target_sources(optiling PRIVATE
        nanovllm_fused_copy_sfa_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            nanovllm_fused_copy_sfa_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_tail_attention/op_host
)

target_sources(opsproto PRIVATE
        nanovllm_fused_copy_sfa_proto.cpp
)
