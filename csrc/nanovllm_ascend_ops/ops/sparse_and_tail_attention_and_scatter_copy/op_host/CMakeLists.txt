add_ops_compile_options(
        OP_NAME NanovllmSparseAndTailAttentionAndScatterCopy
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                --op_relocatable_kernel_binary=true
)

# CANN derives the ACLNN adapter from this definition stem.
target_sources(op_host_aclnn PRIVATE
        nanovllm_sparse_and_tail_attention_and_scatter_copy_def.cpp
)

target_sources(optiling PRIVATE
        nanovllm_sparse_and_tail_attention_and_scatter_copy_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            nanovllm_sparse_and_tail_attention_and_scatter_copy_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_and_tail_attention/op_host
)

target_sources(opsproto PRIVATE
        nanovllm_sparse_and_tail_attention_and_scatter_copy_proto.cpp
)
