add_ops_compile_options(
        OP_NAME NanovllmSparseAndTailAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                --op_relocatable_kernel_binary=true
)

# CANN derives aclnn_nanovllm_sparse_and_tail_attention.cpp from this stem.
target_sources(op_host_aclnn PRIVATE
        nanovllm_sparse_and_tail_attention_def.cpp
)

target_sources(optiling PRIVATE
        nanovllm_sparse_and_tail_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            nanovllm_sparse_and_tail_attention_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        nanovllm_sparse_and_tail_attention_proto.cpp
)
