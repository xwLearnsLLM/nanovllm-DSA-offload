add_ops_compile_options(
        OP_NAME NanovllmSparseTailAttentionMtp
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        nanovllm_sparse_tail_attention_mtp_def.cpp
)

target_sources(opsproto PRIVATE
        nanovllm_sparse_tail_attention_mtp_proto.cpp
)
