add_ops_compile_options(
        OP_NAME NanovllmKvcacheScatterCopy
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        kvcache_scatter_copy_def.cpp
)

target_sources(optiling PRIVATE
        kvcache_scatter_copy_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        kvcache_scatter_copy_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        kvcache_scatter_copy_proto.cpp
)
