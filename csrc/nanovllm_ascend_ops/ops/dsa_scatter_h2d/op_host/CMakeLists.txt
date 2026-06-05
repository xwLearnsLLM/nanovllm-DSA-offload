add_ops_compile_options(
        OP_NAME PagedScatterCopyH2d
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
)

target_sources(op_host_aclnn PRIVATE
        paged_scatter_copy_h2d_def.cpp
)

target_sources(optiling PRIVATE
        paged_scatter_copy_h2d_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            paged_scatter_copy_h2d_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        paged_scatter_copy_h2d_proto.cpp
)
