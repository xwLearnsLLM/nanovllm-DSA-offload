npu_op_kernel_sources(ascendc_kernels
    OP_TYPE DsaIndexUpdate
    KERNEL_DIR .
    KERNEL_FILE dsa_index_update.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
