cmake_minimum_required(VERSION 3.16)
project(ops_lim_mtp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

function(run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

if(NOT DEFINED pybind11_DIR)
  run_python(PYBIND11_CMAKE_DIR
    "import pybind11; print(pybind11.get_cmake_dir())"
    "Failed to locate pybind11 cmake directory")
  set(pybind11_DIR ${PYBIND11_CMAKE_DIR} CACHE PATH "pybind11 cmake directory")
endif()
find_package(pybind11 REQUIRED)

run_python(TORCH_CMAKE_PREFIX
  "import torch; print(torch.utils.cmake_prefix_path)"
  "Failed to locate torch cmake prefix")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

run_python(TORCH_VERSION "import torch; print(torch.__version__)" "Failed to read torch version")
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.9.0")
  message(FATAL_ERROR "Expected PyTorch version 2.9.0, but found ${TORCH_VERSION}")
endif()

if(NOT DEFINED TORCH_NPU_PATH)
  run_python(TORCH_NPU_PATH
    "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    "Failed to locate torch_npu")
endif()

if(NOT DEFINED ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
  elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  elseif(EXISTS "/usr/local/Ascend/latest")
    set(ASCEND_HOME_PATH "/usr/local/Ascend/latest")
  else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set and Ascend toolkit was not found.")
  endif()
endif()

set(NANOVLLM_EXT_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/nanovllm_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/torch_adapter/NPUBridge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/torch_adapter/NPUStorageImpl.cpp)

pybind11_add_module(_C ${NANOVLLM_EXT_SRC})

target_include_directories(_C BEFORE PRIVATE
  ${ASCEND_HOME_PATH}/include
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common/torch_adapter
  ${CMAKE_CURRENT_SOURCE_DIR}/cann_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/fused_copy_sfa_mtp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/fused_li_manage_mtp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/kvcache_scatter_copy
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/sparse_tail_attention_mtp)

target_link_directories(_C PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64)

target_link_libraries(_C PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi)

target_link_options(_C PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/nanovllm-ascend/op_api/lib")

install(TARGETS _C
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .)
