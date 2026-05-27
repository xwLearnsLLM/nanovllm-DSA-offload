cmake_minimum_required(VERSION 3.16)
project(nanovllm_dsa_index_update_ext)

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
  else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set and Ascend toolkit was not found.")
  endif()
endif()

if(NOT DEFINED DSA_INDEX_UPDATE_VENDOR_DIR)
  message(FATAL_ERROR "DSA_INDEX_UPDATE_VENDOR_DIR must point to vendors/dsa_index_update_custom.")
endif()

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(ADAPTER_DIR ${ROOT_DIR}/csrc/nanovllm_ascend_ops/aclnn_torch_adapter)

set(EXT_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/dsa_index_update_ext.cpp
  ${ADAPTER_DIR}/NPUBridge.cpp
  ${ADAPTER_DIR}/NPUStorageImpl.cpp)

pybind11_add_module(_dsa_index_update_C ${EXT_SRC})

target_include_directories(_dsa_index_update_C BEFORE PRIVATE
  ${ASCEND_HOME_PATH}/include
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ROOT_DIR}/csrc/nanovllm_ascend_ops
  ${ADAPTER_DIR})

target_link_directories(_dsa_index_update_C PRIVATE
  ${DSA_INDEX_UPDATE_VENDOR_DIR}/op_api/lib
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64)

target_link_libraries(_dsa_index_update_C PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  dl
  opapi)

target_link_options(_dsa_index_update_C PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/_dsa_index_update_custom/vendors/dsa_index_update_custom/op_api/lib")

install(TARGETS _dsa_index_update_C
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .)
