#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
RAW_SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
CUSTOM_OPS="fused_li_manage;fused_li_manage_mtp;kvcache_scatter_copy;sparse_tail_attention;sparse_tail_attention_mtp;fused_copy_sfa;moe_gating_top_k;matmul_allreduce_add_rmsnorm"
NANOVLLM_EXT_BUILD_JOBS="${NANOVLLM_EXT_BUILD_JOBS:-1}"

prepare_catlass() {
  local catlass_root="${ROOT_DIR}/csrc/third_party/catlass"
  local catlass_include="${catlass_root}/include"
  local catlass_header="${catlass_include}/catlass/catlass.hpp"
  if [[ ! -f "${catlass_header}" ]]; then
    echo "[nanovllm ops] catlass headers are missing, fetch ${catlass_root}"
    rm -rf "${catlass_root}"
    git clone --depth 1 https://gitcode.com/cann/catlass.git "${catlass_root}"
  fi
  if [[ ! -f "${catlass_header}" ]]; then
    echo "[nanovllm ops] ERROR: catlass/catlass.hpp was not found in ${catlass_include}" >&2
    exit 1
  fi
  export CPATH="$(cd "${catlass_include}" && pwd):${CPATH:-}"
}

case "${RAW_SOC_VERSION}" in
  ascend910_93*)
    CANN_OPP_SOC_VERSION="ascend910_93"
    ASCENDC_SOC_VERSION="${RAW_SOC_VERSION}"
    ;;
  ascend910b*)
    CANN_OPP_SOC_VERSION="ascend910b"
    ASCENDC_SOC_VERSION="ascend910b"
    ;;
  ascend310*)
    CANN_OPP_SOC_VERSION="ascend310p"
    ASCENDC_SOC_VERSION="ascend310p1"
    ;;
  *)
    CANN_OPP_SOC_VERSION="${RAW_SOC_VERSION}"
    ASCENDC_SOC_VERSION="${RAW_SOC_VERSION}"
    ;;
esac

export ASCEND_HOME_PATH

echo "[nanovllm ops] root: ${ROOT_DIR}"
echo "[nanovllm ops] python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[nanovllm ops] soc: raw=${RAW_SOC_VERSION}, cann_opp=${CANN_OPP_SOC_VERSION}, ascendc=${ASCENDC_SOC_VERSION}"
echo "[nanovllm ops] ascend: ${ASCEND_HOME_PATH}"
echo "[nanovllm ops] extension build jobs: ${NANOVLLM_EXT_BUILD_JOBS}"

if [[ "${CUSTOM_OPS}" == *"matmul_allreduce_add_rmsnorm"* ]]; then
  prepare_catlass
fi

echo "[nanovllm ops] normalize build script line endings"
find "${ROOT_DIR}/csrc/nanovllm_ascend_ops" -type f \
  \( -name "*.sh" -o -name "*.cmake" -o -name "CMakeLists.txt" \) \
  -exec sed -i 's/\r$//' {} +

if [[ "${NANOVLLM_SKIP_CANN_OPP_BUILD:-0}" == "1" ]]; then
  echo "[nanovllm ops] skip CANN custom OPP build"
else
  pushd "${ROOT_DIR}/csrc/nanovllm_ascend_ops/cann_ops" >/dev/null
  rm -rf build output
  bash build.sh -n "${CUSTOM_OPS}" -c "${CANN_OPP_SOC_VERSION}"
  rm -rf "${ROOT_DIR}/nanovllm/_cann_ops_custom"
  ./output/CANN-custom_ops*.run --install-path="${ROOT_DIR}/nanovllm/_cann_ops_custom"

  BINARY_INFO_CONFIG="$(
    find "${ROOT_DIR}/nanovllm/_cann_ops_custom" \
      -name binary_info_config.json -print -quit
  )"
  if [[ -z "${BINARY_INFO_CONFIG}" ]]; then
    echo "[nanovllm ops] ERROR: installed binary_info_config.json was not found." >&2
    exit 1
  fi
  for OP_TYPE in NanovllmFusedLiManage NanovllmFusedLiManageMtp NanovllmKvcacheScatterCopy NanovllmSparseTailAttention NanovllmSparseTailAttentionMtp NanovllmFusedCopySfa; do
    if ! grep -q "${OP_TYPE}" "${BINARY_INFO_CONFIG}"; then
      echo "[nanovllm ops] ERROR: ${OP_TYPE} is missing from ${BINARY_INFO_CONFIG}." >&2
      exit 1
    fi
  done
  echo "[nanovllm ops] verified FUSED_LI_MANAGE/MTP, SCATTER, SPARSE_TAIL_ATTENTION/MTP and FUSED_COPY_SFA/MTP kernels in ${BINARY_INFO_CONFIG}"
  popd >/dev/null
fi

TORCH_NPU_PATH="$(${PYTHON_BIN} - <<'PY'
import os
import torch_npu
print(os.path.dirname(torch_npu.__file__))
PY
)"

rm -rf "${ROOT_DIR}/build/nanovllm_ascend_ops"

cmake -S "${ROOT_DIR}/csrc/nanovllm_ascend_ops" \
  -B "${ROOT_DIR}/build/nanovllm_ascend_ops" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}/nanovllm" \
  -DPYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" \
  -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
  -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  -DSOC_VERSION="${ASCENDC_SOC_VERSION}"

cmake --build "${ROOT_DIR}/build/nanovllm_ascend_ops" --target install -j"${NANOVLLM_EXT_BUILD_JOBS}"

NANOVLLM_EXT="$(
  find "${ROOT_DIR}/build/nanovllm_ascend_ops" -maxdepth 1 -name "_C*.so" -print -quit
)"
if [[ -z "${NANOVLLM_EXT}" ]]; then
  echo "[nanovllm ops] ERROR: built extension _C*.so was not found." >&2
  exit 1
fi
cp -f "${NANOVLLM_EXT}" "${ROOT_DIR}/nanovllm/"

NANOVLLM_KERNEL_LIB="$(
  find "${ROOT_DIR}/build/nanovllm_ascend_ops" -name "libnanovllm_ascend_kernels.so" -print -quit
)"
if [[ -z "${NANOVLLM_KERNEL_LIB}" ]]; then
  echo "[nanovllm ops] ERROR: built kernel libnanovllm_ascend_kernels.so was not found." >&2
  exit 1
fi
cp -f "${NANOVLLM_KERNEL_LIB}" "${ROOT_DIR}/nanovllm/"

ls -lh "${ROOT_DIR}/nanovllm"/_C*.so "${ROOT_DIR}/nanovllm/libnanovllm_ascend_kernels.so"

ln -s ${ROOT_DIR}/nanovllm/_cann_ops_custom/vendors/nanovllm-ascend/op_api/lib/libcust_opapi.so ${ROOT_DIR}/nanovllm/_cann_ops_custom/vendors/nanovllm-ascend/op_api/lib/libopapi.so

echo "[nanovllm ops] built nanovllm/_C*.so and nanovllm/_cann_ops_custom/"
