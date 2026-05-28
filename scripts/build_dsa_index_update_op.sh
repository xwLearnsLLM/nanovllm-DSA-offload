#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
RAW_SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
JOBS="${NANOVLLM_DSA_INDEX_UPDATE_BUILD_JOBS:-$(nproc)}"

case "${RAW_SOC_VERSION}" in
  ascend910_93*)
    CANN_OPP_SOC_VERSION="ascend910_93"
    ;;
  ascend910b*)
    CANN_OPP_SOC_VERSION="ascend910b"
    ;;
  *)
    CANN_OPP_SOC_VERSION="${RAW_SOC_VERSION}"
    ;;
esac

export ASCEND_HOME_PATH

OP_ROOT="${ROOT_DIR}/dsa_index_update_op"
CANN_ROOT="${OP_ROOT}/cann"
INSTALL_ROOT="${ROOT_DIR}/nanovllm/_dsa_index_update_custom"
VENDOR_DIR="${INSTALL_ROOT}/vendors/dsa_index_update_custom"

echo "[dsa_index_update] root: ${ROOT_DIR}"
echo "[dsa_index_update] python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[dsa_index_update] soc: raw=${RAW_SOC_VERSION}, cann_opp=${CANN_OPP_SOC_VERSION}"
echo "[dsa_index_update] ascend: ${ASCEND_HOME_PATH}"

echo "[dsa_index_update] normalize line endings"
find "${OP_ROOT}" -type f \
  \( -name "*.sh" -o -name "*.cmake" -o -name "CMakeLists.txt" -o -name "*.cpp" -o -name "*.h" \) \
  -exec sed -i 's/\r$//' {} +

echo "[dsa_index_update] source markers"
grep -n "KERNEL_TYPE_AIV_ONLY\|GetTaskRation" \
  "${OP_ROOT}/cann/op_kernel/dsa_index_update.cpp" || true
grep -n "manual_acl_tensor_aiv_only_v4_task_ratio\|DSA_INDEX_UPDATE_CUST_OPAPI_PATH\|AclTensorGuard\|aclCreateTensor\|EXEC_NPU_CMD(aclnnDsaIndexUpdate" \
  "${OP_ROOT}/torch_extension/dsa_index_update_ext.cpp" || true
grep -n "DSA_INDEX_UPDATE_CUST_OPAPI_PATH" \
  "${OP_ROOT}/torch_extension/CMakeLists.txt" || true
grep -n "manual_acl_tensor_aiv_only_v4_task_ratio" \
  "${ROOT_DIR}/nanovllm/models/dsa_index_update_real.py" || true
if ! grep -q "KERNEL_TYPE_AIV_ONLY" \
    "${OP_ROOT}/cann/op_kernel/dsa_index_update.cpp"; then
  echo "[dsa_index_update] ERROR: AIV-only kernel marker is missing." >&2
  exit 1
fi
if ! grep -q "GetTaskRation" \
    "${OP_ROOT}/cann/op_kernel/dsa_index_update.cpp"; then
  echo "[dsa_index_update] ERROR: logical block-id task-ratio marker is missing." >&2
  exit 1
fi
if ! grep -q "manual_acl_tensor_aiv_only_v4_task_ratio" \
    "${OP_ROOT}/torch_extension/dsa_index_update_ext.cpp"; then
  echo "[dsa_index_update] ERROR: binding version marker is missing." >&2
  exit 1
fi
if ! grep -q "manual_acl_tensor_aiv_only_v4_task_ratio" \
    "${ROOT_DIR}/nanovllm/models/dsa_index_update_real.py"; then
  echo "[dsa_index_update] ERROR: Python binding-version guard is stale." >&2
  exit 1
fi
if ! grep -q "DSA_INDEX_UPDATE_CUST_OPAPI_PATH" \
    "${OP_ROOT}/torch_extension/CMakeLists.txt"; then
  echo "[dsa_index_update] ERROR: direct custom opapi path definition is missing." >&2
  exit 1
fi
if grep -q "EXEC_NPU_CMD(aclnnDsaIndexUpdate" \
    "${OP_ROOT}/torch_extension/dsa_index_update_ext.cpp"; then
  echo "[dsa_index_update] ERROR: stale EXEC_NPU_CMD binding is still present." >&2
  exit 1
fi

echo "[dsa_index_update] clean previous build artifacts"
rm -rf "${ROOT_DIR}/build/dsa_index_update_ext"
rm -f "${ROOT_DIR}/nanovllm"/_dsa_index_update_C*.so

if [[ "${NANOVLLM_SKIP_DSA_INDEX_UPDATE_CANN_BUILD:-0}" == "1" ]]; then
  echo "[dsa_index_update] skip CANN custom OPP build"
else
  pushd "${CANN_ROOT}" >/dev/null
  rm -rf build output
  bash build.sh --soc="${CANN_OPP_SOC_VERSION}" -j"${JOBS}"
  popd >/dev/null

  PKG_PATH="$(
    find "${CANN_ROOT}/build" -maxdepth 1 -type f -name "custom_opp_*_aarch64.run" -print -quit
  )"
  if [[ -z "${PKG_PATH}" ]]; then
    echo "[dsa_index_update] ERROR: custom OPP package was not found." >&2
    exit 1
  fi

  rm -rf "${INSTALL_ROOT}"
  "${PKG_PATH}" --install-path="${INSTALL_ROOT}"
fi

if [[ ! -d "${VENDOR_DIR}/op_api/lib" ]]; then
  echo "[dsa_index_update] ERROR: ${VENDOR_DIR}/op_api/lib does not exist." >&2
  exit 1
fi

TORCH_NPU_PATH="$(${PYTHON_BIN} - <<'PY'
import os
import torch_npu
print(os.path.dirname(torch_npu.__file__))
PY
)"

cmake -S "${OP_ROOT}/torch_extension" \
  -B "${ROOT_DIR}/build/dsa_index_update_ext" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}/nanovllm" \
  -DPYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" \
  -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
  -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  -DDSA_INDEX_UPDATE_VENDOR_DIR="${VENDOR_DIR}"

cmake --build "${ROOT_DIR}/build/dsa_index_update_ext" --target install -j"${JOBS}"

DSA_EXT="$(
  find "${ROOT_DIR}/build/dsa_index_update_ext" -maxdepth 1 -name "_dsa_index_update_C*.so" -print -quit
)"
if [[ -n "${DSA_EXT}" ]]; then
  cp -f "${DSA_EXT}" "${ROOT_DIR}/nanovllm/"
fi

ls -lh "${ROOT_DIR}/nanovllm"/_dsa_index_update_C*.so
echo "[dsa_index_update] built nanovllm/_dsa_index_update_C*.so and nanovllm/_dsa_index_update_custom/"
