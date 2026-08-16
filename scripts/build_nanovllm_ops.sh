#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
RAW_SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
CUSTOM_OPS="fused_li_manage_mtp"
EXT_BUILD_JOBS="${NANOVLLM_EXT_BUILD_JOBS:-1}"

case "${RAW_SOC_VERSION}" in
  ascend910_93*) CANN_SOC_VERSION="ascend910_93" ;;
  ascend910b*) CANN_SOC_VERSION="ascend910b" ;;
  ascend310*) CANN_SOC_VERSION="ascend310p" ;;
  *) CANN_SOC_VERSION="${RAW_SOC_VERSION}" ;;
esac

export ASCEND_HOME_PATH

echo "[mtp-ops] root: ${ROOT_DIR}"
echo "[mtp-ops] python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[mtp-ops] soc: raw=${RAW_SOC_VERSION}, cann_opp=${CANN_SOC_VERSION}"
echo "[mtp-ops] ascend: ${ASCEND_HOME_PATH}"
echo "[mtp-ops] extension build jobs: ${EXT_BUILD_JOBS}"

echo "[mtp-ops] normalize build script line endings"
find "${ROOT_DIR}/csrc/nanovllm_ascend_ops" -type f \
  \( -name "*.sh" -o -name "*.cmake" -o -name "CMakeLists.txt" \) \
  -exec sed -i 's/\r$//' {} +

if [[ "${NANOVLLM_SKIP_CANN_OPP_BUILD:-0}" == "1" ]]; then
  echo "[mtp-ops] skip CANN custom OPP build"
else
  pushd "${ROOT_DIR}/csrc/nanovllm_ascend_ops/cann_ops" >/dev/null
  rm -rf build output
  bash build.sh -n "${CUSTOM_OPS}" -c "${CANN_SOC_VERSION}"
  rm -rf "${ROOT_DIR}/nanovllm/_cann_ops_custom"
  ./output/CANN-custom_ops*.run \
    --install-path="${ROOT_DIR}/nanovllm/_cann_ops_custom"

  BINARY_INFO_CONFIG="$(
    find "${ROOT_DIR}/nanovllm/_cann_ops_custom" \
      -name binary_info_config.json -print -quit
  )"
  if [[ -z "${BINARY_INFO_CONFIG}" ]] || \
     ! grep -q "NanovllmFusedLiManageMtp" "${BINARY_INFO_CONFIG}"; then
    echo "[mtp-ops] ERROR: NanovllmFusedLiManageMtp is missing from installed OPP." >&2
    exit 1
  fi
  echo "[mtp-ops] verified fused_li_manage_mtp in ${BINARY_INFO_CONFIG}"
  popd >/dev/null
fi

TORCH_NPU_PATH="$(${PYTHON_BIN} - <<'PY'
import os
import torch_npu
print(os.path.dirname(torch_npu.__file__))
PY
)"

rm -rf "${ROOT_DIR}/build/ops_lim_mtp"
cmake -S "${ROOT_DIR}/csrc/nanovllm_ascend_ops" \
  -B "${ROOT_DIR}/build/ops_lim_mtp" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}/nanovllm" \
  -DPYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" \
  -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
  -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"

cmake --build "${ROOT_DIR}/build/ops_lim_mtp" \
  --target install -j"${EXT_BUILD_JOBS}"

EXTENSION="$(
  find "${ROOT_DIR}/build/ops_lim_mtp" -maxdepth 1 -name "_C*.so" -print -quit
)"
if [[ -z "${EXTENSION}" ]]; then
  echo "[mtp-ops] ERROR: built extension _C*.so was not found." >&2
  exit 1
fi
cp -f "${EXTENSION}" "${ROOT_DIR}/nanovllm/"

OPAPI_DIR="${ROOT_DIR}/nanovllm/_cann_ops_custom/vendors/nanovllm-ascend/op_api/lib"
if [[ -f "${OPAPI_DIR}/libcust_opapi.so" ]]; then
  ln -sfn libcust_opapi.so "${OPAPI_DIR}/libopapi.so"
fi

ls -lh "${ROOT_DIR}/nanovllm"/_C*.so
echo "[mtp-ops] built standalone fused_li_manage_mtp and local OPP"
