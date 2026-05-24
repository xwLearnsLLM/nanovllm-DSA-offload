#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
SOC_VERSION="${SOC_VERSION:-ascend910_93}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
CUSTOM_OPS="lightning_indexer_vllm;sparse_flash_attention;moe_gating_top_k"

export ASCEND_HOME_PATH

echo "[nanovllm ops] root: ${ROOT_DIR}"
echo "[nanovllm ops] python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[nanovllm ops] soc: ${SOC_VERSION}"
echo "[nanovllm ops] ascend: ${ASCEND_HOME_PATH}"

pushd "${ROOT_DIR}/csrc/nanovllm_ascend_ops/cann_ops" >/dev/null
rm -rf build output
bash build.sh -n "${CUSTOM_OPS}" -c "${SOC_VERSION}"
rm -rf "${ROOT_DIR}/nanovllm/_cann_ops_custom"
./output/CANN-custom_ops*.run --install-path="${ROOT_DIR}/nanovllm/_cann_ops_custom"
popd >/dev/null

TORCH_NPU_PATH="$(${PYTHON_BIN} - <<'PY'
import os
import torch_npu
print(os.path.dirname(torch_npu.__file__))
PY
)"

cmake -S "${ROOT_DIR}/csrc/nanovllm_ascend_ops" \
  -B "${ROOT_DIR}/build/nanovllm_ascend_ops" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}/nanovllm" \
  -DPYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" \
  -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
  -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  -DSOC_VERSION="${SOC_VERSION}"

cmake --build "${ROOT_DIR}/build/nanovllm_ascend_ops" --target install -j"$(nproc)"

echo "[nanovllm ops] built nanovllm/_C*.so and nanovllm/_cann_ops_custom/"
