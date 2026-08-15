#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_SPEC="${ROOT}/csrc/ops.json"
SOURCE_ROOT="${ROOT}/csrc/src"
BUILD_ROOT="${ROOT}/build/bf16"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${ROOT}/_custom_opp_bf16"
TORCH_EXTENSION="${ROOT}/torch_extension"
PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"
BUILD_JOBS="${NANOVLLM_A5_OPS_BUILD_JOBS:-$(nproc)}"

SOC_RAW="${A5_SOC_VERSION:-${SOC_VERSION:-ascend950}}"
case "${SOC_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*) SOC="ascend950" ;;
    *) echo "[bf16_build] ERROR: only Ascend 950 is supported; got ${SOC_RAW}." >&2; exit 2 ;;
esac
command -v msopgen >/dev/null 2>&1 || {
    echo "[bf16_build] ERROR: source the CANN 9.1 environment first." >&2
    exit 2
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[bf16_build] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
}

OP_NAMES=(
    A5FusedLiManage
    A5FusedLiManageMtp
    A5KvcacheScatterCopy
    A5SparseTailAttention
    A5FusedCopySparseTailAttention
)
OP_DIRS=(
    fused_li_manage
    fused_li_manage_mtp
    kvcache_scatter_copy
    sparse_tail_attention
    fused_copy_sparse_tail_attention
)
for op_dir in "${OP_DIRS[@]}"; do
    [[ -d "${SOURCE_ROOT}/${op_dir}" ]] || {
        echo "[bf16_build] ERROR: missing ${SOURCE_ROOT}/${op_dir}." >&2
        exit 2
    }
done

echo "[bf16_build] target=${SOC} jobs=${BUILD_JOBS} output=${LOCAL_OPP}"
rm -rf "${GENERATED}"
mkdir -p "${BUILD_ROOT}"
for index in "${!OP_NAMES[@]}"; do
    args=(gen -i "${OP_SPEC}" -f aclnn -c "ai_core-${SOC}" -lan cpp -op "${OP_NAMES[index]}" -out "${GENERATED}")
    if (( index > 0 )); then
        args+=(-m 1)
    fi
    msopgen "${args[@]}"
done

for op_dir in common "${OP_DIRS[@]}"; do
    [[ ! -d "${SOURCE_ROOT}/${op_dir}/op_host" ]] || cp -a "${SOURCE_ROOT}/${op_dir}/op_host/." "${GENERATED}/op_host/"
    [[ ! -d "${SOURCE_ROOT}/${op_dir}/op_kernel" ]] || cp -a "${SOURCE_ROOT}/${op_dir}/op_kernel/." "${GENERATED}/op_kernel/"
done
rm -f "${GENERATED}/op_host/a5_fused_li_manage_def.cpp"

pushd "${GENERATED}" >/dev/null
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
popd >/dev/null
RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
[[ -n "${RUN_PKG}" ]] || { echo "[bf16_build] ERROR: no .run package." >&2; exit 1; }

rm -rf "${LOCAL_OPP}"
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"
mapfile -t OPAPI_LIBS < <(find "${LOCAL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so')
[[ "${#OPAPI_LIBS[@]}" -eq 1 ]] || {
    echo "[bf16_build] ERROR: expected one libcust_opapi.so." >&2
    exit 1
}
VENDOR_DIR="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd)"
for op_name in "${OP_NAMES[@]}"; do
    find "${VENDOR_DIR}" -type f -name binary_info_config.json -print0 | xargs -0 -r grep -q "${op_name}" || {
        echo "[bf16_build] ERROR: ${op_name} is absent from kernel metadata." >&2
        exit 1
    }
done

pushd "${TORCH_EXTENSION}" >/dev/null
rm -rf "${BUILD_ROOT}/torch_extension"
rm -f nanovllm_dsa_a5/_C*.so
MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace --build-temp "${BUILD_ROOT}/torch_extension"
popd >/dev/null

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export NANOVLLM_A5_INSTALL_OPP_PATH="${LOCAL_OPP}"
export NANOVLLM_CUST_OPAPI_LIB="${OPAPI_LIBS[0]}"
PYTHONPATH="${TORCH_EXTENSION}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c "import nanovllm_dsa_a5; print('[bf16_build] torch extension import: OK')"
echo "[bf16_build] complete: ${VENDOR_DIR}"
