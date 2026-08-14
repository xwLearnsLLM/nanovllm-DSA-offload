#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="${ROOT}/csrc"
OP_SPEC="${CSRC}/ops.json"
OP_SOURCE_ROOT="${CSRC}/src"
BUILD_ROOT="${ROOT}/build"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${ROOT}/_custom_opp"
TORCH_EXTENSION="${ROOT}/torch_extension"

PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"
BUILD_JOBS="${NANOVLLM_A5_OPS_BUILD_JOBS:-$(nproc)}"
# The CANN 9.1 AscendC toolchain exposes one generic A5 compute unit:
# ascend950. Normalize 950PR/950DT product names so an old environment value
# cannot turn into the unsupported CMake target ascend950pr_* / ascend950dt_*.
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-${SOC_VERSION:-ascend950}}"
case "${A5_SOC_VERSION_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*)
        A5_SOC_VERSION="ascend950"
        ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: only Ascend 950 is supported; got A5_SOC_VERSION=${A5_SOC_VERSION_RAW}." >&2
        exit 2
        ;;
esac

if ! command -v msopgen >/dev/null 2>&1; then
    echo "[nanovllm_a5_ops] ERROR: msopgen is unavailable; source the CANN 9.1 environment first." >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[nanovllm_a5_ops] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

case "${GENERATED}" in
    "${ROOT}"/*) ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: unsafe generated-project path: ${GENERATED}" >&2
        exit 2
        ;;
esac
case "${LOCAL_OPP}" in
    "${ROOT}"/*) ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: unsafe local OPP path: ${LOCAL_OPP}" >&2
        exit 2
        ;;
esac

echo "[nanovllm_a5_ops] root: ${ROOT}"
echo "[nanovllm_a5_ops] Python: ${PYTHON_BIN}"
echo "[nanovllm_a5_ops] A5 target: ${A5_SOC_VERSION}"
if [[ "${A5_SOC_VERSION_RAW,,}" != "${A5_SOC_VERSION}" ]]; then
    echo "[nanovllm_a5_ops] normalized product name ${A5_SOC_VERSION_RAW} -> ${A5_SOC_VERSION}"
fi
echo "[nanovllm_a5_ops] jobs: ${BUILD_JOBS}"

OP_NAMES=(
    A5FusedLiManage
    A5FusedLiManageMtp
    A5FusedLiManageC8
    A5FusedLiManageMtpC8CacheUpdate
    A5KvcacheScatterCopy
    A5KvcacheScatterCopyC8
    A5SparseTailAttention
    A5FusedCopySparseTailAttention
)

# These directories provide the BF16 and C8 operators visible to the framework.
# Non-MTP C8 LIDU is one repository-local MIX kernel. MTP C8 currently uses
# the official quant LightningIndexer plus a repository-local union/update
# kernel; C8 Attention is an adapter over the native A5 QSFA implementation.
FRAMEWORK_OP_DIRS=(
    fused_li_manage
    fused_li_manage_mtp
    kvcache_scatter_copy
    sparse_tail_attention
    fused_li_manage_c8
    fused_li_manage_mtp_c8
    kvcache_scatter_copy_c8
    sparse_tail_attention_c8
    fused_copy_sparse_tail_attention
)

for op_dir in "${FRAMEWORK_OP_DIRS[@]}"; do
    if [[ ! -d "${OP_SOURCE_ROOT}/${op_dir}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: missing operator directory: ${OP_SOURCE_ROOT}/${op_dir}" >&2
        exit 2
    fi
done

rm -rf "${GENERATED}"
mkdir -p "${BUILD_ROOT}"
for index in "${!OP_NAMES[@]}"; do
    op_name="${OP_NAMES[index]}"
    echo "[nanovllm_a5_ops] generate operator: ${op_name}"
    msopgen_args=(
        gen
        -i "${OP_SPEC}"
        -f aclnn
        -c "ai_core-${A5_SOC_VERSION}"
        -lan cpp
        -op "${op_name}"
        -out "${GENERATED}"
    )
    if (( index > 0 )); then
        msopgen_args+=(-m 1)
    fi
    msopgen "${msopgen_args[@]}"
done

# The generated project owns only its build scaffold. Repository sources are
# grouped by framework operator, then flattened into the msopgen workspace so
# the generated CMake behavior and kernel include layout remain unchanged.
SOURCE_DIRS=(common "${FRAMEWORK_OP_DIRS[@]}")
for op_dir in "${SOURCE_DIRS[@]}"; do
    source_dir="${OP_SOURCE_ROOT}/${op_dir}"
    if [[ -d "${source_dir}/op_host" ]]; then
        cp -a "${source_dir}/op_host/." "${GENERATED}/op_host/"
    fi
    if [[ -d "${source_dir}/op_kernel" ]]; then
        cp -a "${source_dir}/op_kernel/." "${GENERATED}/op_kernel/"
    fi
done
COMPAT_HEADER="${GENERATED}/op_host/a5_sfa_shared/ops_log_compat.h"
if [[ ! -f "${COMPAT_HEADER}" ]]; then
    echo "[nanovllm_a5_ops] ERROR: missing flattened host header: ${COMPAT_HEADER}" >&2
    exit 1
fi
# The checked-in LightningIndexer definition deliberately has the same name
# as the msopgen stub and overwrites it.  Remove the legacy split definition
# if an older worktree left it behind; compiling both creates a duplicate
# section in aic-ascend950-ops-info.ini.
rm -f "${GENERATED}/op_host/a5_fused_li_manage_def.cpp"

pushd "${GENERATED}" >/dev/null
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
popd >/dev/null

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
    echo "[nanovllm_a5_ops] ERROR: msopgen project did not produce a .run package." >&2
    exit 1
fi

rm -rf "${LOCAL_OPP}"
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

mapfile -t OPAPI_LIBS < <(find "${LOCAL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so')
if [[ "${#OPAPI_LIBS[@]}" -ne 1 ]]; then
    echo "[nanovllm_a5_ops] ERROR: expected one local libcust_opapi.so, found ${#OPAPI_LIBS[@]}." >&2
    exit 1
fi
VENDOR_DIR="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd)"

for op_name in "${OP_NAMES[@]}"; do
    if ! find "${VENDOR_DIR}" -type f -name 'binary_info_config.json' -print0 |
        xargs -0 -r grep -q "${op_name}"; then
        echo "[nanovllm_a5_ops] ERROR: ${op_name} is absent from generated kernel metadata." >&2
        exit 1
    fi
done

pushd "${TORCH_EXTENSION}" >/dev/null
rm -rf build
rm -f nanovllm_dsa_a5/_C*.so
MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export NANOVLLM_A5_INSTALL_OPP_PATH="${LOCAL_OPP}"
PYTHONPATH="${TORCH_EXTENSION}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
    "import nanovllm_dsa_a5; print('[nanovllm_a5_ops] torch extension import: OK')"

echo "[nanovllm_a5_ops] build complete"
echo "[nanovllm_a5_ops] local vendor: ${VENDOR_DIR}"
