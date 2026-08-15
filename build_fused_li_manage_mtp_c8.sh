#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_NAME="A5FusedLiManageMtpC8"
OP_DIR="fused_li_manage_mtp_c8"
OP_SPEC="${ROOT}/csrc/ops.json"
OP_SOURCE="${ROOT}/csrc/src/${OP_DIR}"
QLI_ARCH35_SOURCE="${ROOT}/csrc/src/fused_li_manage_c8/op_kernel/arch35"
BUILD_ROOT="${ROOT}/build/${OP_DIR}"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${BUILD_ROOT}/opp"
STAGED_OPP="${BUILD_ROOT}/opp.staged.$$"
STAMP="${BUILD_ROOT}/project.stamp"
FULL_OPP="${ROOT}/_custom_opp"
TORCH_EXTENSION="${ROOT}/torch_extension"

PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"
BUILD_JOBS="${NANOVLLM_A5_OPS_BUILD_JOBS:-$(nproc)}"
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-${SOC_VERSION:-ascend950}}"
case "${A5_SOC_VERSION_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*)
        A5_SOC_VERSION="ascend950"
        ;;
    *)
        echo "[mtp_c8_incremental] ERROR: only Ascend 950 is supported; got ${A5_SOC_VERSION_RAW}." >&2
        exit 2
        ;;
esac

if ! command -v msopgen >/dev/null 2>&1; then
    echo "[mtp_c8_incremental] ERROR: msopgen is unavailable; source the CANN 9.1 environment first." >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[mtp_c8_incremental] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${OP_SOURCE}/op_host" || ! -d "${OP_SOURCE}/op_kernel" || ! -d "${QLI_ARCH35_SOURCE}" ]]; then
    echo "[mtp_c8_incremental] ERROR: fused_li_manage_mtp_c8 sources are incomplete." >&2
    exit 2
fi

mapfile -t FULL_OPAPI_LIBS < <(
    find "${FULL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so' 2>/dev/null
)
if [[ "${#FULL_OPAPI_LIBS[@]}" -ne 1 ]] ||
   ! compgen -G "${TORCH_EXTENSION}/nanovllm_dsa_a5/_C*.so" >/dev/null; then
    echo "[mtp_c8_incremental] ERROR: a complete base build is required once; run bash build.sh first." >&2
    exit 2
fi

mkdir -p "${BUILD_ROOT}"
SPEC_HASH="$(sha256sum "${OP_SPEC}" | awk '{print $1}')"
PROJECT_STAMP="${SPEC_HASH}|${A5_SOC_VERSION}|${ASCEND_HOME_PATH:-${CANN_INSTALL_PATH:-}}"
CURRENT_STAMP="$(cat "${STAMP}" 2>/dev/null || true)"
if [[ ! -x "${GENERATED}/build.sh" || "${CURRENT_STAMP}" != "${PROJECT_STAMP}" ||
      "${NANOVLLM_A5_INCREMENTAL_REGENERATE:-0}" == "1" ]]; then
    echo "[mtp_c8_incremental] generate one-op project"
    rm -rf "${GENERATED}"
    msopgen gen \
        -i "${OP_SPEC}" \
        -f aclnn \
        -c "ai_core-${A5_SOC_VERSION}" \
        -lan cpp \
        -op "${OP_NAME}" \
        -out "${GENERATED}"
    printf '%s\n' "${PROJECT_STAMP}" > "${STAMP}"
else
    echo "[mtp_c8_incremental] reuse generated one-op project"
fi

# The MTP kernel shares the checked-in official C8 QLI service headers with
# fused_li_manage_c8.  Copy sources into the isolated msopgen workspace; no
# source file is loaded from another repository at build or runtime.
cp -a "${OP_SOURCE}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_SOURCE}/op_kernel/." "${GENERATED}/op_kernel/"
mkdir -p "${GENERATED}/op_kernel/arch35"
cp -a "${QLI_ARCH35_SOURCE}/." "${GENERATED}/op_kernel/arch35/"

echo "[mtp_c8_incremental] compile ${OP_NAME} only, jobs=${BUILD_JOBS}"
pushd "${GENERATED}" >/dev/null
# msopgen generated this workspace with -op ${OP_NAME}, so it contains only
# this op.  Avoid -n here: CANN build templates differ on whether -n expects
# the public CamelCase op type or the snake-case kernel file name.
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh -c "${A5_SOC_VERSION}"
popd >/dev/null

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
    echo "[mtp_c8_incremental] ERROR: the one-op build produced no .run package." >&2
    exit 1
fi

rm -rf "${STAGED_OPP}"
mkdir -p "${STAGED_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${STAGED_OPP}"

mapfile -t INCREMENTAL_OPAPI_LIBS < <(
    find "${STAGED_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so'
)
if [[ "${#INCREMENTAL_OPAPI_LIBS[@]}" -ne 1 ]]; then
    echo "[mtp_c8_incremental] ERROR: expected one incremental libcust_opapi.so, found ${#INCREMENTAL_OPAPI_LIBS[@]}." >&2
    exit 1
fi
INCREMENTAL_VENDOR="$(cd "$(dirname "${INCREMENTAL_OPAPI_LIBS[0]}")/../.." && pwd)"
if ! find "${INCREMENTAL_VENDOR}" -type f -name 'binary_info_config.json' -print0 |
     xargs -0 -r grep -q "${OP_NAME}"; then
    echo "[mtp_c8_incremental] ERROR: ${OP_NAME} is absent from incremental kernel metadata." >&2
    exit 1
fi

rm -rf "${LOCAL_OPP}.previous"
if [[ -d "${LOCAL_OPP}" ]]; then
    mv "${LOCAL_OPP}" "${LOCAL_OPP}.previous"
fi
mv "${STAGED_OPP}" "${LOCAL_OPP}"
rm -rf "${LOCAL_OPP}.previous"

PYTHONPATH="${TORCH_EXTENSION}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
    "import nanovllm_dsa_a5; print('[mtp_c8_incremental] active opapi:', nanovllm_dsa_a5.local_opapi_path())"

echo "[mtp_c8_incremental] build complete"
echo "[mtp_c8_incremental] only ${OP_NAME} was rebuilt; start a fresh Python process before testing"
