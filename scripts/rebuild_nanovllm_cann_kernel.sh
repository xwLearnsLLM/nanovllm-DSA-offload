#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_BUILD_DIR="${ROOT_DIR}/csrc/nanovllm_ascend_ops/cann_ops/build"
INSTALLED_OPP_ROOT="${ROOT_DIR}/nanovllm/_cann_ops_custom"
OP_NAME="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/rebuild_nanovllm_cann_kernel.sh fused_li_manage_mtp

This incremental path requires one successful full build first. It is only
for AscendC device-kernel/header changes; host tiling, schema, op-api, binding,
or CMake changes still require scripts/build_nanovllm_ops.sh.
EOF
}

if (( $# != 1 )); then
  usage >&2
  exit 2
fi

case "${OP_NAME}" in
  fused_li_manage_mtp)
    KERNEL_NAME="nanovllm_fused_li_manage_mtp"
    OP_TYPE="NanovllmFusedLiManageMtp"
    DEPENDENCY_SOURCE_DIRS=("fused_li_manage")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

RAW_SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
case "${RAW_SOC_VERSION}" in
  ascend910_93*) CANN_SOC_VERSION="ascend910_93" ;;
  ascend910b*) CANN_SOC_VERSION="ascend910b" ;;
  ascend310*) CANN_SOC_VERSION="ascend310p" ;;
  *) CANN_SOC_VERSION="${RAW_SOC_VERSION}" ;;
esac

if [[ ! -f "${CANN_BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "[nanovllm incremental] ERROR: ${CANN_BUILD_DIR} is not a configured build directory." >&2
  echo "[nanovllm incremental] Run scripts/build_nanovllm_ops.sh once first." >&2
  exit 1
fi
if [[ ! -d "${INSTALLED_OPP_ROOT}" ]]; then
  echo "[nanovllm incremental] ERROR: installed custom OPP was not found at ${INSTALLED_OPP_ROOT}." >&2
  echo "[nanovllm incremental] Run scripts/build_nanovllm_ops.sh once first." >&2
  exit 1
fi

CACHED_ASCEND_HOME="$({
  sed -n \
    -e 's/^CUSTOM_ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' \
    -e 's/^ASCEND_HOME_PATH:[^=]*=//p' \
    "${CANN_BUILD_DIR}/CMakeCache.txt"
} | head -n 1)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CACHED_ASCEND_HOME}}"
if [[ -n "${ASCEND_HOME_PATH}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ASCEND_HOME_PATH}/bin/setenv.bash" >/dev/null
  set -u
fi

if [[ -n "${NANOVLLM_CANN_BUILD_JOBS:-}" ]]; then
  BUILD_JOBS="${NANOVLLM_CANN_BUILD_JOBS}"
else
  BUILD_JOBS="$(nproc)"
  if (( BUILD_JOBS > 64 )); then
    BUILD_JOBS=64
  fi
fi
if ! [[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[nanovllm incremental] ERROR: NANOVLLM_CANN_BUILD_JOBS must be a positive integer." >&2
  exit 1
fi

mapfile -t BIN_DIRS < <(
  find "${CANN_BUILD_DIR}" -type d -path "*/${CANN_SOC_VERSION}/bin" -print
)
if (( ${#BIN_DIRS[@]} != 1 )); then
  echo "[nanovllm incremental] ERROR: expected one ${CANN_SOC_VERSION}/bin directory, found ${#BIN_DIRS[@]}." >&2
  printf '  %s\n' "${BIN_DIRS[@]:-}" >&2
  echo "[nanovllm incremental] Run a full build for SOC_VERSION=${RAW_SOC_VERSION}." >&2
  exit 1
fi
BIN_DIR="${BIN_DIRS[0]}"
BINARY_ROOT="$(dirname "${BIN_DIR}")"
GEN_DIR="${BINARY_ROOT}/gen"
SRC_DIR="${BINARY_ROOT}/src"
TARGET_NAME="${KERNEL_NAME}_${CANN_SOC_VERSION}"
CONFIG_TARGET="ops_config_${CANN_SOC_VERSION}"

TARGET_HELP="$(cmake --build "${CANN_BUILD_DIR}" --target help 2>/dev/null || true)"
if ! grep -Fq "${TARGET_NAME}" <<<"${TARGET_HELP}"; then
  echo "[nanovllm incremental] ERROR: CMake target ${TARGET_NAME} is unavailable." >&2
  grep -F "${OP_NAME}" <<<"${TARGET_HELP}" >&2 || true
  echo "[nanovllm incremental] Run a full build before using the incremental path." >&2
  exit 1
fi

mapfile -t INSTALLED_CONFIGS < <(
  find "${INSTALLED_OPP_ROOT}" -type f \
    -path "*/kernel/config/${CANN_SOC_VERSION}/binary_info_config.json" -print
)
if (( ${#INSTALLED_CONFIGS[@]} != 1 )); then
  echo "[nanovllm incremental] ERROR: expected one installed binary_info_config.json, found ${#INSTALLED_CONFIGS[@]}." >&2
  printf '  %s\n' "${INSTALLED_CONFIGS[@]:-}" >&2
  exit 1
fi
INSTALLED_CONFIG="${INSTALLED_CONFIGS[0]}"
INSTALLED_CONFIG_SOC_DIR="$(dirname "${INSTALLED_CONFIG}")"
INSTALLED_KERNEL_ROOT="$(dirname "$(dirname "${INSTALLED_CONFIG_SOC_DIR}")")"
INSTALLED_KERNEL_SOC_DIR="${INSTALLED_KERNEL_ROOT}/${CANN_SOC_VERSION}"

echo "[nanovllm incremental] op: ${OP_NAME} (${KERNEL_NAME})"
echo "[nanovllm incremental] soc: raw=${RAW_SOC_VERSION}, cann_opp=${CANN_SOC_VERSION}"
echo "[nanovllm incremental] build: ${CANN_BUILD_DIR}"
echo "[nanovllm incremental] install: ${INSTALLED_OPP_ROOT}"
echo "[nanovllm incremental] jobs: ${BUILD_JOBS}"

# CANN's generated build graph uses copied sources and .done files as outputs.
# Invalidate only this kernel and its shared source copy; all unrelated kernels
# retain their stamps and are therefore no-ops when ops_config is regenerated.
rm -rf "${SRC_DIR}/${KERNEL_NAME}" "${SRC_DIR}/${OP_NAME}"
for dependency in "${DEPENDENCY_SOURCE_DIRS[@]}"; do
  rm -rf "${SRC_DIR}/${dependency}"
done
find "${GEN_DIR}" -maxdepth 1 -type f \
  -name "${TARGET_NAME}_*.done" -delete
find "${CANN_BUILD_DIR}" -type f \
  -name "${TARGET_NAME}_src_copy.done" -delete
for source_dir_name in "${OP_NAME}" "${DEPENDENCY_SOURCE_DIRS[@]}"; do
  find "${CANN_BUILD_DIR}" -type f \
    -name "${source_dir_name}_${CANN_SOC_VERSION}_src_copy.done" -delete
done
rm -rf "${BIN_DIR:?}/${KERNEL_NAME}" "${BIN_DIR:?}/${OP_NAME}"
rm -f \
  "${BIN_DIR}/${KERNEL_NAME}.json" \
  "${BIN_DIR}/${OP_NAME}.json" \
  "${BIN_DIR}/binary_info_config.json"

BUILD_GENERATOR="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' \
  "${CANN_BUILD_DIR}/CMakeCache.txt" | head -n 1)"
if [[ "${BUILD_GENERATOR}" == "Unix Makefiles" ]]; then
  # -B forces the selected kernel's dependency closure to run again.  This is
  # needed because OPC/CMake may otherwise reuse a binary after only a shared
  # transitive device header changed.
  cmake --build "${CANN_BUILD_DIR}" \
    --target "${TARGET_NAME}" -j"${BUILD_JOBS}" -- -B
else
  cmake --build "${CANN_BUILD_DIR}" \
    --target "${TARGET_NAME}" -j"${BUILD_JOBS}"
fi
cmake --build "${CANN_BUILD_DIR}" \
  --target "${CONFIG_TARGET}" -j"${BUILD_JOBS}"

BUILT_CONFIG="${BIN_DIR}/binary_info_config.json"
if [[ ! -f "${BUILT_CONFIG}" ]] || ! grep -q "${OP_TYPE}" "${BUILT_CONFIG}"; then
  echo "[nanovllm incremental] ERROR: rebuilt binary_info_config.json does not contain ${OP_TYPE}." >&2
  exit 1
fi

copy_directory_atomically() {
  local source_dir="$1"
  local destination_dir="$2"
  local staging_dir="${destination_dir}.tmp.$$"
  rm -rf "${staging_dir}"
  mkdir -p "${staging_dir}"
  cp -a "${source_dir}/." "${staging_dir}/"
  rm -rf "${destination_dir}"
  mv "${staging_dir}" "${destination_dir}"
}

copy_file_atomically() {
  local source_file="$1"
  local destination_file="$2"
  local staging_file="${destination_file}.tmp.$$"
  mkdir -p "$(dirname "${destination_file}")"
  cp -f "${source_file}" "${staging_file}"
  mv -f "${staging_file}" "${destination_file}"
}

COPIED_KERNEL_DIRS=0
for kernel_dir_name in "${KERNEL_NAME}" "${OP_NAME}"; do
  source_dir="${BIN_DIR}/${kernel_dir_name}"
  if [[ -d "${source_dir}" ]]; then
    mkdir -p "${INSTALLED_KERNEL_SOC_DIR}"
    copy_directory_atomically \
      "${source_dir}" "${INSTALLED_KERNEL_SOC_DIR}/${kernel_dir_name}"
    COPIED_KERNEL_DIRS=$((COPIED_KERNEL_DIRS + 1))
  fi
  source_json="${BIN_DIR}/${kernel_dir_name}.json"
  if [[ -f "${source_json}" ]]; then
    copy_file_atomically \
      "${source_json}" "${INSTALLED_CONFIG_SOC_DIR}/${kernel_dir_name}.json"
  fi
done
if (( COPIED_KERNEL_DIRS == 0 )); then
  echo "[nanovllm incremental] ERROR: rebuilt kernel directory was not found under ${BIN_DIR}." >&2
  exit 1
fi

copy_file_atomically "${BUILT_CONFIG}" "${INSTALLED_CONFIG}"
if ! grep -q "${OP_TYPE}" "${INSTALLED_CONFIG}"; then
  echo "[nanovllm incremental] ERROR: installed config verification failed." >&2
  exit 1
fi

echo "[nanovllm incremental] rebuilt and installed ${KERNEL_NAME} only"
echo "[nanovllm incremental] verified ${OP_TYPE} in ${INSTALLED_CONFIG}"
