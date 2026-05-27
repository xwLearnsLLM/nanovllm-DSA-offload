#!/bin/bash
set -e

SUPPORT_COMPUTE_UNITS=("ascend910b" "ascend910_93")

BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
BUILD_PATH="${BASE_PATH}/build"

CORE_NUMS=$(grep -c "processor" /proc/cpuinfo 2>/dev/null || echo 8)
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for DsaIndexUpdate operator"
  echo "Usage: bash build.sh --soc=<soc> [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --soc=soc_version       ascend910b, ascend910_93"
  echo "  --list-socs             List supported SoC versions"
  echo "  -j[n]                   Build jobs, default ${CORE_NUMS}"
  echo "  --make_clean            Clean build directory"
  echo "  -s, --st                Deprecated; use ut_ops/probe_dsa_index_update.py"
}

check_compute_unit() {
  local unit="$1"
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "$unit" == "$support_unit" ]]; then
      return 0
    fi
  done
  return 1
}

THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
RUN_ST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      printf "%s\n" "${SUPPORT_COMPUTE_UNITS[@]}"
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "${THREAD_NUM}" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    --make_clean)
      rm -rf "${BUILD_PATH}"
      exit 0
      ;;
    -s|--st)
      echo "[WARN] --st is deprecated; use ut_ops/probe_dsa_index_update.py after repository-level build."
      RUN_ST=false
      shift
      ;;
    *)
      echo "[ERROR] Invalid option: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${COMPUTE_UNIT}" ]; then
  echo "[ERROR] --soc is required"
  usage
  exit 1
fi

COMPUTE_UNIT=$(echo "${COMPUTE_UNIT}" | tr '[:upper:]' '[:lower:]')
if ! check_compute_unit "${COMPUTE_UNIT}"; then
  echo "[ERROR] Unsupported SoC: ${COMPUTE_UNIT}"
  exit 1
fi

mkdir -p "${BUILD_PATH}"
rm -f "${BUILD_PATH}/CMakeCache.txt"

echo "[INFO] Configuring for ${COMPUTE_UNIT}"
cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" -DASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}"

echo "[INFO] Building"
cmake --build "${BUILD_PATH}" --target all binary package -- -j "${THREAD_NUM}"

PKG_PATH=$(find "${BUILD_PATH}" -maxdepth 1 -type f -name "custom_opp_*_aarch64.run" | head -1)
if [ -z "${PKG_PATH}" ] || [ ! -s "${PKG_PATH}" ]; then
  echo "[ERROR] Package not found under ${BUILD_PATH}/custom_opp_*_aarch64.run"
  exit 1
fi

echo "[INFO] Build completed: ${PKG_PATH}"

if [ "${RUN_ST}" = true ]; then
  echo "[WARN] Internal ST tests are not shipped for this adapted interface."
fi
