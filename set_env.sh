#!/usr/bin/env bash
# Source this file before running python3 run_test.py on this machine.

export CANN_HOME="${CANN_HOME:-/home/lenovo/.codex/memories/cann-9.0.0/cann-9.0.0}"
export ARCH="${ARCH:-x86_64-linux}"
export SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
export NPU_TYPE="${NPU_TYPE:-$SOC_VERSION}"
export CORE_ARCH="${CORE_ARCH:-dav-c310-vec}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
export ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}"

if [ -f "$CANN_HOME/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$CANN_HOME/set_env.sh"
fi

export ASCEND_HOME_PATH="$CANN_HOME"
export ASCEND_CANN_PACKAGE_PATH="$CANN_HOME"
export ASCEND_TOOLKIT_HOME="$CANN_HOME"
export ASCEND_OPP_PATH="$CANN_HOME/opp"
export PYTHONPATH="$CANN_HOME/python/site-packages:${PYTHONPATH:-}"

prepend_path() {
  local d="$1"
  [ -d "$d" ] && export PATH="$d:${PATH:-}"
}

prepend_ld_path() {
  local d="$1"
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
}

prepend_path "$CANN_HOME/$ARCH/bin"
prepend_path "$CANN_HOME/$ARCH/simulator/bin"
prepend_path "$CANN_HOME/$ARCH/simulator/dav_3510"
prepend_path "$CANN_HOME/bin"
prepend_path "$CANN_HOME/compiler/bin"

prepend_ld_path "$CANN_HOME/$ARCH/simulator/$SOC_VERSION/lib"
prepend_ld_path "$CANN_HOME/tools/simulator/$SOC_VERSION/lib"
prepend_ld_path "$CANN_HOME/$ARCH/simulator/dav_3510/lib"
prepend_ld_path "$CANN_HOME/$ARCH/simulator/dav_3510/camodel"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/linux/aarch64"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/linux/x86_64"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/device"
prepend_ld_path "$CANN_HOME/$ARCH/devlib"
prepend_ld_path "$CANN_HOME/$ARCH/lib64/device/lib64"
prepend_ld_path "$CANN_HOME/$ARCH/lib64"
prepend_ld_path "$CANN_HOME/runtime/lib64"
prepend_ld_path "$CANN_HOME/fwkacllib/lib64"
prepend_ld_path "$CANN_HOME/lib64"

unset LD_PRELOAD

echo "[INFO] CANN_HOME=$CANN_HOME"
echo "[INFO] ARCH=$ARCH"
echo "[INFO] SOC_VERSION=$SOC_VERSION"
echo "[INFO] CORE_ARCH=$CORE_ARCH"
