#!/bin/bash
# Build the xllm x_attention reference. This repo ships SOURCE ONLY -- and, for the
# operator itself, not even that: the op source lives in the pinned `third_party/
# xllm_ops` submodule (the CANN custom-op build framework), which is the single
# source of truth. We vendor only our own pybind bridge (src/pybind, python/).
#
#   backends/xllm/ref/
#     src/pybind/         our trimmed torch-extension bridge (RegisterOps + setup) [tracked]
#     python/custom_ops.py  our trimmed python wrapper                             [tracked]
#     third_party/xllm_ops/ git submodule: op source + CANN build framework        [submodule]
#     opp/                installed custom_xllm_math opp vendor package            [built, gitignored]
#     python/*.so         custom_ops_lib pybind extension                          [built, gitignored]
#
# The build artifacts (opp/, python/*.so) are NOT committed; this script (re)builds
# them. Keep the framework path fixed so the build cannot silently pick up an
# unrelated external xllm_ops checkout.
set -e

REF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC="${SOC:-ascend910_93}"
FRAMEWORK="${REF}/third_party/xllm_ops"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

# --- 0. Ensure the framework submodule + its catlass headers are present ------ #
if [ ! -f "${FRAMEWORK}/xllm_ops/build.sh" ]; then
  echo ">>> [0/2] framework not populated at ${FRAMEWORK}"
  if git -C "${REF}" rev-parse --git-dir >/dev/null 2>&1; then
    echo ">>>       initializing submodule third_party/xllm_ops ..."
    git -C "${REF}" submodule update --init -- third_party/xllm_ops
  fi
fi
if [ ! -f "${FRAMEWORK}/xllm_ops/build.sh" ]; then
  echo "!!! build framework missing: ${FRAMEWORK}/xllm_ops/build.sh not found." >&2
  echo "!!! run 'git submodule update --init' in the xa repository." >&2
  exit 1
fi
# The op includes catlass headers from third_party/catlass/include -- a nested
# submodule of the framework. Init ONLY catlass (x_attention does not use pto-isa),
# which avoids extra network to cann/pto-isa.
if [ ! -e "${FRAMEWORK}/third_party/catlass/include/catlass" ]; then
  echo ">>>       initializing nested catlass submodule (headers) ..."
  git -C "${FRAMEWORK}" submodule update --init -- third_party/catlass
fi

# --- 1. Operator: build aclnnXAttention straight from the submodule ----------- #
# No source staging: the submodule already holds x_attention + common/kernel_common
# + catlass at the pinned commit. Building its own tree keeps a single source of
# truth (no vendored copy to drift).
echo ">>> [1/2] building x_attention operator (soc=${SOC}) via ${FRAMEWORK}"
(
  cd "${FRAMEWORK}/xllm_ops"
  export CPATH="${FRAMEWORK}/third_party/catlass/include:${CPATH}"
  # The bottleneck is the framework compiling protobuf + abseil from source (~4 min,
  # one-time), NOT the op (~15s). Reusing build/ makes reruns ~15s. Set
  # SETUP_REF_CLEAN=1 to force a clean rebuild (e.g. after changing --soc).
  [ -n "${SETUP_REF_CLEAN}" ] && rm -rf build output build_out
  bash build.sh --pkg --ops=x_attention --soc="${SOC}"
)
# The final self-extracting package lands directly in build/ (CPack also leaves an
# identically-named _CPack_Packages/.../*.run *directory*, so restrict to a plain
# file at build/ top level).
RUN=$(find "${FRAMEWORK}/xllm_ops/build" -maxdepth 1 -type f -name "cann-ops-xllm-custom*.run" | head -1)
if [ -z "${RUN}" ] || [ ! -f "${RUN}" ]; then
  echo "!!! build produced no .run package under ${FRAMEWORK}/xllm_ops/build" >&2
  exit 1
fi
rm -rf "${REF}/opp" && mkdir -p "${REF}/opp"
bash "${RUN}" --install-path="${REF}/opp" --quiet

# --- 2. pybind bridge: build into ref/python --------------------------------- #
# RegisterOps.cpp needs pytorch_npu_helper.hpp; setup.py includes it from the
# in-repo submodule (test/python_test).
echo ">>> [2/2] building custom_ops_lib pybind"
(
  cd "${REF}/src/pybind"
  rm -rf build dist *.egg-info
  python setup.py build_ext --inplace
)
cp "${REF}"/src/pybind/custom_ops_lib*.so "${REF}/python/"

echo ">>> done. opp -> ${REF}/opp ; python -> ${REF}/python (both gitignored)"
echo ">>> source env.sh to use it."
