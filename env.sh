#!/bin/bash
XA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XA_ROOT
if [ -f "${XA_ROOT}/env.local.sh" ]; then
    source "${XA_ROOT}/env.local.sh"
else
    echo "请从 env.local.sh.example 创建本机配置并设置 TileLang 环境。" >&2
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
_xa_reference="${XA_REFERENCE_ROOT:-${XA_ROOT}/backends/xllm/ref}"
if [ -d "${_xa_reference}/opp/vendors/custom_xllm_math" ]; then
    export ASCEND_CUSTOM_OPP_PATH="${_xa_reference}/opp/vendors/custom_xllm_math:${ASCEND_CUSTOM_OPP_PATH}"
    export PYTHONPATH="${_xa_reference}/python:${PYTHONPATH}"
else
    echo "xLLM 参考尚未构建，请运行 bash backends/xllm/ref/setup_ref.sh。" >&2
fi
_xa_torch_lib="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
_xa_tnpu_lib="$(python -c 'import torch_npu,os;print(os.path.join(os.path.dirname(torch_npu.__file__),"lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH="${_xa_torch_lib}:${_xa_tnpu_lib}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${XA_ROOT}:${PYTHONPATH}"
unset _xa_reference _xa_torch_lib _xa_tnpu_lib
