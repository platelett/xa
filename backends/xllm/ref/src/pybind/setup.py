import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# RegisterOps.cpp includes pytorch_npu_helper.hpp for the EXEC_NPU_CMD macro. That
# header is not vendored (it is a verbatim upstream file); we include it straight
# from the in-repo xllm_ops submodule -- single source of truth.
REF = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
FRAMEWORK = os.path.join(REF, "third_party", "xllm_ops")
HELPER_INC = os.path.join(FRAMEWORK, "test", "python_test")
if not os.path.isfile(os.path.join(HELPER_INC, "pytorch_npu_helper.hpp")):
    raise SystemExit(
        f"pytorch_npu_helper.hpp not found under {HELPER_INC}\n"
        "Init the in-repo xllm_ops submodule with git submodule update --init."
    )

source_files = glob.glob(os.path.join(BASE_DIR, "*.cpp"), recursive=True)

exts = []
ext = NpuExtension(
    name="custom_ops_lib",
    sources=source_files,
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-I' + HELPER_INC,
    ],
)
exts.append(ext)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
