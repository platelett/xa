"""Backend metadata + the contracts the framework talks to.

Two roles, kept apart:

  * A *backend* is a pure declaration (a module under backends/<name>/backend.py)
    that exposes ``META`` plus a few small "core logic" functions. It does NOT
    know how to compile, launch, profile, or schedule.

  * A *driver* (core/drivers/<type>.py) owns all the type-specific dirty work --
    turning a backend declaration into something runnable, and compiling it.
    The framework picks the driver by ``META.type``.

The framework (runner/bench) only ever uses ``META`` + the driver, so it is fully
backend-agnostic.

Backend module contract by type:
  type == "tilelang":  kernel(case, *, kernel_name: str) -> JITKernel
                         build the operator from the problem sizes only (shape-only,
                         cacheable, safe without an NPU). The kernel auto-allocates
                         its Output + scratch/workspace (out_idx / workspace_idx).
                         Set global_symbol to kernel_name before JIT compilation;
                         the driver supplies the registry's backend directory name
                         and checks the result on both compile and run paths.
                       assemble(case, canonical) -> args_tuple
                         map the shared canonical tensors to the kernel's positional
                         args (needs the real NPU tensors); returned output is already
                         in canonical shape.
  type == "torch"   :  adapt(case, canonical) -> op_args
                       call(op_args) -> attn_out in canonical shape
  type == "ascendc" :  adapt(case, canonical) -> op_args
                       call(op_args) -> attn_out in canonical shape

Driver module contract:
  requires_compile: bool
  compile(spec, case) -> None
  make_call(spec, case, canonical) -> Callable[[], Tensor]   # one kernel/op run
"""

from dataclasses import dataclass


class KernelNameError(ValueError):
    """A TileLang backend violated the harness's compiled-symbol contract."""


@dataclass(frozen=True)
class Meta:
    name: str                    # legacy; framework keys backends by directory name
    type: str                    # "tilelang" | "torch" | "ascendc"  (selects the driver)
    op_type: str                 # fallback kernel-name filter for profiling (used when
                                 # the profiled kernel name isn't the directory name,
                                 # e.g. torch/ascendc ops like FlashAttentionScore)
    is_reference: bool = False   # exactly one backend is the correctness reference
    author: str = ""
    date: str = ""
    notes: str = ""
