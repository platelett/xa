"""ascendc driver -- all ascendc-type dirty work (op build/load + call shape).

A prebuilt aclnn op (e.g. the xllm custom_xllm_math vendor package) needs no
compile step; a backend that builds from source may expose an optional
``build(case)`` hook, which the driver will call.
"""

requires_compile = False


def compile(spec, case):
    """Optionally build/prepare the op; prebuilt vendor packages are a no-op."""
    hook = getattr(spec, "build", None)
    if hook is not None:
        hook(case)


def make_call(spec, case, canonical):
    """One ready-to-run op call. Mirrors the tilelang driver's build/feed split, but
    the op is prebuilt so there's nothing to build: ``adapt`` maps the shared
    canonical tensors to this op's inputs, ``call`` invokes it (returns the output)."""
    op_args = spec.adapt(case, canonical)
    return lambda: spec.call(op_args)
