"""torch driver -- all torch-type dirty work (call shape; no compile).

A torch/torch_npu op needs no compile step. The driver mirrors the tilelang
driver's build/feed split, but the op is prebuilt so there's nothing to build:
``adapt`` maps the shared canonical tensors to the op's inputs, ``call`` invokes
it (and returns the output already in canonical [batch, heads_q, seq_len, dim]
shape).
"""

requires_compile = False


def compile(spec, case):
    """Nothing to build for a prebuilt torch op."""
    return None


def make_call(spec, case, canonical):
    """One ready-to-run op call: adapt the canonical inputs, then invoke."""
    op_args = spec.adapt(case, canonical)
    return lambda: spec.call(op_args)
