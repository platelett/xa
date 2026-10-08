"""Existing non-paged xAttention example, shape-parameterized for the standard harness."""
from core.contract import Meta
META = Meta(name="expert_v1", type="tilelang", op_type="main_kernel", date="2026-10-08",
            notes="ascendc_pto@4c726701 xattention.py; unchanged device schedule, 20 shared/4 private Cube groups.")


class CanonicalOutput:
    """Apply the same output view as the original example's host entry."""

    def __init__(self, compiled, case):
        self.compiled = compiled
        self.prim_func = compiled.prim_func
        self.output_shape = (case.num_tokens, case.num_heads, case.head_dim)

    def __call__(self, *args):
        return self.compiled(*args).reshape(self.output_shape)


def kernel(case, *, kernel_name: str):
    from backends.expert_v1 import impl
    compiled = impl.xattention(case.num_tokens, case.request_num, case.kv_seqlen, case.decode_step,
                              case.num_heads, case.kv_heads, case.head_dim, kernel_name=kernel_name)
    return CanonicalOutput(compiled, case)
def assemble(case, canonical):
    from backends.expert_v1 import impl
    return impl.make_args(case, canonical)
