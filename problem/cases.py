"""The x_attention test matrix -- the *problem*, decoupled from any backend.

A Case carries only logical workload dimensions (no tiling, no derived kernel
constants, no _require). It is the shared, editable definition every backend must
solve. Mirrors the 18-case parametrization of xLLM-AI/xllm_ops
test/python_test/test_x_attention.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    request_num: int
    beam_size: int
    kv_seqlen: int               # shared prefix length per request
    num_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    decode_step: int = 2         # unshared decode length (valid entries)

    @property
    def group_size(self) -> int:
        return self.num_heads // self.kv_heads

    @property
    def num_tokens(self) -> int:
        return self.request_num * self.beam_size

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def name(self) -> str:
        return f"R{self.request_num}_B{self.beam_size}_KV{self.kv_seqlen}"


_BEAMS = (128, 256, 512, 1024, 2048, 4096)

# The full 18-case matrix (request_num x beam x kv_seqlen), upstream order.
CASES = (
    tuple(Case(1, b, 1024) for b in _BEAMS)
    + tuple(Case(2, b, 1024) for b in _BEAMS)
    + tuple(Case(2, b, 2048) for b in _BEAMS)
)


def index_of(case: Case) -> int:
    return CASES.index(case)
