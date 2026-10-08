"""Canonical decode inputs -- backend-agnostic, a pure function of the Case.

Both backends consume the *same* canonical tensors (each adapts the layout it
needs), so their outputs are directly comparable. Depends only on Case, never on
any backend's tiling config.
"""

import torch


def build_canonical(case, seed: int = 42) -> dict:
    """Allocate the canonical inputs on the NPU.

    Shapes (N = request_num * beam_size; one shared block per request):
      query        [N, num_heads, head_dim]
      shared_key   [request_num * kv_seqlen, kv_heads, head_dim]
      shared_value [request_num * kv_seqlen, kv_heads, head_dim]
      unshared_key [N, kv_heads, decode_step, head_dim]
      unshared_val [N, kv_heads, decode_step, head_dim]
    """
    torch.manual_seed(seed)
    torch.set_default_device("npu")
    rand = lambda *shape: torch.empty(shape, dtype=torch.bfloat16).uniform_(-1, 1)

    n = case.num_tokens
    return {
        "query": rand(n, case.num_heads, case.head_dim),
        "shared_key": rand(case.request_num * case.kv_seqlen, case.kv_heads, case.head_dim),
        "shared_value": rand(case.request_num * case.kv_seqlen, case.kv_heads, case.head_dim),
        "unshared_key": rand(n, case.kv_heads, case.decode_step, case.head_dim),
        "unshared_value": rand(n, case.kv_heads, case.decode_step, case.head_dim),
    }
