"""xllm -- the upstream Ascend C x_attention operator, as the reference backend.

Pure declaration: how to adapt the canonical inputs to the op's arg layout and how
to call it. The op requires exactly one paged side, so the unshared KV is packed
into the paged 5-D layout with an identity request->slot table (numerically
identical to a continuous unshared layout); the shared KV stays continuous.

Building/loading the operator is the ascendc driver's job; calling it needs the
custom_ops pybind bridge + the custom_xllm_math vendor package on the runtime
search path (see env.sh).
"""

from core.contract import Meta

META = Meta(
    name="xllm",
    type="ascendc",
    op_type="XAttention",
    is_reference=True,
    author="xLLM-AI/xllm_ops",
    date="2026-06-30",
    notes="aclnnXAttention (custom_xllm_math), built from source for CANN 9.1 / ascend910_93.",
)

_MAX_DECODE_STEP = 3  # upstream pads unshared KV to this many steps


def adapt(case, canonical):
    import torch

    beam = case.beam_size
    r = case.request_num
    n = case.num_tokens

    uk = torch.zeros(r, beam, case.kv_heads, _MAX_DECODE_STEP, case.head_dim,
                     dtype=torch.bfloat16, device="npu")
    uv = torch.zeros_like(uk)
    uk_src = canonical["unshared_key"].reshape(r, beam, case.kv_heads, case.decode_step, case.head_dim)
    uv_src = canonical["unshared_value"].reshape(r, beam, case.kv_heads, case.decode_step, case.head_dim)
    uk[:, :, :, : case.decode_step, :] = uk_src
    uv[:, :, :, : case.decode_step, :] = uv_src

    return {
        "query": canonical["query"].reshape(n, case.num_heads, case.head_dim),
        "key_cache": canonical["shared_key"],
        "value_cache": canonical["shared_value"],
        "unshared_key": uk,
        "unshared_value": uv,
        "actual_shared_kvlen": torch.tensor([case.kv_seqlen] * r, dtype=torch.int32, device="npu"),
        "decode_step": torch.tensor([case.decode_step], dtype=torch.int32, device="npu"),
        "shared_block_tables": None,
        "unshared_block_tables": torch.tensor([[i] for i in range(r)], dtype=torch.int32, device="npu"),
    }


def call(op_args):
    import custom_ops

    return custom_ops.x_attention_npu(
        op_args["query"], op_args["key_cache"], op_args["value_cache"],
        op_args["unshared_key"], op_args["unshared_value"],
        op_args["actual_shared_kvlen"], op_args["decode_step"],
        op_args["shared_block_tables"], op_args["unshared_block_tables"],
    )
