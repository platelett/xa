"""Expert v2 from TileLang-Ascend PR #1864@b41fdd3f."""
from core.contract import Meta
META = Meta(name="expert_v2", type="tilelang", op_type="main_kernel", date="2026-10-08",
            notes="PR #1864 public kernel; BF16, 24 Cube groups, composed softmax, inline private-tail merge.")
def kernel(case, *, kernel_name: str):
    from backends.expert_v2 import impl
    return impl.flash_attention_xattn_fwd(case.request_num, case.beam_size, case.kv_heads, case.group_size,
                                       case.kv_seqlen, case.decode_step, case.head_dim, kernel_name=kernel_name)
def assemble(case, canonical):
    from backends.expert_v2 import impl
    return impl.make_args(case, canonical)
