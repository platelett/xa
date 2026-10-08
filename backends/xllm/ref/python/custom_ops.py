"""Minimal custom_ops bridge for the x_attention reference operator.

Trimmed from xLLM-AI/xllm_ops test/python_test/custom_ops.py to only the op this
harness needs. Backed by the custom_ops_lib pybind extension (sibling .so), which
dlsyms aclnnXAttention from the custom_xllm_math opp vendor package.
"""

import custom_ops_lib


def x_attention_npu(query, key_cache, value_cache, unshared_key, unshared_value,
                    actual_shared_kvlen, decode_step,
                    shared_block_tables=None, unshared_block_tables=None):
    return custom_ops_lib.x_attention(
        query, key_cache, value_cache, unshared_key, unshared_value,
        shared_block_tables, unshared_block_tables, actual_shared_kvlen, decode_step,
    )
