// Trimmed from xLLM-AI/xllm_ops test/python_test/RegisterOps.cpp -- only the
// x_attention op the harness needs. Builds the custom_ops_lib pybind extension,
// which dlsyms aclnnXAttention from the custom_xllm_math opp vendor package.
#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

at::Tensor x_attention_impl_npu(const at::Tensor& query,
                                const at::Tensor& key_cache,
                                const at::Tensor& value_cache,
                                const at::Tensor& unshared_key,
                                const at::Tensor& unshared_value,
                                const c10::optional<at::Tensor>& shared_block_tables,
                                const c10::optional<at::Tensor>& unshared_block_tables,
                                const at::Tensor& actual_shared_kvlen,
                                const at::Tensor& decode_step) {
  at::Tensor attnOut = at::empty_like(query);

  EXEC_NPU_CMD(aclnnXAttention,
               query,
               key_cache,
               value_cache,
               unshared_key,
               unshared_value,
               unshared_block_tables,
               actual_shared_kvlen,
               decode_step,
               shared_block_tables,
               attnOut);

  return attnOut;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("x_attention", &x_attention_impl_npu, "x_attention");
}
