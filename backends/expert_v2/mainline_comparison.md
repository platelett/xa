# 当前主线：Expert v1 与 v2 的完整矩阵对照

## Motivation

用户要求用原 XA 的性能功能比较主线旧版与新增示例，先建立公开仓库，再直接运行
原 `core.bench`。不使用另写的计时器或临时挑选的小输入替代原 18 个测试点。

## Expectation

测量前的问题是：选中的完整 v2 实现，在当前主线编译器和标准输入下，相对旧版是否
仍有收益？不预设每个用例都更快；正确性失败、缺失数据或退化都应保留。
沿用 harness 的 20 次内部重复，不另套多轮测量。

## Change

- 旧版：TileLang-Ascend [`ascendc_pto@4c726701`](https://github.com/tile-ai/tilelang-ascend/commit/4c7267018af02284ff28f1cd090a1492c7985bed)
  的 `examples/xattention/xattention.py`，接入 [expert_v1](../expert_v1/impl_tl.py)。
  只把问题尺寸提为参数，增加 kernel 符号名并构造原输入、scratch、别名和控制张量；
  Host 端按原入口将输出恢复为 `[R*B,32,128]`。不改变设备计算和同步。
- 新版：[PR #1864](https://github.com/tile-ai/tilelang-ascend/pull/1864)
  的 `b41fdd3f8eb8917b46fb20a7b8e6d2e0d029960d`，接入本目录的
  [impl_tl.py](impl_tl.py)。内核与 PR 一致，仅适配 harness 的符号名、缓存开关与输入入口。
- `core/`、`problem/` 与研究仓的原 harness 相同。编译器固定为上述主线版本，
  使用其官方 TVM patches；两版均使用框架默认的 24 个逻辑 Cube 核。
  源码及 harness 的 SHA-256 见 [source_manifest.json](source_manifest.json)。

2026-10-08 测量环境为 Atlas A3（Ascend910_9392）、CANN 9.1.0-beta.1、
torch_npu 2.10.0。BF16 输入输出；32 个 query 头、8 个 K/V 头、维度 128、
每个 beam 两个私有 token。全部用例及标准输入来自
[cases.py](../../problem/cases.py) 和 [canonical.py](../../problem/canonical.py)，seed 为 0。

复现时先按[仓库说明](../../README.md#测试点与运行)准备依赖，选择空闲设备：

```bash
source env.sh
npu-smi info
python -m core.bench --backends expert_v1,expert_v2 --cases all --devices 14
```

14 是本次使用的设备编号，不是固定要求。本次 CPU 阶段已编译 36 个新旧内核；
测量阶段复用这批缓存，以 `--no-compile` 执行上述命令。
每个用例预热 3 次，原 torch_npu profiler 统计 20 次设备内核 Duration 的均值；
不计编译、张量分配或 Host 布局准备耗时。正确性参考为 xLLM `aclnnXAttention`，
`atol=rtol=1e-2`。本次复用已有参考二进制，未验证全新 checkout 的参考构建；
公开仓保留构建配方与固定版本的 submodule，不提交该二进制。

## Result

18 个用例的新旧版均通过正确性检查，最大绝对误差不超过 0.00049。
该批次 v2 在全部用例上更快，v1/v2 为 1.024–1.651x。下表来自同一份当前实测，
不混用[优化历程](README.md)里的历史结果；只支持这些输入与环境下相对旧版的收益，
不证明 v2 在所有输入或设备上最快。

| 用例 | expert_v1，us | expert_v2，us | v1 / v2 |
|---|---:|---:|---:|
| R1_B128_KV1024 | 46.84 | 43.83 | 1.069x |
| R1_B256_KV1024 | 70.29 | 58.23 | 1.207x |
| R1_B512_KV1024 | 114.78 | 86.98 | 1.320x |
| R1_B1024_KV1024 | 202.08 | 150.15 | 1.346x |
| R1_B2048_KV1024 | 388.39 | 262.87 | 1.477x |
| R1_B4096_KV1024 | 774.14 | 494.60 | 1.565x |
| R2_B128_KV1024 | 70.84 | 58.60 | 1.209x |
| R2_B256_KV1024 | 115.39 | 85.70 | 1.346x |
| R2_B512_KV1024 | 201.86 | 151.08 | 1.336x |
| R2_B1024_KV1024 | 386.86 | 260.83 | 1.483x |
| R2_B2048_KV1024 | 787.36 | 495.97 | 1.588x |
| R2_B4096_KV1024 | 1590.37 | 963.37 | 1.651x |
| R2_B128_KV2048 | 108.34 | 105.78 | 1.024x |
| R2_B256_KV2048 | 182.83 | 150.30 | 1.216x |
| R2_B512_KV2048 | 332.68 | 279.25 | 1.191x |
| R2_B1024_KV2048 | 647.17 | 491.30 | 1.317x |
| R2_B2048_KV2048 | 1278.06 | 975.46 | 1.310x |
| R2_B4096_KV2048 | 2622.82 | 1876.78 | 1.398x |

[mainline_comparison.csv](mainline_comparison.csv) 保留原报告的全部 54 行，
包括参考用时和逐用例正确性结果。v1/v2 比值以报告中的两位小数计算。
保留 v1 作为对照、v2 作为所选实现；未进一步修改已测内核。

运行过程中的无效批次没有混入该表：最初选用的设备出现其他进程后取消测量；
随后旧版 Host 输出形状未恢复，导致 harness 中断。修正的仅是 backend 的输出视图，
设备 PrimFunc 未变；复用已编译缓存在空闲设备 14 重新跑完 18 点。
Host 回归 8 项通过。
