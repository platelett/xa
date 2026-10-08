# xAttention on Ascend

本仓库用同一组输入比较现有 TileLang XA、Expert v2 和 xLLM 参考算子。它计算非因果
decode attention：每个请求的多个 beam 共用历史 K/V，每个 beam 再带两个私有 token。
共享和私有部分共同参与一次完整 softmax。

## 实现

| Backend | 说明 |
|---|---|
| `xllm` | `aclnnXAttention`，正确性参考 |
| `expert_v1` | TileLang-Ascend `ascendc_pto@4c726701` 的原非分页 XA，20 个共享 Cube 核和 4 个私有 Cube 核，写回后合并 |
| `expert_v2` | [PR #1864](https://github.com/tile-ai/tilelang-ascend/pull/1864) 的实现，24 个参与核处理共享前缀，私有部分用 Vector 计算并在最后一个共享块就地合并 |

v1 只增加问题尺寸参数、harness 的符号名及输入适配，不更改原调度与计算。
两版输入输出均为 BF16，32 个 query 头共享 8 个 K/V 头，头维度 128，私有长度为 2。
v2 的中文优化历程和历史数据见 [实现说明](backends/expert_v2/README.md)；历史结果不替代当前主线实测。

## 当前主线性能

2026-10-08，使用下述原 harness 的完整 18 点测量，v1/v2 均通过正确性检查；
v2 的设备内核用时为旧版的 60.6%–97.6%，对应 1.024–1.651x 加速。
这是当前环境下相对旧版的结果，不是跨设备或所有实现的最快保证。
完整性能表、环境与复现命令见[主线对照](backends/expert_v2/mainline_comparison.md)，
原报告见[CSV](backends/expert_v2/mainline_comparison.csv)。资料与内核放在同一目录。

## 测试点与运行

沿用研究仓的 [18 个测试点](problem/cases.py)：请求数/共享长度为
`(1,1024)`、`(2,1024)`、`(2,2048)`，每请求 beam 数分别为
`128,256,512,1024,2048,4096`。标准输入由 [canonical.py](problem/canonical.py) 生成，
默认 benchmark seed 为 0。没有另选小问题替代这组矩阵。

先提供已构建的 TileLang-Ascend 主线环境；本次编译器基线为
[`4c726701`](https://github.com/tile-ai/tilelang-ascend/commit/4c7267018af02284ff28f1cd090a1492c7985bed)。
运行环境需要 Python、PyTorch、`torch_npu`、CANN/BiSheng 和已构建的 TileLang-Ascend；
本仓库不安装这些系统依赖。公开源码下载方式：

```bash
git clone https://github.com/platelett/xa.git
cd xa
git submodule update --init
```

将 `env.local.sh.example` 复制为忽略的 `env.local.sh`，配置环境脚本。
xLLM 参考的源码依赖固定在 submodule；构建方式见 [参考说明](backends/xllm/ref/README.md)。
也可通过 `XA_REFERENCE_ROOT` 复用兼容的已构建参考，不提交任何二进制。

```bash
source env.sh
npu-smi info
# 仅选择当前空闲设备；示例中的 10 不是固定要求。
python -m core.bench --backends expert_v1,expert_v2 --cases all --devices 10
```

直接使用原 `core.bench`：先按问题尺寸编译，再在同一设备上对各版本预热、
测设备内核用时，并与 xLLM 检查结果；默认测 20 次取平均。每行报告单位为微秒，
包含 `Avg_us`、正确性结果和最大误差。本地报告写到忽略的 `results/`。
框架和问题定义从研究仓原样保留，不另写计时器。

## 修改源文件

`impl_tl.py` 是可编辑源文件，`impl.py` 为生成结果；本次接入保留原始 Expert 指令，
仅适配 harness 入口。重新生成使用随实现保留的 processor：

```bash
python backends/expert_v2/preprocess.py \
  backends/expert_v2/impl_tl.py -o backends/expert_v2/impl.py
```

这是主线对照与可运行实现的公开整理，不包含研究仓的所有诊断/消融版本。
