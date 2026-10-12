# H20 环境记录功能实跑验证

2026-10-12 06:11 UTC，在 NVIDIA H20-3e GPU 0 上运行采集器，完成真实 CUDA 延迟测量和 NCU profiling，退出码为 0。

## 产物

- [baseline.json](baseline.json)：未经修改的采集器输出，包含 `_environment`、`_scaling_factors` 和算子性能记录。
- [collection.log](collection.log)：原始 stdout/stderr。
- [ops/torch_add_comparison.py](ops/torch_add_comparison.py)：本次临时验证算子，调用 `torch.add`；两份长度为 1048576 的 float32 输入，随机种子 42。

当前环境未安装 vLLM/SGLang，所以使用独立 PyTorch CUDA 算子验证采集链路。这是环境记录功能的验证，不代表 vLLM 算子覆盖率或正式跨平台验收基线。

## 实测信息

| 项目 | 结果 |
|---|---|
| GPU / 显存 | NVIDIA H20-3e / 150111977472 bytes / 139.8 GiB |
| 驱动 / Triton | 595.45.04 / 3.6.0 |
| Torch 构建 CUDA / NCU | 12.9 / 2025.2.1.0 |
| CPU | Intel Xeon 6759P-C，2 插槽、120 物理核心、240 逻辑 CPU |
| NUMA / 进程 CPU 亲和性数量 / 虚拟化 | 2 / 240 / KVM |
| 算子延迟 | 9.53365142944209 μs |
| kernel 数 / DRAM 访存量 | 1 / 8394240 bytes |

CPU 拓扑是虚拟机操作系统可见值。性能数据来自单次采集，不用于评估环境记录功能的性能开销。

## 源码与路径说明

采集时 HEAD 为 `75902f1`，工作区包含尚未提交的 `total_memory_gib` 代码及样例改动，因此原始 JSON 中 `git_dirty` 为 `true`。这些改动随后提交为 `40b8b54`；采集代码对应这一提交。保留原始元数据，不回填后来的 commit。

`reference_chip` 保留原默认值 H800，仅用于折算系数；实际采集卡是 H20-3e。原 JSON 和日志中的 `/tmp/...` 是当时运行路径。约 7 MiB 的 NCU 二进制报告未纳入 Git，JSON 已包含解析后的指标；`report_path` 是采集机器上的历史路径，不是仓库中的文件链接。

## 复现

从仓库根目录运行下列命令。环境需具备 Torch、Triton、可用 CUDA GPU 和 NCU profiling 权限。输出到新目录，保留归档原件：

```bash
python3 tools/collect_baseline_nvidia.py \
  --repo "$PWD/validation/h20-20261012/ops" \
  --ops torch_add_comparison \
  --device 0 \
  --output /tmp/h20-environment-rerun/baseline.json \
  --report-dir /tmp/h20-environment-rerun/ncu_reports \
  --op-timeout 120
```

`--repo` 使用绝对目录以选择归档的独立验证算子。重新运行时的时间戳、Git 状态、路径和性能值会随环境改变。
