# vllm-baseline

**语言：** 中文 · [English](README.md)

采集 NVIDIA 原生算子（vLLM / CUDA kernel）的性能 baseline，供国产后端做跨平台性能验收。字段口径对齐《算子后端缺失性能基准方案参考.md》。

## 目录结构

| 路径 | 说明 |
|---|---|
| `tools/collect_baseline_nvidia.py` | 采集主程序：测 latency + 用 Nsight Compute（ncu）抓计算量/访存量/利用率 |
| `tools/baseline_shape.yaml` | 声明式算子的 shape 配置（简单算子，纯位置参数） |
| `ops/` | 自定义算子模块（复杂算子：多张量、约束张量、量化等），每个算子一个文件 |
| `tools/hardware_specs.py` | 硬件峰值算力/带宽参数 |

## 运行

```bash
# 完整采集（latency + NCU），输出 JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# 只测 latency，跳过 NCU
python tools/collect_baseline_nvidia.py --no-ncu

# 指定 ncu 报告目录（.ncu-rep 可用 ncu-ui 打开）
python tools/collect_baseline_nvidia.py --report-dir ncu_reports

# 只跑指定的一到多个算子（逗号分隔）
python tools/collect_baseline_nvidia.py --ops moe_sum,add_rms_norm

# 白名单：只跑名单内的算子；黑名单：跳过名单内的算子
# 名单可以是逗号分隔字符串，也可以是文件路径（每行一个算子名，# 开头为注释）
python tools/collect_baseline_nvidia.py --whitelist ops_whitelist.txt
python tools/collect_baseline_nvidia.py --blacklist flash_mla,megamoe
```

依赖：CUDA GPU、`vllm`、`triton`、`ncu`(Nsight Compute)。采集器优先扫描 `ops/`，其余算子回落到 `baseline_shape.yaml`；两者同名时以 `ops/` 为准。

`--ops` / `--whitelist` / `--blacklist` 可组合使用：算子被保留需同时满足「在 `--ops` 内（若指定）」「在白名单内（若指定）」「不在黑名单内」。名单中的未知算子名会打印告警但不报错。

## 新增算子

在 `ops/` 下新建一个模块，导出契约字段（`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`，复杂算子可选 `config()`）。完整契约见 `ops/__init__.py`，可参考 `ops/fused_add_rms_norm.py`、`ops/grouped_topk.py`。

## 输出格式

`{算子: {native_api, shapes: {shape: {dtype: 记录}}}}`。每条记录含（对齐参考文档符号）：

| 字段 | 含义 | 单位 |
|---|---|---|
| `T_us` | kernel 运行时间 T | us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core 实际计算量 | FLOP |
| `B_mem` | Global Memory 实际访存量 | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | 三维硬件利用率 | %（0–100）|
| `U_bottle_neck` / `bottle_neck_unit` | 瓶颈侧利用率 / 瓶颈单元 | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | 一次调用的 kernel 数 / 逐 kernel 明细 | — |
| `config` | 复杂算子的真实输入输出 shape | — |

另含一组同值冗余的兼容旧字段（`flops_*`、`util_*`、`bottleneck` 等）。

## 环境依赖与配置（H800 全量采集）

要在 H800 上尽可能跑全所有算子，除基础依赖（CUDA GPU、`vllm`、`triton`、`ncu`）外，还需配置以下项，否则对应算子会被跳过或 NCU 采不到数据：

| 依赖 / 配置 | 影响的算子 | 不配置的后果 | 如何配置 |
|---|---|---|---|
| **`nvcc` 在 `PATH` 中**（CUDA toolkit） | `fp8_einsum`、`fp8_fp4_mqa_logits`、`fp8_fp4_paged_mqa_logits` | DeepGEMM JIT 编译失败（`std::filesystem::exists(nvcc_path)` 断言），这些算子被整体跳过 | 安装 CUDA toolkit 并把 `nvcc` 加入 `PATH`（如 `export PATH=/usr/local/cuda/bin:$PATH`）；确认 `which nvcc` 有输出 |
| **MoE / FP8 的 device-specific 调优配置 json** | `fused_experts_impl`、`fused_marlin_moe_*`、`w8a8_block_fp8_matmul` | 仅告警 `Using default MoE config`，走默认配置，**不影响采集**（性能可能非最优） | 可选：放置 `vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=NVIDIA_H800.json` 等 |
| **`flash-linear-attention`（FLA）** | `chunk_gated_delta_rule_fwd`、`chunk_kda` 等 FLA 系算子 | `native()` 解析不到，算子被跳过 | 安装对应 vllm 版本依赖的 FLA；注意 import 路径随 vllm 版本不同（`vllm.third_party.flash_linear_attention...` 或 `vllm.model_executor.layers.fla...`） |
| **足够显存（大 E MoE）** | `fused_experts_impl`、`fused_marlin_moe_*` 的 e256/e512 大档 | 采集器已在 NCU 前释放主进程实参，单份权重（~22GB）可过；若显存更小仍可能 OOM | 用 `--op-timeout` 调整超时；必要时 `--blacklist` 跳过超大档 |
| **`tilelang`** | `mhc_pre`、`mhc_post` | `vllm.model_executor.layers.mhc` 的 `HAS_TILELANG_MHC` 门控为 False，`native()` 返回 None 跳过 | 安装 `tilelang`（vLLM 的 MHC kernel 依赖）；否则属预期跳过 |
| **DeepSeek-V4 / qwen4_exp 等模型子树完整** | `qsa_pre_indexer`、`qsa_select_paged_decode`、`qsa_select_paged_prefill`（`vllm.models.qwen4_exp.*`）、`compressor` | 对应 vllm 模型子树不存在或需引擎上下文，`native()` 解析不到 | qsa 系需 vllm 带 `qwen4_exp` 子树；`compressor` 为 nn.Module 需引擎 VllmConfig，无法脱离引擎单采 |
| **昇腾 / 海光等国产后端专属算子** | `group_list_cumsum`、`indexer_epilogue`、`lightning_indexer`、`kv_rmsnorm_rope_cache`、`sparse_attn_sharedkv`、`int8_einsum`、`fused_inv_rope_int8_quant` 等 | H800 **无对应 NV 原生 kernel**，`native()` 如实返回 None 跳过 | 无法在 H800 采集，属预期跳过（这些算子的验收数据来自昇腾/海光） |

说明：被跳过的算子不影响其余算子采集（每个算子独立子进程）。`fused_marlin_moe_w4a16_int4` 因逐专家纯 Python 量化很慢，已将大 E 档位裁剪到两端 token 档以避免子进程超时。FLA 系算子（`chunk_gated_delta_rule_fwd`、`chunk_kda`）与 `fused_q_kv_rmsnorm` 的 `native()` 已改为多候选路径回退，兼容 `vllm.model_executor.layers.fla` / `vllm.third_party.flash_linear_attention` 等不同 vllm 布局。
