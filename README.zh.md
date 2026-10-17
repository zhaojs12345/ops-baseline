# vllm-baseline

**语言：** 中文 · [English](README.md)

采集 NVIDIA 原生算子（vLLM / CUDA kernel）的性能 baseline，供国产后端做跨平台性能验收。字段口径对齐《算子后端缺失性能基准方案参考.md》。

## 目录结构

| 路径 | 说明 |
|---|---|
| `tools/collect_baseline_nvidia.py` | 采集主程序：测 latency + 用 Nsight Compute（ncu）抓计算量/访存量/利用率 |
| `tools/baseline_shape.yaml` | 声明式算子的 shape 配置（简单算子，纯位置参数） |
| `ops/<repo>/` | 按来源 repo 分目录的自定义算子模块（`ops/vllm`、`ops/sglang` 等），每个算子一个文件，由 `--repo` 选择（缺省 `vllm`） |
| `tools/hardware_specs.py` | 硬件峰值算力/带宽参数 |

## 运行

```bash
# 完整采集（latency + NCU），输出 JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# 只测 latency，跳过 NCU
python tools/collect_baseline_nvidia.py --no-ncu

# 指定 ncu 报告目录（.ncu-rep 可用 ncu-ui 打开）
python tools/collect_baseline_nvidia.py --report-dir ncu_reports

# 选择算子来源 repo：扫描 ops/<repo>/（缺省 vllm）
python tools/collect_baseline_nvidia.py --repo vllm
python tools/collect_baseline_nvidia.py --repo sglang
python tools/collect_baseline_nvidia.py --repo pytorch --ops mm,layer_norm

# 只跑指定的一到多个算子（逗号分隔）
python tools/collect_baseline_nvidia.py --ops moe_sum,add_rms_norm

# 白名单：只跑名单内的算子；黑名单：跳过名单内的算子
# 名单可以是逗号分隔字符串，也可以是文件路径（每行一个算子名，# 开头为注释）
python tools/collect_baseline_nvidia.py --whitelist ops_whitelist.txt
python tools/collect_baseline_nvidia.py --blacklist flash_mla,megamoe

# 只校验每个算子的 native() 能否被调用（不采集、不写 JSON）
python tools/collect_baseline_nvidia.py --check --repo vllm
```

依赖：CUDA GPU、`vllm`、`triton`、`ncu`(Nsight Compute)。采集器扫描 `ops/<repo>/`（由 `--repo` 选择，缺省 `vllm`），其余算子回落到 `baseline_shape.yaml`；两者同名时以算子模块为准。

`--repo` 决定去 `ops/` 下哪个子目录找算子（`vllm` / `sglang` …），缺省 `vllm`。`--ops` / `--whitelist` / `--blacklist` 可组合使用：算子被保留需同时满足「在 `--ops` 内（若指定）」「在白名单内（若指定）」「不在黑名单内」。名单中的未知算子名会打印告警但不报错。

PyTorch 算子集合来自固定版本的 FlagGems kernelgen-dev，覆盖范围、shape 口径、验证和待决定项见 [ops/pytorch/README.md](ops/pytorch/README.md)。

## 新增算子

在 `ops/<repo>/` 下新建一个模块（如 `ops/vllm/`，sglang 算子放 `ops/sglang/`），导出契约字段（`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`，复杂算子可选 `config()`）。完整契约见 `ops/vllm/__init__.py`，可参考 `ops/vllm/fused_add_rms_norm.py`、`ops/vllm/grouped_topk.py`。采集时用对应的 `--repo` 指向该目录。

## 输出格式

`{算子: {native_api, shapes: {shape: {dtype: 记录}}}}`。每条记录含（对齐参考文档符号）：

| 字段 | 含义 | 单位 |
|---|---|---|
| `T_us` | native 调用延迟 T（独立于 NCU）| us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core 实际计算量 | FLOP |
| `B_mem` | Global Memory 实际访存量 | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | 三维硬件利用率 | %（0–100）|
| `U_bottle_neck` / `bottle_neck_unit` | 瓶颈侧利用率 / 瓶颈单元 | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | 一次调用的 kernel 数 / 逐 kernel 明细 | — |
| `config` | 复杂算子的真实输入输出 shape | — |

另含一组同值冗余的旧字段（`flops_*`、`util_*`、`bottleneck` 等）。NCU 未测得的指标在新旧字段中统一写为 `null`；数值 `0` 只用于实际读到的零值，下游必须支持 nullable 指标。

每条记录增加 `ncu_status`、`ncu_reason`、`ncu_diagnostic`、`ncu_scope` 与 `metric_validity`（字段名 → 有效性）：

| `ncu_status` | 含义 |
| --- | --- |
| `collected` | 每个被采样 kernel 的所有已请求 counter 均读取成功。 |
| `partial` | 有 kernel，但部分 counter 缺失、非数字或非有限数；受影响的汇总字段为 `null`，不返回不完整的部分和。 |
| `no_kernel` | NCU 成功退出且明确输出 “No kernels were profiled”；仅表示选定 NVTX 范围未采到 kernel，不能据此区分 memcpy、view/no-op 和过滤条件不匹配。 |
| `failed` | 导出/导入失败、超时，或缺少有效报告/记录且没有明确的无 kernel 证据。 |
| `disabled` | 本次使用 `--no-ncu`。 |
| `unknown` | 离线迁移时历史证据不足，无法确定原始 NCU 结果。 |

`metric_validity` 的值包括：`measured`（实测）、`derived`（瓶颈利用率由实测利用率推导）、`estimated`（逐 kernel 的 `T_us` 由端到端延迟分摊）、`not_collected`（该 dtype 没有选定 Tensor Core FLOP counter）、`unavailable`（不可用）。逐 kernel 明细也有自己的有效性映射；多 kernel 缺少可用 duration 权重时不分摊耗时。未请求的 Tensor Core FLOP 为 `null`，不假定为零。

NCU 不可用不影响独立测得的 `T_us` / `latency_ms`。下游应逐记录判断：延迟比较检查 `metric_validity.T_us == "measured"`，利用率统计检查相应字段为 `measured`；禁止用 `record.get("U_mem") or 0` 填补缺失值。不要因为算子级 `profiling_incomplete` 就丢弃该算子的全部记录。保守的任务完整性策略仍将 `no_kernel`、`partial` 和 `failed` 归入 `profiling_incomplete`，这不表示已有延迟和其他实测字段无效。

旧 JSON 没有有效性字段，零值可能是占位值；本次不会自动迁移旧文件，也不能仅凭报告不存在就追认为 `no_kernel`。更新代码不会修改正在运行的采集及其输出。


输出 JSON 顶层的 `_environment` 保存本次采集的环境信息：实际 GPU 型号、UUID、显存字节数、计算能力、SM 数、CUDA 逻辑设备编号和驱动版本；CPU 信息；操作系统、Python、Torch/Triton/vLLM/SGLang、Torch 构建 CUDA、nvcc 和 NCU 版本；UTC 时间、采集器 Git commit/分支/工作区状态，以及筛选后算子列表与运行参数。`CUDA_VISIBLE_DEVICES` 和 `CUDA_DEVICE_ORDER` 也会记录。无法获取的字段为 `null`，不会因此中断采集。

元数据在首个算子运行前写入，后续增量落盘保留；内部 worker 不重复采集环境。`_scaling_factors.reference_chip` 是配置中的折算基准，实际运行 GPU 请看 `_environment.hardware.gpu`。下游遍历时应按是否具有 `shapes` 对象识别算子；`_reshape_alias_copy` 等算子名也以下划线开头。CPU 记录具体型号、架构、插槽数、物理核心数、逻辑 CPU 数、每核心线程数、NUMA 节点数、进程 CPU 亲和性允许的 CPU 数及虚拟化厂商；拓扑是操作系统可见值，虚拟机中不代表宿主机物理配置，亲和性计数不代表 cgroup CPU 配额。缺少 lscpu 时对应字段为空。驱动兼容 Torch 的裸 UUID，设备查询失败时仅接受全机一致的驱动版本；Triton 优先记录实际导入模块的版本，兼容 pytorch-triton/FlagTree 提供的安装。

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

说明：被跳过的算子不影响其余算子采集（每个算子独立子进程）。`fused_marlin_moe_w4a16_int4` 因逐专家纯 Python 量化很慢，但其 shape 已与 benchmark 完全对齐（全 32 组档位），大 E 架构可能逼近 `--op-timeout`（缺省 1800s）；实跑若子进程超时，用 `--op-timeout` 调大或 `--blacklist` 跳过最大档单独处理。FLA 系算子（`chunk_gated_delta_rule_fwd`、`chunk_kda`）与 `fused_q_kv_rmsnorm` 的 `native()` 已改为多候选路径回退，兼容 `vllm.model_executor.layers.fla` / `vllm.third_party.flash_linear_attention` 等不同 vllm 布局。

## 多卡动态采集

```bash
python tools/collect_baseline_nvidia.py --repo pytorch \
  --devices 0,1,2,3,4,5,6,7 --output pytorch_h800.json
# 使用完全相同的参数追加 --resume，跳过已完成算子，重试失败/跳过项
```

`--devices` 与 `--device` 互斥。填写 GPU 可见性 token（卡号或 UUID）；如果已有
`CUDA_VISIBLE_DEVICES`，只能选择其中的 token，不把它们重新解释为逻辑卡号。
每张卡有一个消费者，从共享队列动态领取算子；单算子仍通过原隔离采集器执行。
超时/停止会清理该任务的整个进程组，包括 NCU 子进程。各卡使用独立报告目录。

**最终结果只有 `--output` 指定的一个 JSON，不需要手动合并。** 主线程在每个任务状态
变化时原子更新该文件。现有算子字段和 `_scaling_factors` 保留；`_environment` 是首个
已返回任务的环境，`_environments` 按 GPU 保存环境，`_collection.tasks` 保存每个算子的
卡号、状态和诊断路径。不要把 `_environment` 单独解释为所有卡的环境。
`<output>.work/` 保存内部中间结果、日志和 NCU 报告，以便检查部分失败。

NCU 未取得 kernel 指标的算子标记 `profiling_incomplete`；这也可能是没有 GPU kernel
的主机/视图操作，需要查看日志，而不是视为测得零工作量。任意未完成任务使程序退出 1，
但已完成和部分结果仍保存在统一 JSON。`--no-ncu` 模式不要求 NCU 指标。
续跑要求相同算子选择、卡列表、采集参数和源码内容，防止混入不同版本的数据。
运行前确认卡空闲；NCU 多进程是否并行以及实际吞吐受驱动/NCU 版本限制。

每次重试使用独立的尝试目录，保留此前的日志、部分 JSON 和 NCU 报告。
重试失败不会删除或降低已保留算子快照的覆盖；使用旧数据时，任务中的
`retained_result` 标明原设备、产物路径和环境，`previous_attempts` 保存历史状态。
不会把不同尝试的数据逐 workload 混合；只有新快照覆盖旧 workload 及其已有 profiler
证据时才替换。续跑开始前清除终止标记，并记录 `attempt_started_at_utc`。
协调器或落盘发生异常时先停止消费者，再等待进程清理。自定义 `--report-dir`
使用完整输出路径的摘要与尝试 ID 隔离，避免不同目录的同名输出互相覆盖报告。

## 已结束采集的离线有效性补充

```bash
CUDA_VISIBLE_DEVICES= python tools/annotate_baseline_validity.py \
  --input /absolute/path/to/finished.json --workers 4
```

从独立 checkout 运行。工具要求所有选中算子均有终态，检查残留采集/NCU进程并持有原输出锁，先创建以输入哈希命名的只读原始备份，再在 CPU 上导入已有 NCU 报告；不会执行 export 或 GPU workload。能与具体记录对应的明确 NCU 失败日志标 `failed`，没有报告且缺乏明确证据的标 `unknown` / `legacy_reason_unknown`，不猜测为 `no_kernel`。

原子替换前检查算子/shape/dtype、延迟、输入配置、采集历史及环境元数据保持不变，验证严格 JSON、字段有效性、无运行中的写者，以及原始输入 SHA256 未变。`_metric_validity_migration` 记录工具/parser 版本与哈希、输入哈希、备份路径、时间和状态统计；每条记录保存报告哈希或历史证据。拒绝重复迁移。

## 单独补采与合并

`--ncu-timeout` 设置单次 NCU export 超时（默认 600 秒），与整算子的 `--op-timeout`（默认 1800 秒）独立；该值传给隔离 worker 并纳入续跑配置。定向补采可使用 `--op-timeout 14400 --ncu-timeout 1800`。

`tools/collect_baseline_supplement.py --base BASE --supplement RETRY --output FINAL --ops CSV --devices 0,1,2,3,4,5,6,7` 先等待 BASE 已启动的离线有效性补充结束（该迁移在替换前会检查容器内有无运行中的采集），再用上述放宽的默认值单独采集指定算子，最后自动合并到一个新的最终 JSON；不重启迁移，也不覆盖两个来源文件。FINAL 旁的 pipeline JSON 记录当前阶段。也可直接使用 `tools/merge_baseline_supplement.py`。

合并保留原先完整成功的记录不变，只添加缺失项或升级不丢失原有实测字段的不完整记录；输入配置不一致则拒绝合并。新增/升级记录保存补采设备与环境来源。原采集历史保持不变，`_supplemental_merge` 记录来源哈希、补采运行信息、有效任务状态与计数及最终 NCU 状态统计；补采失败仍可贡献部分数据，但不能标成成功。
