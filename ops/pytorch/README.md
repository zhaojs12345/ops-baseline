# ops/pytorch

存放 PyTorch 原生算子基准模块（方案 B）。目录结构与契约同 `ops/vllm`
（见 `ops/vllm/__init__.py`）：每个算子一个 `ops/pytorch/<OP_NAME>.py`，导出
`OP_NAME / DTYPES / IS_INPLACE / native() / grid() / build_inputs() / key_shape()`
（复杂算子再加 `config()`）。

与 `ops/vllm`、`ops/sglang` 的区别在于 **native 来源**：这里 `native()` 绑定的是
PyTorch 官方算子（`torch.*` / `torch.nn.functional.*` / `torch.ops.aten.*`），
即 FlagGems 的对照基线侧，而非某个推理框架的自定义 kernel。

算子清单与每个算子的 native 坐标 / shape / 输入构造，以 FlagGems 仓库为来源：

- 清单：`FlagGems/conf/operators.yaml`（`id` / `for` / `labels` / `stage`）
- benchmark：`FlagGems/benchmark/test_<op>.py`（`torch_op=` 即 native，
  `input_fn` 即 `build_inputs` 的依据，`dtypes=` 即 `DTYPES`）
- shape：`FlagGems/benchmark/core_shapes.yaml`（按 `op_name` → 基准类名 → 默认值
  三级回退，见 `benchmark/base.py:set_shapes`）

采集时用 `--repo pytorch` 指向本目录：

    python3 tools/collect_baseline_nvidia.py --repo pytorch [--ops ...] [--device N]

缺省 `--repo vllm` 走 `ops/vllm`。

## kernelgen-dev 扩充集合

固定来源：[FlagGems kernelgen-dev `9d57a23b4386bd669152a935fe88a96db907da7c`](https://github.com/flagos-ai/FlagGems/tree/9d57a23b4386bd669152a935fe88a96db907da7c)。新增 813 个模块，加上原有 `silu` 共 814 个模块（FP16 FFT 单列）。来源涵盖 pointwise、reduction、BLAS、索引、归一化、池化、卷积、RNN、FFT、linalg、部分稀疏接口和明确的原生 backward API。

- `_kernelgen/manifest.json` 逐算子记录原生表达式、benchmark 测试函数/行号、shape 来源、dtype 和采集点数量。
- `_kernelgen/cases.json` 保存从源输入构造枚举出的描述符、调用变体和各 dtype 的输入元数据，不保存张量数据。
- `_kernelgen/` 包提供输入/契约复用，不依赖 FlagGems 或 pytest，也不会被采集器当作算子扫描。
- [PENDING.md](PENDING.md) 和 [coverage.json](coverage.json) 记录未纳入项、原因和待决定问题。

### 输入和 shape 口径

基础档位取源 `set_shapes` 的算子名 → 类 MRO → 默认值回退；显式默认描述符和输入函数里固定的几何同样沿用来源。固定几何只记录一次，使用真实首张量 shape 标识；没有把无效的 YAML 描述符重复附加到同一组输入。`COMPREHENSIVE` 额外 shape 不在本次范围，基础档位本身的调用变体全部保留。真实各位置参数/关键字参数的 shape、dtype、stride、标量配置保存在 JSON `config.inputs`。

多个调用变体用 `caseN` 区分；嵌套描述符使用稳定的短哈希键，完整描述符仍保存在 config，避免 NCU 文件名超长。单一变体的普通 shape 使用既有列表主键形式。`native_api` 继续沿用公共采集器的 `ops.<module>` 约定，真实接口请看模块 docstring/manifest。

少数源输入函数随机选择标量参数（padding、dim、steps、index）。这 7 个模块使用依赖算子/描述符/变体/dtype 的稳定 seed，使元数据、latency 与 NCU 重建参数一致；构造后恢复 Python 和 CPU RNG 状态，CUDA 张量数据仍按来源随机生成。共享 `generate_tensor_input` 的 bool 输入显式指定 bool dtype，以避免源 helper 默认生成 int64、却以 bool dtype 记录的问题。其余支持范围由源构造和本机实际 CUDA 验证决定；明确的 dtype 排除见 coverage.json。

### 运行与验证

建议先选择一个集合采集，例如：

```bash
python3 tools/collect_baseline_nvidia.py --repo pytorch \
  --ops mm,addmm,bmm,index_select,layer_norm --device 0 \
  --output pytorch_baseline.json --report-dir pytorch_ncu_reports
```

不传 `--ops` 将扫描整个集合。保留了来源中的大 shape，完整集合采集可能较长；已有 `--whitelist` / `--blacklist` / `--op-timeout` 仍可使用。

```bash
# CPU/meta 上验证所有新增描述符、dtype、变体、主键和 config 的一致性
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v

# CUDA 上：每个算子选择最小非空来源几何，检查该几何全部变体和 dtype；
# 检查元数据、重复调用、位置输入不变及实际 CUDA kernel。
# 限制只作用于验证探针，不会缩小正式采集 grid。
python3 tools/validate_pytorch_ops.py --device 0 \
  --max-input-elements 134217728 --output /tmp/pytorch_validation.json
```

旧的 `_smoke_ops.py` 用简化 torch stub，不能模拟本集合所有复杂输入；以真实 Torch 的契约测试和 CUDA 探针为准。CUDA 探针是小档位验证，并不声称所有大 shape 均完成 latency/NCU 采集；meta 无实现的数据依赖/低层接口另由 CUDA 探针验证。完整计时与 NCU 仍通过原采集器运行。

### 自定义构造函数复查

原清单中 78 条 `custom __init__ needs manual adaptation` 已全部复查：44 条来源记录完成适配，其余 34 条保留实际依赖/输入对象/计时边界/原生接口可用性原因，不再把自定义 `__init__` 本身当作阻塞。详细结果在 `coverage.json.constructor_review`。

新增的 concat/cat/stack 家族保留来源 shape 和 `MAX_ELEMENTS` 过滤；einsum 保留矩阵与批矩阵两个方程；multi-margin-loss 家族保留 p、weighted、reduction 的全部 12 种组合及输出/梯度缓冲区形状。`convert_weight_to_int4pack` 只使用配对输入工厂中的 native 参数，与来源 `_run_metric` 相同。

`to_copy` 的多个源 dtype 使用 `to_copy_from_float32` 等 7 个身份区分，DTYPES 仍表示实际目标 dtype，完整转换对沿用来源循环；`type_as` 沿用来源默认 shape 和不同 dtype 的配对。这些分组方式避免覆盖或误标多个同名构造函数的配置。

### NVIDIA 来源集合比对

按固定来源的 NVIDIA VendorDescriptor 能力和 CORE 档位选取输入。`topk` 使用 NVIDIA 的 14 个 shape 和 FP16/FP32/BF16，不再使用 Ascend 的 32 个 FP32 cases。capability 控制的 BF16/FP64/complex128/int64/FP8 组合保留在对应模块中；CSR 验证包含 int32 和 int64 索引。

`fft_rfft` 保留 FP32/FP64 的完整 case；`fft_rfft_float16` 单列 FP16 的 power-of-two 子集（28 个 case）。这是因为当前采集契约对每个 shape 遍历全部 DTYPES，两种 grid 不能直接混为一个矩形集合。两模块合起来覆盖 NVIDIA 来源的 RFFT 类型和 case，manifest 记录 `source_op` 与 `source_dtype_partition`。

以下工具从本地 FlagGems git 仓库导出 manifest 固定提交，不切换其工作区。工具使用源码的 VendorDescriptor、benchmark 构造函数、CORE shape 选择和输入迭代器；只替换候选后端和计时集成，并用 meta 张量比较输入结构多重集合。需安装 Torch、pytest 和 PyYAML；不需要安装 FlagGems。

```bash
python3 tools/compare_pytorch_source_cases.py \
  --source-repo /path/to/FlagGems --output /tmp/source_parity.json

# 把本轮修复的 NVIDIA 来源比对纳入测试（不设置时该集成测试跳过）
FLAGGEMS_SOURCE_REPO=/path/to/FlagGems \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

`--ops topk,ger` 可缩小比较范围。发现任何差异或构造错误时退出码为 1，不会静默放过已知适配。全目录仍有 10 个模块的有意差异：7 个随机标量参数固定的模块、weight-norm FP16 saved-norm 的 meta/CUDA dtype 对齐，以及 `_cast_Int` / `conj_physical` 的 no-op 类型排除。具体说明见 `coverage.json.source_adaptations`。因此不将全目录描述为与来源完全一致；结构相同也不代表全部 case 已完成数值或 CUDA/NCU 验证。

### cuDNN grid sampler 与稀疏 mask

`cudnn_grid_sampler` 保留来源的 6 组 input/grid 描述符与 FP32/FP16/FP64。
`sparse_mask` 保留 4 个方阵、FP16/FP32/BF16，以及小尺寸 10%、大尺寸 2% 的
Bernoulli 非零概率；实际 nnz 并不等于概率乘元素数。为保证 latency 与 NCU
子进程处理同一个稀疏工作负载，mask 使用独立 CPU generator，seed 为
`20261019 + rows`，不修改全局 RNG。`config.mask` 保存实际 nnz、COO 索引
SHA256、seed 和布局；重建发现 RNG/版本导致索引变化时明确报错，不复用旧身份。
稀疏索引在计时前搬到目标设备，计时仍只调用原生 ATen 算子。

该 mask 是来源分布的一份固定样本，不代表已有 FlagGems 随机 case 的精确输入。
跨后端比较必须重放同一索引并核对完整 shape/dtype/layout/config；这里没有增加
FlagGems adapter 或跨机正确性 golden。采集文件的 `_environment` 标识实际 GPU，
H20 验证数据不能当作 H800 数据。
