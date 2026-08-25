# sglang 算子融合到顶了吗：剩余四个融合点为什么做不了

当前使用的启动命令（GLM-5.2 step111 FP8，B200 TP8）：

```bash
python3 -m sglang.launch_server \
  --model-path /data0/models/glm52-step111-fp8 \
  --served-model-name glm52-step111-fp8 \
  --host 0.0.0.0 --port 10100 --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-cache-report --page-size 64 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend flashinfer_trtllm \
  --flashinfer-allreduce-fusion-backend auto \
  --model-impl sglang \
  --mem-fraction-static 0.90 --max-total-tokens 5000000 \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-write-policy write_back --hicache-mem-layout page_first \
  --hicache-storage-backend file --file-storage-path /root/hicache \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 --enable-metrics
```

这套配置实测 793 tok/s（比融合全关的 baseline 524 tok/s 快 +51%）。这篇用 profile 数据 + 代码排除法回答：沿这套配置继续做算子融合还能不能提升，以及那四个"看起来能融合"的点为什么实际做不了。

## TL;DR

算子融合已到顶。decode 阶段 66 个 kernel 逐一排除后，只剩 4 个"可融合且未生效"的点，但全部被硬性条件挡住：

| 融合点 | 占比 | 挡路条件 |
|---|---|---|
| embedding allreduce | 27.8% | 无下一层 layernorm 吸收（不是融合对象） |
| shared-expert append | 3.2% | GLM 模型层 + flashinfer_trtllm 双重 disable |
| QK RoPE + KV cache write | 4.6% | 要求 bf16 KV cache，GLM 用 fp8_e4m3 |
| FP8 quant + activation | 3.7% | flashinfer_trtllm 闭源 kernel 不开放 epilogue |

## 排除法：为什么只剩这四个

decode 共 66 个 kernel，占比 ≥1% 的 23 个。逐一过五类排除，只剩 4 个既是 allreduce/quant/prep 类（有融合对象）、又未生效的。

### ① 已是融合 kernel（无需再融合）

| kernel | 占比 | 说明 |
|---|---|---|
| `oneshotAllreduceFusionKernel` | 3.6% | allreduce fusion 已开，这是融合后的产物 |
| `rmsNormLamport` | 1.9% | allreduce + rmsnorm 融合的产物 |
| `twoshotAllreduceKernel` | 1.7% | 同上 |
| `fused_a_gemm`（2 变体） | 3.1% | MLA A 投影，名字带 fused |
| `RopeQuantizeKernel` | 1.0% | RoPE + quant 融合 kernel |
| `fused_*_indexer_*`（3 个） | 0.6% | DSA indexer 融合，已融合 |

### ② GEMM 计算密集，不是融合范畴

融合是"把多个小 kernel 合并成一个"，省的是 launch + 访存往返。GEMM 本身是计算密集的大 kernel，没有"和谁融合"的对象。

| kernel | 占比 | 说明 |
|---|---|---|
| `deep_gemm`（dense 层 fp8） | 11.7% | 计算密集，无相邻小 kernel 可合并 |
| `bmm_E4m3`（MoE 专家 GEMM，3 变体） | 16.9% | flashinfer trtllm MoE GEMM，已是单一大 kernel |
| `nvjet`（lm_head GEMM，3 变体） | 3.1% | lm_head bf16 GEMM，计算密集 |
| `cublasLt::splitKreduce` | 0.8% | GEMM 内部 splitK 归约 |
| `deep_gemm::sm100_paged_mqa_logits` | 0.5% | DSA paged MQA logits，计算密集 |

### ③ flashinfer trtllm MoE 内部子 kernel，闭源无法单独动

| kernel | 占比 | 说明 |
|---|---|---|
| `moe::finalize::finalizeKernel` | 2.2% | trtllm MoE 内部，闭源预编译 |
| `moe::routing::routingIndicesDynBlockKernel` | 2.0% | trtllm MoE 内部 routing，闭源 |
| `moe::activation::activationDeepSeekKernel` | 1.6% | trtllm MoE 内部 activation，闭源 |

### ④ 太小的 elementwise / framework 开销

| kernel | 占比 | 说明 |
|---|---|---|
| `vectorized_elementwise`（5 变体） | 5.8% | PyTorch 零碎 elementwise（compare/where/bitwise_and），单个 <2%，无明确融合对象 |
| `set_mla_kv_buffer_kernel` | 0.5% | MLA KV cache 写入，太小 |
| `cunn_SoftMaxForward` | 0.2% | 已融在 fmha kernel 里 |
| `topk_small_batch_kernel` | 0.2% | 太小 |

### ⑤ attention 主体已最优

| kernel | 占比 | 说明 |
|---|---|---|
| `fmhaSm100fKernel`（2 变体） | 3.2% | flashinfer fused MLA attention，已是最优 kernel |
| `rmsnormRMSNormKernel` | 1.8% | 单独 RMSNorm（非 residual+norm 融合那部分），量级小 |

### 排除后的剩余

66 个 kernel 排完五类，只剩 4 个"可融合且未生效"——就是开头表里的那四个。

> 📐 **数据来源**：`analyze_llm_torch_profile.py --input glm_prof_fusion`（TP-0 DECODE trace，2026-08-25 于 .8）。66 个 kernel 全量统计，占比 ≥1% 的 23 个，总 GPU 时间 110.7ms。

## 四个融合点为什么做不了

### 1. embedding allreduce（27.8%）—— 不是融合对象

这是 decode 头号瓶颈。通过 trace 时序分析定位，它的前驱 kernel 是 `_vocab_parallel_embedding_kernel`：

| 次数 | dur | 前驱 kernel |
|---|---|---|
| 第1次 | 26.89ms（warmup 异常） | `_vocab_parallel_embedding_kernel` |
| 第3次 | 1.73ms | `_vocab_parallel_embedding_kernel` |
| 第5次 | 2.10ms | `_vocab_parallel_embedding_kernel` |

这是 **TP8 下 vocab embedding 的 allreduce**：每个 rank 只算自己那部分 vocab 的 embedding，然后 allreduce 把 8 个 rank 的 partial 求和。

**为什么不能融合**：allreduce fusion 的原理是"allreduce + 下一层的 residual + rmsnorm 融合成一个 kernel"。但 embedding 后面直接是第一层 attention，**没有 layernorm 来吸收这个 allreduce**——它不是 residual+rmsnorm 的融合对象，是 TP vocab parallel embedding 的固有开销。

> 📐 **数据来源**：TP-0 DECODE trace 的 kernel 时序分析，`all_reduce_kernel` 事件的前驱 kernel 均为 `_vocab_parallel_embedding_kernel`。

### 2. shared-expert append（3.2%）—— flashinfer_trtllm kernel 没实现

fuse 表说"已有 `fused_moe_triton_kernels.py` 融合路径"，但那条路径只对 triton backend 有效。flashinfer_trtllm backend 的 MoE kernel **本身不支持 shared expert 融合**，代码里有一个硬 assert：

```python
# moe_runner/flashinfer_trtllm.py:1194
assert (
    runner_config.num_fused_shared_experts == 0
), "Fused shared experts are not supported for flashinfer trtllm moe"
```

所以 sglang 在启用 flashinfer_trtllm 时会**预先**把 `disable_shared_experts_fusion=True`（`arg_groups/overrides.py:2027`，日志 `FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set.`），避免走到那个 assert 崩溃。

> 注：GLM 模型层（`glm4_moe.py:1183-1212`）也有一套 shared-expert disable 条件，但在 B200（sm100）+ TP8（无 EP）+ fp8_w8a8（非 w4afp8）下，这些条件**都不命中**——真正挡路的是 flashinfer_trtllm 那个 assert，不是 GLM 模型层。

**为什么不能融合**：这是 flashinfer 预编译闭源 kernel 的能力缺失（没实现），不是配置冲突。要开它只能把 `--moe-runner-backend` 换回 `triton`（triton 的 fused_moe 支持共享专家融合），但 triton + allreduce fusion + `--enable-fused-moe-sum-all-reduce` 实测只有 603 tok/s（比当前 793 慢 24%）——得不偿失。

> 📐 **数据来源**：`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:1194-1195`（assert）、`python/sglang/srt/arg_groups/overrides.py:2027`（预 disable）、`python/sglang/srt/models/glm4_moe.py:1183-1212`（GLM 层条件，B200 下不命中）。

### 3. QK RoPE + KV cache write（4.6%）—— 能实现，但量化逻辑不等价会损精度

fuse 表说"已有 `attention/utils.py` 融合路径"。底层 triton kernel 有 fp8 路径，中间层 arg 构造的 CUDA 分支没接上（gate 也只放行 bf16）。改 3 处源码后能跑通：

1. gate `enable_fused_set_kv_buffer` 放行 fp8 dtype
2. arg 构造 `create_fused_set_kv_buffer_arg` CUDA 分支处理 fp8 scale（返回 dict，参考 ROCm 分支）
3. fallback 路径 `base.py:386` 去掉 `_is_hip` 限制（让 CUDA fallback 也走 `fused_qk_rope_reshape_and_cache`）

实测结果（同 bench 参数）：

| 指标 | 改前 | 改后 | 差异 |
|---|---:|---:|---:|
| 输出吞吐 (tok/s) | 793.20 | 802.78 | +1.2% |
| TPOT 均值 (ms) | 17.00 | 16.92 | -0.5% |
| TTFT 均值 (ms) | 3254.78 | 3090.67 | -5.0% |

**但会损失精度**——融合 kernel 的 fp8 量化和 DSA 专用路径不等价：

| | 非融合（DSA `quantize_k_cache`） | 融合（triton `fused_qk_rope_reshape_and_cache`） |
|---|---|---|
| nope 量化粒度 | per-block（128 一组，每组一个 scale） | 单个 `k_scale`（整个 token 一个 scale） |
| rope 部分 | **不量化**，保留 bf16 | **也量化**成 fp8 |
| 输出 layout | `[nope_fp8 \| scales \| rope_bf16]` | 统一 fp8（layout 不同） |

最关键的损失在 rope 部分：非融合路径保留 rope 为 bf16（位置编码精度敏感），融合路径把 rope 也量化成 fp8，会损失位置信息。简单输入下看不出（"1+1=2"正确），但复杂输入下 DSA 的 sparse attention 依赖精确 KV 做 topk 选择，rope 精度损失可能导致 topk 选错。

**无损实现不可行**（在现有融合 kernel 上改）。下面逐层展开三重不兼容，并补充需要的背景知识。

#### 背景知识：MLA 的 K 和 fp8 KV cache

GLM-5.2 用 MLA（Multi-head Latent Attention），每个 token 的 K 分成两部分：

- **k_nope（512 维）**：不带位置编码的部分，是"内容"
- **k_rope（64 维）**：带 RoPE 位置编码的部分，是"位置"

RoPE 只作用于 k_rope 的 64 维，k_nope 的 512 维不动。

fp8 KV cache 是把 bf16 的 K 压缩成 fp8 存（省一半显存）。但 fp8 精度低（只有 8 位），直接全压会丢信息。DSA 的做法是**分区处理**：

- k_nope（512 维）：按 128 维一组切 4 块，每块单独算一个 scale，量化成 fp8（per-block 量化，精度更高）
- k_rope（64 维）：**不量化**，保留 bf16（位置编码精度敏感，不能压）

所以 DSA 的 KV buffer 里每个 token 存的是**混合 dtype**：

```
[ k_nope 的 fp8 量化值（512 字节） | 4 个 fp32 scale（16 字节） | k_rope 的 bf16 原值（128 字节） ]
```

fp8 和 bf16 在同一个 buffer 里混排，这就是"混合 dtype buffer"。

#### 背景知识：paged KV buffer

sglang 的 KV cache 用 paged layout：把显存分成固定大小的 page（page_size=64 token），每个 page 存一批 token 的 KV。buffer 的 dtype 是统一的——一个 `torch.Tensor` 只能有一种 dtype。DSA 的混合 dtype 是通过把 fp8 和 bf16 **以字节形式拼在一起、用 uint8 view 存储**实现的（`output.view(self.store_dtype)`，store_dtype 是 fp8，但 rope 部分的字节其实是 bf16 的 raw bytes）。

#### 不兼容 1：buffer 结构——混合 dtype vs 统一 dtype

现有融合 kernel（`fused_qk_rope_reshape_and_cache`）写的是**统一 dtype 的 paged buffer**：整个 key（nope+rope）用同一个 dtype（fp8）写入，一个 `tl.store` 搞定。

DSA 要的是**混合 dtype**：nope 写 fp8 + scale，rope 写 bf16，三者拼在一个 buffer 里。融合 kernel 的 `tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))` 把整个 key 转成 buffer 的 dtype（fp8）写入——**它不知道 rope 部分应该保留 bf16**，也不知道 scale 该写在哪。

这不是加个 if 能解决的——混合 dtype 的字节布局（fp8 | scale | bf16 拼排）和统一 dtype 的 paged 布局（全是 fp8）是两种完全不同的内存结构。

#### 不兼容 2：量化 block 结构——per-block vs single scale

DSA 的 nope 量化是 **per-block**：把 512 维切成 4 个 128 维的 block，每个 block 独立算 scale（`y_s = tl.max(tl.abs(y)) / FP8_MAX`）。这样每个 block 的动态范围独立，精度更高。

现有融合 kernel 用 **single scale**：整个 token 的 key 用一个外部传入的 `k_scale`（`k_pe = k_pe * (1 / k_scale)`）。一个 scale 覆盖 576 维，精度低。

更重要的是 **block 的划分维度不同**：

- DSA 量化 kernel 的 block 是按 **128 维的 K 维** 切的（`GROUP_SIZE=128`，program_id 按 token × block 切）
- 融合 kernel 的 block 是按 **head × dim** 切的（`pid_t × pid_hk`，做 RoPE 的基本单元）

融合 kernel 在一个 block 里处理一个 head 的 RoPE，而 DSA 量化在一个 block 里处理 128 维的 scale 计算——两者的 block 含义不同，**没法在同一个 block 里同时做 RoPE 和 per-block 量化**。要改 block 结构让它同时满足两种切法，等于重写 kernel 的并行策略。

#### 不兼容 3：要无损 = 写全新 kernel

三重不兼容叠加后，唯一出路是**从头写一个 DSA 专用的融合 triton kernel**，在一个 kernel 里同时完成三件事：

1. 对 k_rope（64 维）做 RoPE，结果保留 bf16
2. 对 k_nope（512 维）按 128 group 做 per-block fp8 量化，算 scale
3. 把 fp8 量化值 + scale + bf16 rope 按混合 layout 写入 paged buffer

这不是改现有 kernel 的几行代码，是设计一个新的并行策略（block 怎么切才能同时覆盖 RoPE 的 head 维和量化的 128 维 group），写新的 triton kernel，然后验证正确性。中大型工程。

> 📐 **数据来源**：非融合路径 `python/sglang/kernels/ops/attention/dsa/quant_k_cache.py:133-175`（`_quantize_k_cache_fast`，rope 保留 bf16）+ `:267-330`（kernel 实现，per-block scale，`y_s = tl.max(tl.abs(y)) / FP8_MAX`）；融合路径 `python/sglang/kernels/ops/kvcache/rope_cache.py:340-372`（`HAVE_K_SCALE`，rope 也量化）；`python/sglang/srt/mem_cache/memory_pool.py:4213`（`rope_storage_dtype = torch.bfloat16`）+ `:4166-4172`（`quantize_k_cache` 调用，混合 layout）；实测 .8 2026-08-25。

### 4. FP8 quant + activation（3.7%）—— trtllm 闭源不开放 epilogue

`per_token_group_quant_8bit_v2_kernel`（3.7%）是 MoE 专家 GEMM 前的 FP8 per-token 量化。catalog 里 vLLM 有 `fuse_act_quant` pass（SiLU+mul + FP8 quant 融合），sglang 的 triton fused_moe 里有类似 epilogue，但 **flashinfer_trtllm backend 的 MoE kernel 是闭源预编译 kernel，不开放 epilogue 定制**——没法把量化融进去。

**为什么不能融合**：要开它只能把 `--moe-runner-backend` 换回 `triton`（triton 的 `fused_moe` 有融合 epilogue），但 triton 实测只有 603 tok/s（比当前 793 慢 24%），同样得不偿失。

> 📐 **数据来源**：profile 里 `per_token_group_quant` 的 python location（`fused_moe.py:232`，走 triton path 才有融合 epilogue）。

## 总结

| 融合点 | 占比 | 能不能动 | 原因 |
|---|---|---|---|
| embedding allreduce | 27.8% | ❌ | 不是融合对象（无 layernorm 吸收） |
| shared-expert append | 3.2% | ❌ | GLM + flashinfer_trtllm 双重 disable，换 triton 慢 24% |
| QK RoPE + KV cache write | 4.6% | ❌ | 要求 bf16 KV cache，GLM 用 fp8_e4m3 |
| FP8 quant + activation | 3.7% | ❌ | trtllm 闭源不开放 epilogue，换 triton 慢 24% |

四个剩余融合点全部被硬性条件挡住，且都与"换回 triton backend"或"换 KV dtype"绑定，代价都大于收益。**这套配置的算子融合已到顶。**

要继续提升性能，方向不再是"算子融合"：
1. 减少 embedding allreduce（`--enable-attn-tp-input-scattered`，需实测 MLA/DSA 兼容性）
2. 换更大 batch 摊薄通信开销（27.8% 的 allreduce 在大 batch 下占比会降）
3. 等 flashinfer_trtllm 后续版本开放 shared-expert fusion / epilogue 定制

## 完整环境

| 组件 | 版本 |
|---|---|
| 硬件 | 8× NVIDIA B200（179G/卡），`yotta-arc-b200-8` |
| 镜像 | `b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200`（digest `1839d8f3c89b`） |
| sglang | `sgl-project/sglang` release/v0.5.15，HEAD `0b3bb0c` |
| torch | 2.11.0+cu130 |
| flashinfer | python 0.6.14 / cubin 0.6.14 / jit-cache 0.6.14+cu130（已对齐） |
| 模型 | `/data0/models/glm52-step111-fp8`（E=257 MoE，topk=8，hidden=6144，fp8_w8a8，MLA + DSA） |
| serve 关键参数 | `--moe-runner-backend flashinfer_trtllm --flashinfer-allreduce-fusion-backend auto --kv-cache-dtype fp8_e4m3 --tp 8` |
| benchmark | `bench_serving --random-input-len 8192 --random-output-len 1024 --num-prompts 64 --max-concurrency 16`，793 tok/s |

> 📐 **数据来源**：本报告所有代码引用均来自 `sgl-project/sglang` release/v0.5.15（HEAD `0b3bb0c`），profile 数据来自 .8 实测（2026-08-25）。
