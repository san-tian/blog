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

### 2. shared-expert append（3.2%）—— GLM + flashinfer_trtllm 双重 disable

fuse 表说"已有 `fused_moe_triton_kernels.py` 融合路径"，但 GLM-5.2 的模型代码（`glm4_moe.py:1195-1212`）对 shared-expert fusion 有硬性 disable 条件：

```python
# glm4_moe.py:1195-1212
if disable_reason is None and (
    get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mori()
):
    disable_reason = "GLM-4.5 cannot use shared experts fusion under deepep expert parallelism."
elif self.quant_config and self.quant_config.get_name() == "w4afp8":
    disable_reason = "GLM-4.5 W4AFP8 uses different quant method for routed/shared experts."

if disable_reason is not None:
    declare_load_time_override(..., {"disable_shared_experts_fusion": True})
```

flashinfer_trtllm backend 启动时还会额外自动设 disable（启动日志：`FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set.`）。

**为什么不能融合**：两层 disable 叠加。要开它只能把 `--moe-runner-backend` 换回 `triton`，但 triton + allreduce fusion + `--enable-fused-moe-sum-all-reduce` 实测只有 603 tok/s（比当前 793 慢 24%）——得不偿失。

> 📐 **数据来源**：`python/sglang/srt/models/glm4_moe.py:1195-1212`、启动日志。

### 3. QK RoPE + KV cache write（4.6%）—— KV dtype 不满足

fuse 表说"已有 `attention/utils.py` 融合路径"，但它的启用函数 `enable_fused_set_kv_buffer`（`utils.py:281`）有硬性条件：

```python
# models/utils.py:290
return (
    _is_cuda
    and pool.dtype == torch.bfloat16      # ← 要求 bf16 KV cache
    and not isinstance(pool, SWAKVPool)
    and not is_prefill_context_parallel_enabled()
    and getattr(forward_batch, "dcp_kv_mask", None) is None
)
```

GLM-5.2 serve 用 `--kv-cache-dtype fp8_e4m3`（fp8 KV cache），**`pool.dtype` 是 fp8 不是 bf16，条件不满足，融合不启用**。

**为什么不能融合**：要开它只能把 KV cache 换回 bf16——但 fp8 KV cache 是 GLM-5.2 在 B200 上省显存的关键（让 max-total-tokens 能到 5000000），换回 bf16 会 OOM 或大幅降容量。

> 📐 **数据来源**：`python/sglang/srt/models/utils.py:281-300`、serve 参数 `--kv-cache-dtype fp8_e4m3`。

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
