# sglang 算子融合：从天花板到突破，写新 kernel 无损实现 +63%

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

这套配置实测 793 tok/s（比融合全关的 baseline 524 tok/s 快 +51%）。profile 排除法发现 decode 66 个 kernel 里只剩 4 个"可融合且未生效"的点，其中 QK RoPE + KV cache write（4.6%）被"三重不兼容"挡住。但写一个 DSA 专用融合 triton kernel 后无损突破，端到端 **856 tok/s（+63%）**。

完整性能链条：

| 配置 | 输出吞吐 (tok/s) | TPOT (ms) | vs baseline |
|---|---:|---:|---:|
| baseline（融合全关） | 524.62 | 26.72 | — |
| + allreduce fusion + flashinfer_trtllm | 793.20 | 17.00 | +51% |
| + QK RoPE 有损融合（改 gate） | 802.78 | 16.92 | +53% |
| **+ QK RoPE 无损融合（新 kernel）** | **856.64** | **15.61** | **+63%** |

> 📐 **数据来源**：`bench_serving`（8192 input + 1024 output, 64 prompts, concurrency 16），.8 容器 2026-08-25。

## 排除法：为什么只剩这四个

decode 共 66 个 kernel，占比 ≥1% 的 23 个。逐一过五类排除，只剩 4 个既是 allreduce/quant/prep 类（有融合对象）、又未生效的。

### 怎么 profile 出这张 kernel 表（可复现）

「66 个 kernel」来自 torch profiler 采样 + 一段内联统计。完整流程四步：

**① 触发采样**（sglang serve 的 torch profiler 端点，采 5 步、按 stage 分）：

```bash
curl -s -X POST http://127.0.0.1:10100/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir":"/tmp/glm_prof_fusion","num_steps":5,"activities":["GPU","CPU"],"profile_by_stage":true,"with_stack":true,"record_shapes":false,"profile_id":"glm-fusion","profile_prefix":"glm"}'
```

**② 打 decode 流量**（短输入 + 长输出，让 decode 阶段跑满采样窗口）：

```bash
for i in $(seq 1 6); do
  curl -s -m 180 http://127.0.0.1:10100/generate -H "Content-Type: application/json" \
    -d '{"text":"Explain quantum computing in depth.","sampling_params":{"max_new_tokens":180,"temperature":0.6}}' >/dev/null 2>&1 &
done
```

**③ 拉回 TP-0 的 DECODE trace**：

```bash
ssh root@38.255.28.8 'docker cp sglang-fusion-b8:/tmp/glm_prof_fusion /tmp/glm_prof_fusion_out'
scp root@38.255.28.8:/tmp/glm_prof_fusion_out/glm-glm-fusion-TP-0-DECODE.trace.json.gz \
    /home/dev/Deployment/analysis/glm_prof_fusion/
```

**④ 数 kernel**（`66` = 去重后的 GPU kernel 名数量）：

```python
import gzip, json
from collections import Counter
with gzip.open('/home/dev/Deployment/analysis/glm_prof_fusion/glm-glm-fusion-TP-0-DECODE.trace.json.gz') as f:
    tr = json.load(f)
evs = tr.get('traceEvents', tr if isinstance(tr, list) else [])
total = 0; c = Counter()
for e in evs:
    d = e.get('dur',0)
    if d > 0 and e.get('cat') in ('kernel','gpu'):
        total += d; c[e.get('name','')[:70]] += d
print(f'decode 总 GPU 时间: {total/1000:.1f}ms, 共 {len(c)} 个不同 kernel')
print('=== 全部 kernel（按占比降序）===')
cum = 0
for n, d in c.most_common():
    pct = d/total*100
    cum += pct
    print(f'{pct:5.1f}%  {d/1000:7.2f}ms  {n}{"  <<< 占比>=1%" if pct>=1 else ""}')
    if cum > 99: break
```

输出第一行 `decode 总 GPU 时间: 110.7ms, 共 66 个不同 kernel`，接着是全部 kernel 按占比降序、`>=1%` 的加标记。

**统计口径三个关键点**：

- **`66` 是去重后的 kernel 名数量，不是 launch 次数**——同一个 `bmm_E4m3_...` launch 了 225 次也只算 1 个（launch 总数要上千）。
- **按 `name[:70]` 分组，模板实例算不同 kernel**——`bmm_E4m3_..._t128x8x128u2` 和 `bmm_Bfloat16_...` 是同一 kernel 的不同模板实例，name 字符串不同就各算一个（不同实例 = 不同 CUDA kernel）。
- **采样不是全量**——`num_steps=5` 只采了 5 步，`66` 是采样窗口内出现的 distinct kernel；采样步数或请求 pattern 变了，数字会变。

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

> 📐 **数据来源**：TP-0 DECODE trace（`glm-glm-fusion-TP-0-DECODE.trace.json.gz`，.8 容器 2026-08-25，采样 5 步）。「66 个 kernel / 110.7ms / 占比≥1% 的 23 个」由上文的「④ 数 kernel」脚本统计得到；`analyze_llm_torch_profile.py`（`.claude/skills/llm-torch-profiler-analysis/scripts/`）输出的是聚合三表，不含 distinct kernel 计数。

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

### 3. QK RoPE + KV cache write（4.6%）—— 写新 kernel 无损实现

fuse 表说"已有 `attention/utils.py` 融合路径"。底层 triton kernel 有 fp8 路径，但中间层 arg 构造的 CUDA 分支没接上（gate 也只放行 bf16）。改 3 处源码后能跑通，但融合 kernel 的量化逻辑和 DSA 专用路径不等价——会损精度。

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

#### 有损版本（改 gate，+1.2%）

改 3 处源码（gate 放行 fp8 + arg 构造处理 scale + fallback 路径去掉 `_is_hip`）后能跑通，但融合 kernel 把整个 key（含 rope）统一量化成 fp8，而 DSA 专用路径保留 rope 为 bf16。实测 803 tok/s（+1.2%），但 rope 精度损失会影响复杂输入。

#### 无损版本（写新 kernel，+8%）

写了一个 DSA 专用的融合 triton kernel（`fused_dsa_quant_store`），在一个 kernel 里同时完成：

1. 对 k_nope（512 维）按 128 group 做 per-block fp8 量化（每个 block 独立 scale）
2. 对 k_rope（64 维）保留 bf16 不量化
3. 三个 dtype 的数据（fp8 + fp32 scale + bf16）按混合 layout 直接写入 paged KV buffer

**混合 dtype 写入的解法**：传 3 个 dtype 不同的指针（fp8/fp32/bf16），各指向同一 paged buffer 的不同字节偏移——nope 指针写 fp8 量化值（offset 0-511）、scale 指针写 fp32（offset 512-527）、rope 指针写 bf16（offset 528-655），绕过"一个 tensor 只能一种 dtype"的限制。

替代原来的 `quantize_k_cache_separate()` + `set_mla_kv_buffer_triton()` 两步，消除中间 tensor 分配 + 额外 kernel launch。

**正确性验证**：standalone test 对比融合 kernel 和非融合路径的输出，128 token、字节级对比 nope（fp8）/ scale（fp32）/ rope（bf16）三部分全部一致——**完全无损**。

**端到端数据**：

| 指标 | 改前（793） | 无损新 kernel | 差异 |
|---|---:|---:|---:|
| 输出吞吐 (tok/s) | 793.20 | **856.64** | **+8.0%** |
| TPOT 均值 (ms) | 17.00 | 15.61 | -8.2% |
| TTFT 均值 (ms) | 3254.78 | 3150.11 | -3.2% |

无损版本比有损版本还快 6.7%（856 vs 803），因为新 kernel 省了 `quantize_k_cache_separate` 的中间 tensor 分配 + `set_mla_kv_buffer_triton` 的额外 launch。

> 📐 **数据来源**：standalone test（`fused_dsa_quant_store.py` test_correctness，128 token，字节级对比）；端到端 `bench_serving`（8192 input + 1024 output, 64 prompts, concurrency 16），.8 容器 2026-08-25。新 kernel 代码 `python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py`，接入点 `python/sglang/srt/mem_cache/memory_pool.py:3951`（`_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支）。

### 4. FP8 quant + activation（3.7%）—— trtllm 闭源不开放 epilogue

`per_token_group_quant_8bit_v2_kernel`（3.7%）是 MoE 专家 GEMM 前的 FP8 per-token 量化。catalog 里 vLLM 有 `fuse_act_quant` pass（SiLU+mul + FP8 quant 融合），sglang 的 triton fused_moe 里有类似 epilogue，但 **flashinfer_trtllm backend 的 MoE kernel 是闭源预编译 kernel，不开放 epilogue 定制**——没法把量化融进去。

**为什么不能融合**：要开它只能把 `--moe-runner-backend` 换回 `triton`（triton 的 `fused_moe` 有融合 epilogue），但 triton 实测只有 603 tok/s（比当前 793 慢 24%），同样得不偿失。

> 📐 **数据来源**：profile 里 `per_token_group_quant` 的 python location（`fused_moe.py:232`，走 triton path 才有融合 epilogue）。

## 总结

| 融合点 | 占比 | 状态 | 原因 |
|---|---|---|---|
| embedding allreduce | 27.8% | ❌ 不能 | 不是融合对象（无 layernorm 吸收） |
| shared-expert append | 3.2% | ❌ 不能 | flashinfer_trtllm 闭源 kernel 没实现 |
| **QK RoPE + KV cache write** | **4.6%** | **✅ 已实现** | **写新 triton kernel 无损突破，+8%** |
| FP8 quant + activation | 3.7% | ❌ 不能 | trtllm 闭源不开放 epilogue |

### QK RoPE 融合怎么实现的

有损版本和无损版本都实现了，区别在于量化逻辑是否和 DSA 专用路径等价。

#### 有损版本（改 3 处 gate，803 tok/s，+1.2%）

改 3 处 sglang 源码，让现有融合 kernel（`fused_qk_rope_reshape_and_cache`）的 fp8 路径走通：

| 改动 | 文件 | 内容 |
|---|---|---|
| gate 放行 fp8 | `python/sglang/srt/models/utils.py:294` | `enable_fused_set_kv_buffer` 的 CUDA 分支从 `pool.dtype == torch.bfloat16` 改成 `pool.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)` |
| arg 构造处理 scale | `python/sglang/srt/models/utils.py:316` | `create_fused_set_kv_buffer_arg` 的 CUDA 分支，当 `layer.k_scale is not None` 时返回 dict（参考 ROCm 分支），不再 assert scale is None |
| fallback 路径去掉 `_is_hip` | `python/sglang/srt/layers/rotary_embedding/base.py:386` | `if fused_set_kv_buffer_arg is not None and _is_hip:` 改成 `if fused_set_kv_buffer_arg is not None:`，让 CUDA fallback 也走 `fused_qk_rope_reshape_and_cache` |

**为什么损精度**：现有融合 kernel 用单个 `k_scale` 把整个 key（含 rope）统一量化成 fp8，而 DSA 专用路径对 nope 做 per-block 量化 + rope 保留 bf16。rope 被量化会损失位置编码精度。

#### 无损版本（写新 triton kernel，856 tok/s，+8%）

写了一个 DSA 专用的融合 triton kernel（`fused_dsa_quant_store`），在一个 kernel 里同时完成三件事：

1. **k_nope（512 维）per-block fp8 量化**：按 128 维一组切 4 块，每块独立算 scale（`y_s = max(abs(y)) / FP8_MAX`），量化成 fp8——和 DSA 原来的 `quantize_k_cache_separate` 逻辑完全一致，保证无损。
2. **k_rope（64 维）保留 bf16**：直接拷贝，不量化——位置编码精度敏感，DSA 原来也是这么做的。
3. **混合 dtype 写入 paged KV buffer**：DSA 的 KV buffer 是混合 layout（`[fp8(512) | fp32 scale(16) | bf16(128)]` 拼在一个 buffer 里），解法是传 3 个 dtype 不同的指针（fp8 / fp32 / bf16），各指向同一 paged buffer 的不同字节偏移，绕过"一个 tensor 只能一种 dtype"的限制。

替代原来的 `quantize_k_cache_separate()` + `set_mla_kv_buffer_triton()` 两步，消除中间 tensor 分配 + 额外 kernel launch。

#### 代码位置

| 版本 | 分支 | 文件 | 说明 |
|---|---|---|---|
| 有损版 | [`pex/qk-rope-fusion-lossy`](https://github.com/MindLab-Research/sglang/tree/pex/qk-rope-fusion-lossy) | `python/sglang/srt/models/utils.py`（gate + arg 构造）<br>`python/sglang/srt/layers/rotary_embedding/base.py`（fallback 路径） | 改 3 处现有代码，让 `fused_qk_rope_reshape_and_cache` 的 fp8 路径走通 |
| 无损版 | [`pex/qk-rope-fusion-lossless`](https://github.com/MindLab-Research/sglang/tree/pex/qk-rope-fusion-lossless) | `python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py`（新 kernel）<br>`python/sglang/srt/mem_cache/memory_pool.py:3951`（接入点） | 新写 triton kernel + 替换 `_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支 |

> 📐 **数据来源**：有损版实测 802.78 tok/s（改 3 处 gate）；无损版 standalone 正确性验证（128 token 字节级对比 nope/scale/rope 全一致）+ 端到端 `bench_serving` 856.64 tok/s；.8 容器 2026-08-25。

#### 为什么要这么改

原始路径（b300-glm52 的 `_write_mla_kv_buffer`）把「量化」和「写入」拆成两步：

```python
# memory_pool.py 原 dsa_kv_cache_store_fp8 分支
cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(cache_k_nope, cache_k_rope)  # ① 量化，产出中间 tensor
set_mla_kv_buffer_triton(dst_buffer, loc, cache_k_nope_fp8, cache_k_rope_fp8)                # ② 按 loc 写入 paged buffer
```

两步的代价是三类调度开销：**两次 kernel launch**（GPU 每次 launch 有固定开销）、**一次中间 tensor 的显存分配 + 读写往返**、**一次按 loc 的 scatter 拷贝**。decode 阶段每步 token 数很小（M ≈ 当前并发请求数），单次 kernel 的计算量本来就小，这三类开销反而占大头——所以省掉它们能带来可见收益。

现成的融合 kernel（`fused_qk_rope_reshape_and_cache`）之所以不能直接用，是因为它的量化逻辑和 DSA 不等价：它用**单个 `k_scale` 把整个 key（含 rope）统一量化成 fp8**，而 DSA 的 KV cache 要求 **k_nope 做 per-block 量化、k_rope 保留 bf16**。硬套上去，rope 位置编码被量化、精度受损——这正是有损版 803 tok/s 的代价。

所以最终选择自己写一个 DSA 专用的融合 triton kernel（`fused_dsa_quant_store`），把「per-block 量化 + bf16 rope 保留 + paged 直接写入」压缩进一个 kernel，且量化公式与 DSA 原 `quantize_k_cache_separate` 完全一致——既拿到融合收益（省 launch + 省中间 tensor + 省 scatter），又不损精度（856 tok/s）。

#### 涉及哪些基础知识

**1. MLA 的 K 拆成 nope + rope**

GLM-5.2 用 MLA（Multi-head Latent Attention），每个 token 的 K 由两部分拼成：**k_nope（512 维）**是内容向量、不带位置信息；**k_rope（64 维）**是位置向量。两者在 KV cache 里的存储需求不同（nope 可量化、rope 不能），这是「为什么要分开处理」的根因。

**2. RoPE 位置编码为什么不能量化**

RoPE（Rotary Position Embedding）通过旋转矩阵把「相对位置」编码进 k_rope 的 64 维。位置编码的微小误差会直接改变 token 之间的注意力关系，属于精度敏感部分，所以 DSA 对 rope 保留 bf16 原值、不做 fp8 量化。

**3. fp8 量化与 per-block scale**

fp8 e4m3fn 只有 4 位指数 + 3 位尾数，动态范围远小于 bf16。量化公式是 `y_q = clamp(y / scale, -448, 448)`，其中 `scale = max(|y|) / 448`（448 是 e4m3fn 的最大可表示值）。

- **per-tensor 量化**：整个 tensor 一个 scale，简单，但个别大值会把 scale 撑大、让多数小值的量化误差变大。
- **per-block 量化**：把 512 维按 128 一组切 4 块，每块独立算 scale（4 个 fp32 scale 共 16 字节），每块内数值范围相近，量化误差显著更小——这是 DSA 无损的关键。

**4. Paged KV cache 的写入方式**

KV cache 按固定大小 page 分配，`loc` 是每个 token 落到的 page 号。写入时不能简单 `buffer[loc] = tensor`，而要按 loc 把每个 token 的 656 字节 scatter 到对应 page——这正是 `set_mla_kv_buffer_triton` 那一步做的事，也是融合 kernel 里 `cache_loc = loc[token_id]` 之后按偏移写入的原因。

**5. 混合 dtype buffer layout**

DSA 每个 token 的 KV 存储是 656 字节，三种 dtype 挤在同一个 buffer 里：

```
[ k_nope 的 fp8（512 B）| 4 个 scale 的 fp32（16 B）| k_rope 的 bf16（128 B）]
```

一个 tensor 只能有一种 dtype，解法是**对同一块内存开三个不同 dtype 的 view**，各自指向不同字节偏移：fp8 view 写 nope（offset 0）、fp32 view 写 scale（offset 512）、bf16 view 写 rope（offset 528）。三个 view 共享底层内存，写入即落到正确位置。

**6. Triton kernel 的 2D grid 设计**

kernel 用 2D grid：`(num_tokens, 5)`——第 0 维是 token，第 1 维是 5 个 block（4 个 nope 块 + 1 个 rope 块）。`program_id(0)` 取 token、`program_id(1)` 分派到「量化哪个 nope 块」还是「拷 rope」。`GROUP_SIZE`、`DIM_NOPE` 等用 `tl.constexpr` 编译期常量，让 triton 按固定形状编译出最优指令；rope 块用 `mask = offs < DIM_ROPE` 处理 64 维对齐到 128 的越界。

**7. 算子融合到底省什么**

融合省的不是计算量，而是调度开销：

| 开销 | 原始两步 | 融合后 |
|---|---|---|
| kernel launch | 2 次（量化 + 写入） | 1 次 |
| 中间 tensor | `quantize_k_cache_separate` 的返回值 | 无（直接写进 KV buffer） |
| scatter 拷贝 | 量化结果 → 按 loc 拷贝进 buffer | 无（量化时直接写目标位置） |

这也是为什么融合收益出现在 decode（小 batch、launch 开销占比高），而不是 prefill（大 batch、计算密集、launch 开销被摊薄）。

四个融合点里，三个被硬性条件挡住（闭源 kernel / 不是融合对象），但 QK RoPE + KV cache write 通过写一个 DSA 专用的融合 triton kernel 突破了——无损实现 per-block fp8 量化 + bf16 rope 保留 + 混合 layout paged 写入，端到端从 793 提升到 **856 tok/s（+8%）**，总收益从 baseline 的 **+63%**。

完整性能链条：baseline 524 → allreduce fusion + flashinfer_trtllm 793（+51%）→ 新 kernel 无损 QK RoPE 融合 **856（+63%）**。

要继续提升，方向不再是"算子融合"：
1. 减少 embedding allreduce（27.8%，`--enable-attn-tp-input-scattered`，需实测 MLA/DSA 兼容性）
2. 换更大 batch 摊薄通信开销
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
