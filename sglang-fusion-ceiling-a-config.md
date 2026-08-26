# sglang 算子融合：从天花板到突破，写新 kernel 无损实现

当前使用的启动命令（GLM-5.2 coding-venti FP8，B200 TP8）：

```bash
python3 -m sglang.launch_server \
  --model-path /data0/models/glm52-coding-venti-fp8 \
  --served-model-name glm52-coding-venti-fp8 \
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

这套配置实测 789 tok/s（比融合全关的 baseline 524 tok/s 快 +50%）。这篇用 profile 数据 + 代码排除法回答：沿这套配置继续做算子融合还能不能提升，以及那些"看起来能融合"的点为什么实际做不了。

## TL;DR

这套配置实测 789 tok/s（比融合全关的 baseline 524 tok/s 快 +50%）。profile 排除法发现 decode 66 个 kernel 里只剩 4 个"可融合且未生效"的点，其中 QK RoPE + KV cache write（4.6%）被"三重不兼容"挡住。但写一个 DSA 专用融合 triton kernel 后无损突破，端到端 **796 tok/s（+52%）**。

完整性能链条：

| 配置 | 输出吞吐 (tok/s) | TPOT (ms) | vs baseline |
|---|---:|---:|---:|
| baseline（融合全关） | 524.34 | 26.77 | — |
| + allreduce fusion + flashinfer_trtllm | 789.20 | 17.04 | +50% |
| **+ QK RoPE 无损融合（新 kernel）** | **796.04** | **16.96** | **+52%** |

> 📐 **数据来源**：`bench_serving`（8192 input + 1024 output, 64 prompts, concurrency 16），.13 容器，fusionfix 镜像，2026-08-26。

## 排除法：66 个 kernel 排到只剩一个能做

decode 共 66 个 kernel，占比 ≥1% 的 23 个。按「为什么不能融合」归成 4 类，只剩 QK RoPE + KV cache write 一个能做（下一章）。

### 怎么 profile 出这张 kernel 表（可复现）

「66 个 kernel」来自 torch profiler 采样 + 一段内联统计。完整流程四步：

**① 触发采样**（sglang serve 的 torch profiler 端点，采 5 步、按 stage 分）：

```bash
curl -s -X POST http://127.0.0.1:10100/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir":"/tmp/glm_prof_fusion","num_steps":5,"activities":["GPU","CPU"],"profile_by_stage":true,"with_stack":true,"record_shapes":false,"profile_id":"glm-fusion","profile_prefix":"glm"}'
```

`10100` 就是启动命令里的 `--port 10100`（serving 主端口，`/generate`、`/metrics` 也挂在同一端口上）；`/start_profile` 是 sglang 内置的 torch profiler 采样开关——收到请求后 scheduler 进程直接开启 `torch.profiler`，serve 跑着就能采 kernel 级 trace，不用改代码、不用插桩。

请求体字段（`ProfileReq`）：

| 字段 | 作用 |
|---|---|
| `output_dir` | trace 落盘目录（默认 `/tmp`，可用环境变量 `SGLANG_TORCH_PROFILER_DIR` 覆盖） |
| `num_steps` | 采几步后自动停止，不用手动调 `/stop_profile` |
| `activities` | 采集的 kernel/op 事件（`["GPU","CPU"]`） |
| `profile_by_stage` | prefill / decode 分开落盘 |
| `with_stack` | 记录 python 调用栈 → 用于定位每个 kernel 的 python location |
| `profile_id` / `profile_prefix` | 拼进 trace 文件名 |

每个 TP rank 各产出一份 trace，文件名格式 `{profile_prefix}-{profile_id}-TP-{rank}-{STAGE}.trace.json.gz`——所以拉回 TP-0 的 decode 就是 `glm-glm-fusion-TP-0-DECODE.trace.json.gz`。

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

### 不能融合的 4 类原因

**① 已是融合 kernel / 已是最优实现**（无需再动）

| kernel | 占比 | 说明 |
|---|---|---|
| `oneshotAllreduceFusionKernel` / `rmsNormLamport` / `twoshotAllreduceKernel` | 3.6% / 1.9% / 1.7% | allreduce fusion 已开，融合后的产物 |
| `fused_a_gemm`（2 变体） | 3.1% | MLA A 投影，名字带 fused |
| `RopeQuantizeKernel` | 1.0% | RoPE + quant 融合 kernel |
| `fused_*_indexer_*`（3 个） | 0.6% | DSA indexer 融合 |
| `fmhaSm100fKernel`（2 变体） | 3.2% | flashinfer fused MLA attention，已是最优 |
| `rmsnormRMSNormKernel` | 1.8% | 单独 RMSNorm，已是最优 |

**② 不是融合对象**（没有可融合的相邻操作）

融合是"把多个小 kernel 合并成一个"，省的是 launch + 访存往返。两类没有可融合对象：

GEMM——计算密集大 kernel，无相邻小 kernel 可合并：

| kernel | 占比 | 说明 |
|---|---|---|
| `deep_gemm`（dense 层 fp8） | 11.7% | 计算密集 |
| `bmm_E4m3`（MoE 专家 GEMM，3 变体） | 16.9% | flashinfer trtllm MoE GEMM，已是单一大 kernel |
| `nvjet`（lm_head GEMM，3 变体） | 3.1% | lm_head bf16 GEMM |
| `cublasLt::splitKreduce` | 0.8% | GEMM 内部 splitK 归约 |
| `deep_gemm::sm100_paged_mqa_logits` | 0.5% | DSA paged MQA logits |

**embedding allreduce**（27.8%，decode 头号瓶颈）：

这是 decode 头号瓶颈。通过 trace 时序分析定位，它的前驱 kernel 是 `_vocab_parallel_embedding_kernel`：

| 次数 | dur | 前驱 kernel |
|---|---|---|
| 第1次 | 26.89ms（warmup 异常） | `_vocab_parallel_embedding_kernel` |
| 第3次 | 1.73ms | `_vocab_parallel_embedding_kernel` |
| 第5次 | 2.10ms | `_vocab_parallel_embedding_kernel` |

**前置知识：TP 下 vocab embedding 怎么拆、为什么要 allreduce**

TP8 下，`VocabParallelEmbedding` 把 154880 行的 embedding 权重按 **vocab 维度**切成 8 份，每个 rank 只存 19360 行（`python/sglang/srt/layers/vocab_parallel_embedding.py`）。forward 分两步：

```python
# VocabParallelEmbedding.forward
output_parallel = self._embed_local_shard(input_)                    # ① 每个 rank 只查自己分片内的 token
output_parallel = tensor_model_parallel_all_reduce(output_parallel)  # ② allreduce 汇总
```

① `_embed_local_shard`：对输入的 token id，**落在本 rank 分片内的**才查本地 embedding 表，**落在其他 rank 的**直接填 0。于是每个 rank 得到一个「partial」结果——只有自己负责的那几个 token 位置有值，其余全 0。

② allreduce（sum）：8 个 rank 的 partial 相加。因为每个 token 位置只在「拥有它的那个 rank」上非零，sum 后正好还原出完整的 embedding。

这就是 decode 头号瓶颈（27.8%）的来源：**每个 decode step 都要为最新生成的 1 个 token 走一遍「embedding 查表 + 8 卡 allreduce」**，而 decode 阶段 token 数极少（M ≈ 并发数），通信开销无法被计算摊薄。

**为什么不能融合**：allreduce fusion 融合的是「allreduce + residual_add + rmsnorm」三个连续操作，这个模式只出现在**每个 transformer 层的输出处**——attention/MoE 的输出是 TP partial，后面紧跟「加回残差 + 归一化」，所以能合成一个 kernel。而 embedding 的 allreduce 有两个致命不同：

1. **没有 residual 可加**：embedding 是模型的第一个输入，`residual = None`（GLM `forward` 里 `residual = None`），根本没有「上一层的残差」。
2. **不构成「allreduce → rmsnorm」的紧邻序列**：embedding 的 allreduce 在 `embed_tokens` 内部结束，而下一层的 `input_layernorm` 在 layer.0 内部才执行，两者隔着层边界，不是连续 kernel。

所以 embedding 的 allreduce 是「裸的 allreduce」，没有任何可以合并进去的操作——它是 TP vocab parallel embedding 的固有开销，不是融合的漏网之鱼。

> 📐 **数据来源**：TP-0 DECODE trace 的 kernel 时序分析，`all_reduce_kernel` 事件的前驱 kernel 均为 `_vocab_parallel_embedding_kernel`。


**③ flashinfer_trtllm 闭源挡住**（换回 triton 慢 24%）

这三个要融合，都得动 flashinfer_trtllm 的闭源预编译 kernel；而换回 `triton` 又慢 24%（603 vs 789），得不偿失。

trtllm MoE 内部子 kernel（本来就在闭源 kernel 里，无法单独动）：

| kernel | 占比 | 说明 |
|---|---|---|
| `moe::finalize::finalizeKernel` | 2.2% | trtllm MoE 内部，闭源预编译 |
| `moe::routing::routingIndicesDynBlockKernel` | 2.0% | trtllm MoE 内部 routing |
| `moe::activation::activationDeepSeekKernel` | 1.6% | trtllm MoE 内部 activation |

**shared-expert append**（3.2%）：

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


**FP8 quant + activation**（3.7%）：

`per_token_group_quant_8bit_v2_kernel`（3.7%）是 MoE 专家 GEMM 前的 FP8 per-token 量化。catalog 里 vLLM 有 `fuse_act_quant` pass（SiLU+mul + FP8 quant 融合），sglang 的 triton fused_moe 里有类似 epilogue，但 **flashinfer_trtllm backend 的 MoE kernel 是闭源预编译 kernel，不开放 epilogue 定制**——没法把量化融进去。

**为什么不能融合**：要开它只能把 `--moe-runner-backend` 换回 `triton`（triton 的 `fused_moe` 有融合 epilogue），但 triton 实测只有 603 tok/s（比当前 793 慢 24%），同样得不偿失。

> 📐 **数据来源**：profile 里 `per_token_group_quant` 的 python location（`fused_moe.py:232`，走 triton path 才有融合 epilogue）。
**④ 占比太小不值得**（单个 <2%，无明确融合对象）

| kernel | 占比 | 说明 |
|---|---|---|
| `vectorized_elementwise`（5 变体） | 5.8% | PyTorch 零碎 elementwise（compare/where/bitwise_and），单个 <2% |
| `set_mla_kv_buffer_kernel` | 0.5% | MLA KV cache 写入，太小 |
| `cunn_SoftMaxForward` | 0.2% | 已融在 fmha kernel 里 |
| `topk_small_batch_kernel` | 0.2% | 太小 |

> 📐 **数据来源**：TP-0 DECODE trace（`glm-glm-fusion-TP-0-DECODE.trace.json.gz`，.8 容器 2026-08-25，采样 5 步）。「66 个 kernel / 110.7ms / 占比≥1% 的 23 个」由上文的「④ 数 kernel」脚本统计得到；`analyze_llm_torch_profile.py`（`.claude/skills/llm-torch-profiler-analysis/scripts/`）输出的是聚合三表，不含 distinct kernel 计数。

66 个 kernel 按这 4 类排除完，只剩 QK RoPE + KV cache write（4.6%）一个能做。

## 能融合的：QK RoPE + KV cache write（工作重心）

先厘清「patch」指什么。本次实验**镜像统一**是 fusionfix（`v0.5.15.post1-cuda13-b200-fusionfix`，把 flashinfer 三件套对齐到 0.6.14，否则开融合会崩 tvm_ffi 报错）。变量只有两个——**启动命令**（开不开融合）和**代码改动**（挂不挂新 kernel）：

| 配置 | 启动命令（差异部分） | 代码改动 |
|---|---|---|
| 融合全关 | `--moe-runner-backend triton` + `--enforce-disable-flashinfer-allreduce-fusion` + `--disable-custom-all-reduce` | 无 |
| 开融合（无 patch） | `--moe-runner-backend flashinfer_trtllm` + `--flashinfer-allreduce-fusion-backend auto` | 无 |
| 开融合 + 无损 patch | 同上 | 挂载 `fused_dsa_quant_store.py`（新 kernel）+ `memory_pool.py`（接入改动） |

下面「patch 前」= 开融合但无 patch，「patch 后」= 开融合 + 无损 patch。

fuse 表说「已有 `attention/utils.py` 融合路径」，但那条路径（`fused_qk_rope_reshape_and_cache`）用单个 k_scale 把整个 key（含 rope）统一量化成 fp8，而 DSA 要求 nope 做 per-block 量化、rope 保留 bf16——直接用它会把 rope 位置编码量化掉、损精度。所以要自己写一个 DSA 专用的融合 kernel，量化逻辑和 DSA 原路径逐行等价。

### 背景知识：MLA 的 K 和 fp8 KV cache

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

### patch 前：两步（量化 + 写入）

开融合但没打 patch 时，`_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支把「量化」和「写入」拆成两步：

```python
# memory_pool.py 原 dsa_kv_cache_store_fp8 分支
cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(cache_k_nope, cache_k_rope)  # ① 量化，产出中间 tensor
set_mla_kv_buffer_triton(dst_buffer, loc, cache_k_nope_fp8, cache_k_rope_fp8)                # ② 按 loc 写入 paged buffer
```

两步的代价：两次 kernel launch + 一次中间 tensor 的显存分配与读写往返 + 一次按 loc 的 scatter 拷贝。

### patch 后：一步（新 kernel 融合）

写了一个 DSA 专用的融合 triton kernel（`fused_dsa_quant_store`），在一个 kernel 里同时完成：

1. 对 k_nope（512 维）按 128 group 做 per-block fp8 量化（每个 block 独立 scale）
2. 对 k_rope（64 维）保留 bf16 不量化
3. 三个 dtype 的数据（fp8 + fp32 scale + bf16）按混合 layout 直接写入 paged KV buffer

**混合 dtype 写入的解法**：传 3 个 dtype 不同的指针（fp8/fp32/bf16），各指向同一 paged buffer 的不同字节偏移——nope 指针写 fp8 量化值（offset 0-511）、scale 指针写 fp32（offset 512-527）、rope 指针写 bf16（offset 528-655），绕过"一个 tensor 只能一种 dtype"的限制。

替代原来的 `quantize_k_cache_separate()` + `set_mla_kv_buffer_triton()` 两步，消除中间 tensor 分配 + 额外 kernel launch。

**正确性验证**：standalone test 对比融合 kernel 和非融合路径的输出，128 token、字节级对比 nope（fp8）/ scale（fp32）/ rope（bf16）三部分全部一致——**完全无损**。

**端到端数据**：

| 指标 | patch 前（无 patch） | patch 后（无损 patch） | 差异 |
|---|---:|---:|---:|
| 输出吞吐 (tok/s) | 789.20 | **796.04** | **+0.9%** |
| TPOT 均值 (ms) | 17.04 | 16.96 | -0.5% |
| TTFT 均值 (ms) | 3317.22 | 3220.75 | -2.9% |

patch 后省掉了两步里的「中间 tensor 分配 + 一次 launch + scatter 拷贝」——这是速度增益的来源（增益很小，只有 +0.9%，在 benchmark 噪声范围内）。

> 📐 **数据来源**：standalone test（`fused_dsa_quant_store.py` test_correctness，128 token，字节级对比）；端到端 `bench_serving`（8192 input + 1024 output, 64 prompts, concurrency 16），.13 容器，fusionfix 镜像，2026-08-26。新 kernel 代码 `python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py`，接入点 `python/sglang/srt/mem_cache/memory_pool.py:3951`（`_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支）。

## 总结

| 融合点 | 占比 | 状态 | 原因 |
|---|---|---|---|
| embedding allreduce | 27.8% | ❌ 不能 | 不是融合对象（无 layernorm 吸收） |
| shared-expert append | 3.2% | ❌ 不能 | flashinfer_trtllm 闭源 kernel 没实现 |
| **QK RoPE + KV cache write** | **4.6%** | **✅ 已实现** | **写新 triton kernel 无损实现（字节级一致）** |
| FP8 quant + activation | 3.7% | ❌ 不能 | trtllm 闭源不开放 epilogue |

### QK RoPE 融合怎么实现的

现成的融合 kernel（`fused_qk_rope_reshape_and_cache`）用单个 `k_scale` 统一量化整个 key（含 rope），会把 rope 位置编码量化掉、损精度。所以自己写了一个 DSA 专用的融合 kernel，量化逻辑和 DSA 原路径逐行等价（无损）。

#### 无损版本（写新 triton kernel）

写了一个 DSA 专用的融合 triton kernel（`fused_dsa_quant_store`），在一个 kernel 里同时完成三件事：

1. **k_nope（512 维）per-block fp8 量化**：按 128 维一组切 4 块，每块独立算 scale（`y_s = max(abs(y)) / FP8_MAX`），量化成 fp8——和 DSA 原来的 `quantize_k_cache_separate` 逻辑完全一致，保证无损。
2. **k_rope（64 维）保留 bf16**：直接拷贝，不量化——位置编码精度敏感，DSA 原来也是这么做的。
3. **混合 dtype 写入 paged KV buffer**：DSA 的 KV buffer 是混合 layout（`[fp8(512) | fp32 scale(16) | bf16(128)]` 拼在一个 buffer 里），解法是传 3 个 dtype 不同的指针（fp8 / fp32 / bf16），各指向同一 paged buffer 的不同字节偏移，绕过"一个 tensor 只能一种 dtype"的限制。

替代原来的 `quantize_k_cache_separate()` + `set_mla_kv_buffer_triton()` 两步，消除中间 tensor 分配 + 额外 kernel launch。

#### 无损的证明（理论 + 实验）

「无损」不是口头承诺，可以从理论和实验两方面证死。

**理论：量化公式与 DSA 原 kernel 逐行等价**

无损版 kernel（`_fused_dsa_quant_store_kernel`）的量化逻辑，和 DSA 原本的 `_quantize_k_cache_fast_kernel`（`quantize_k_cache_separate` 的底层 kernel，位于 `python/sglang/kernels/ops/attention/dsa/quant_k_cache.py`）**逐行等价**。

DSA 原 kernel 的量化核心：

```python
y_s = tl.max(tl.abs(y)) / FP8_MAX              # scale = max(|y|) / 448
y_s_inv = 1.0 / y_s
y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(fp8)   # 量化
tl.store(dst_q_ptr, y_q, mask=mask)            # 存 fp8 量化值
tl.store(dst_s_ptr, y_s)                        # 存 fp32 scale
# rope 部分：bf16 原样拷贝，不量化
data = tl.load(src_ptr, mask=mask)
tl.store(dst_ptr, data, mask=mask)
```

无损版 kernel 的量化核心：

```python
y_s = tl.max(tl.abs(y)) / FP8_MAX
y_s_inv = 1.0 / y_s
y_q = tl.clamp(y * y_s_inv, -FP8_MAX, FP8_MAX).to(fp8)
tl.store(dst_q, y_q)
tl.store(dst_s, y_s)
# rope 部分：bf16 原样拷贝
data = tl.load(..., mask=mask, other=0.0)
tl.store(dst, data, mask=mask)
```

逐项对照：

| 环节 | DSA 原 kernel | 无损版 kernel | 等价性 |
|---|---|---|---|
| scale 公式 | `y_s = max(abs(y)) / FP8_MAX` | 同 | ✅ 完全一致 |
| 量化公式 | `clamp(y/y_s, FP8_MIN, FP8_MAX)` | `clamp(y/y_s, -FP8_MAX, FP8_MAX)` | ✅ fp8 e4m3fn 对称，`FP8_MIN = -FP8_MAX = -448` |
| scale 存储 | 每 128 维一块，fp32 | 同 | ✅ |
| rope 处理 | bf16 原样拷贝，不量化 | 同 | ✅ |
| block 划分 | 512/128 = 4 个 nope 块 + 1 个 rope 块 | 同（`num_blocks_per_token = 5`） | ✅ |
| 字节布局 | `[fp8(512) \| fp32(16) \| bf16(128)]` | 同（offset 0 / 512 / 528） | ✅ |

唯一区别是**写入目标**：原路径先量化进中间 tensor，再由 `set_mla_kv_buffer_triton` 按 `loc` scatter 进 paged buffer；无损版在量化时直接把三个 dtype 写到 paged buffer 的对应偏移。**写进去的 656 字节完全相同**。

为什么「字节相同 = 无损」：KV cache 里存的就是这 656 字节，下游 attention（flashinfer/trtllm）读的也是这 656 字节。字节一致 ⇒ 反量化出来的 K 完全一致 ⇒ decode 逐 token 输出相同。

**实验：128 token 字节级对比**

standalone 测试（`fused_dsa_quant_store.py` 的 `test_correctness`，随分支提交）：随机 128 个 token 的 k_nope/k_rope + 随机 loc，非融合路径（`quantize_k_cache` + `kv_buffer[loc] = ...`）和融合路径（`fused_dsa_quant_store`）各自写 paged buffer，逐字节对比三个区域：

| 区域 | 字节范围 | 结果 |
|---|---|---|
| nope fp8 量化值 | `[0, 512)` | ✅ 完全一致 |
| scale fp32 | `[512, 528)` | ✅ 完全一致 |
| rope bf16 | `[528, 656)` | ✅ 完全一致 |

128 token × 656 字节 = 83968 字节，逐字节全等。

注：ground truth 用的 `quantize_k_cache`（concat 路径）和 b300-glm52 实际用的 `quantize_k_cache_separate` 共用同一个 `_quantize_k_cache_fast_kernel`，官方 `quant_k_cache.py` 的 `__main__` 已验证两者字节级一致，所以用 `quantize_k_cache` 做 ground truth 等价。

**实验代码**（`fused_dsa_quant_store.py`，clone 分支后在带 GPU 的容器里 `python fused_dsa_quant_store.py` 即可复现）：

```python
def test_correctness():
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
    torch.manual_seed(42)
    num_tokens, size = 128, 256
    k_nope = torch.randn(num_tokens, DIM_NOPE, dtype=torch.bfloat16, device="cuda")
    k_rope = torch.randn(num_tokens, DIM_ROPE, dtype=torch.bfloat16, device="cuda")
    loc = torch.randint(0, size, (num_tokens,), dtype=torch.int32, device="cuda")

    # 非融合 ground truth：concat 量化 + index 写入
    kv_buffer_ref = torch.zeros(size, 1, TOTAL_BYTES, dtype=torch.float8_e4m3fn, device="cuda")
    cache_k = torch.cat([k_nope.unsqueeze(1), k_rope.unsqueeze(1)], dim=-1).unsqueeze(1)
    cache_k_quant = quantize_k_cache(cache_k).squeeze(1).squeeze(1)
    kv_buffer_ref[loc] = cache_k_quant.unsqueeze(1)

    # 融合路径：一个 kernel 直接写 paged buffer
    kv_buffer_fused = torch.zeros(size, 1, TOTAL_BYTES, dtype=torch.float8_e4m3fn, device="cuda")
    fused_dsa_quant_store(k_nope, k_rope, kv_buffer_fused, loc)

    # 字节级对比三个区域
    ref_u8 = kv_buffer_ref.view(torch.uint8)
    fused_u8 = kv_buffer_fused.view(torch.uint8)
    ref_tokens, fused_tokens = ref_u8[loc], fused_u8[loc]
    nope_match = torch.equal(ref_tokens[:, :, :DIM_NOPE], fused_tokens[:, :, :DIM_NOPE])
    scale_match = torch.equal(ref_tokens[:, :, DIM_NOPE:DIM_NOPE+SCALE_BYTES],
                               fused_tokens[:, :, DIM_NOPE:DIM_NOPE+SCALE_BYTES])
    rope_match = torch.equal(ref_tokens[:, :, DIM_NOPE+SCALE_BYTES:],
                              fused_tokens[:, :, DIM_NOPE+SCALE_BYTES:])
    print(f"nope fp8 量化值:  {'✅ 一致' if nope_match else '❌ 不一致'}")
    print(f"scale (fp32):    {'✅ 一致' if scale_match else '❌ 不一致'}")
    print(f"rope (bf16):     {'✅ 一致' if rope_match else '❌ 不一致'}")
    assert nope_match and scale_match and rope_match, "无损验证失败"
```

> 📐 **数据来源**：`test_correctness` 在带 GPU 的容器（B200，fusionfix 镜像）内运行，`torch.manual_seed(42)`，输出三行 `✅ 一致`。完整代码（含失败时 diff 打印的 `else` 分支）随分支 `pex/qk-rope-fusion-lossless` 的 `fused_dsa_quant_store.py` 提交，末尾 `if __name__ == "__main__": test_correctness()`。

#### 代码位置

| 版本 | 分支 | 文件 | 说明 |
|---|---|---|---|
| 无损版 | [`pex/qk-rope-fusion-lossless`](https://github.com/MindLab-Research/sglang/tree/pex/qk-rope-fusion-lossless) | `python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py`（新 kernel）<br>`python/sglang/srt/mem_cache/memory_pool.py:3951`（接入点） | 新写 triton kernel + 替换 `_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支 |

> 📐 **数据来源**：无损版 standalone 正确性验证（128 token 字节级对比 nope/scale/rope 全一致）+ 端到端 `bench_serving` 796.04 tok/s；.13 容器，fusionfix 镜像，2026-08-26。

#### 为什么要这么改

原始路径（b300-glm52 的 `_write_mla_kv_buffer`）把「量化」和「写入」拆成两步：

```python
# memory_pool.py 原 dsa_kv_cache_store_fp8 分支
cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(cache_k_nope, cache_k_rope)  # ① 量化，产出中间 tensor
set_mla_kv_buffer_triton(dst_buffer, loc, cache_k_nope_fp8, cache_k_rope_fp8)                # ② 按 loc 写入 paged buffer
```

两步的代价是三类调度开销：**两次 kernel launch**（GPU 每次 launch 有固定开销）、**一次中间 tensor 的显存分配 + 读写往返**、**一次按 loc 的 scatter 拷贝**。decode 阶段每步 token 数很小（M ≈ 当前并发请求数），单次 kernel 的计算量本来就小，这三类开销反而占大头——所以省掉它们能带来可见收益。

现成的融合 kernel（`fused_qk_rope_reshape_and_cache`）之所以不能直接用，是因为它的量化逻辑和 DSA 不等价：它用**单个 `k_scale` 把整个 key（含 rope）统一量化成 fp8**，而 DSA 的 KV cache 要求 **k_nope 做 per-block 量化、k_rope 保留 bf16**。硬套上去，rope 位置编码被量化、精度受损——这正是要自己写 kernel 的原因。

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

排除剩下的 4 个融合点里，3 个被硬性条件挡住（闭源 kernel / 不是融合对象），唯一能做的是 QK RoPE + KV cache write——写一个 DSA 专用的融合 triton kernel 无损实现，端到端从 789 提升到 **796 tok/s（+0.9%）**，总收益从 baseline 的 **+52%**。

完整性能链条：baseline 524 → allreduce fusion + flashinfer_trtllm 789（+50%）→ 新 kernel 无损 QK RoPE 融合 **796（+52%）**。

无损版的速度增益很小（+0.9%，在 benchmark 噪声范围内），它的核心价值是**字节级无损**——把「量化 + 写入」两步合成一步、且保证和开融合无 patch 的输出完全一致（见「无损的证明」），而不是「再快一档」。

要继续提升，方向不再是"算子融合"：
1. 减少 embedding allreduce（27.8%，`--enable-attn-tp-input-scattered`，需实测 MLA/DSA 兼容性）
2. 换更大 batch 摊薄通信开销
3. 等 flashinfer_trtllm 后续版本开放 shared-expert fusion / epilogue 定制

## 完整环境

| 组件 | 版本 |
|---|---|
| 硬件 | 8× NVIDIA B200（179G/卡），.13 |
| 镜像 | `b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-fusionfix` |
| sglang | `sgl-project/sglang` release/v0.5.15，HEAD `0b3bb0c` |
| torch | 2.11.0+cu130 |
| flashinfer | python 0.6.14 / cubin 0.6.14 / jit-cache 0.6.14+cu130（已对齐） |
| 模型 | `/data0/models/glm52-coding-venti-fp8`（E=257 MoE，topk=8，hidden=6144，fp8_w8a8，MLA + DSA） |
| serve 关键参数 | `--moe-runner-backend flashinfer_trtllm --flashinfer-allreduce-fusion-backend auto --kv-cache-dtype fp8_e4m3 --tp 8` |
| benchmark | `bench_serving --random-input-len 8192 --random-output-len 1024 --num-prompts 64 --max-concurrency 16`，796 tok/s |

> 📐 **数据来源**：本报告所有代码引用均来自 `sgl-project/sglang` release/v0.5.15（HEAD `0b3bb0c`），profile 数据来自 .13 实测（2026-08-26）。
