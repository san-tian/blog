# sglang MoE 内核调优报告（GLM-5.2 step111 FP8 / B200）

> 日期：2026-08-24
> 机器：两台 B200 机器（8×179G/卡），下称 .13 / .8
> 镜像：`<ACR>/mindverse/sglang:v0.5.15.post1-cuda13-b200`
> 模型：`/data0/models/glm52-step111-fp8`（E=257 expert，topk=8，hidden=6144，fp8_w8a8，block[128,128]）

---

## 0. 结论速览（TL;DR）

> **调优对象明确一下**：这次调的是 `fused_moe_triton` —— **MoE 专家层（gate/up/down 投影）的 triton kernel，仅此一个**。dense 层（DeepGEMM）是“编译”不是“tune”；注意力/FlashInfer 是预编译库、没有 config 可调。所以下文所有“调优”都特指这个 MoE triton kernel 的 tile 参数搜索。

| 调优项 | 状态 | 收益 |
|---|---|---|
| ① MoE up 投影 config（full 18 batch）| ✅ 完成 | **13/14 个 batch 与 fallback 不同**；bs=16 up 投影 **~4-8% 快** |
| ② DeepGEMM 预编译 | ✅ 完成 | 冷机器首 serve 省 10-20min JIT |
| ③ MoE down 投影 TMA | ✅ 量化 | **down 投影开 TMA 快 ~8-16%**（bs=16）|

**核心发现（含端到端实测修正）**：`Config file not found` 回退到的是**旧版 triton 3.5.1 的 tuned config**。kernel 级 micro-benchmark 显示重新 tune 的 config 快 3-8%、down 开 TMA 快 8-16%（MoE kernel 合计 ~12%）；**但端到端实测（tokens/sec）证明这 12% 不成立**——tuned config 端到端吞吐 0~-10%（略慢或无差别）。**最终结论：保留官方 3.5.1 fallback，不要用重新 tune 的 config。**（详见 §7）

---

## 1. 为什么需要 tune（背景）

sglang 的 `fused_moe_triton` 内核，对不同 shape（`E`=expert 数、`N`=中间维、`dtype`、`block_shape`）各有一份 config 文件，内容是按 batch size（`M`）划分的 tiling 参数：

```json
{
    "16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
}
```

- 这些 config 放在镜像内：`/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/`
- 若某个 shape 缺 config，启动日志会打 `Config file not found ... Fallback to triton 3.5.1 ... Performance might be sub-optimal`。
- "补充 tune" = 跑官方 benchmark 脚本，把缺的 shape 的最优 config 找出来写进 config 目录。

---

## 1b. 调优里的「batch」和「调什么」

### 「batch」（config 里的 M）指什么

不是「并发请求数」，而是**当前一次 MoE 前向里处理的 token 数**：

```
[num_tokens, hidden] × [hidden, intermediate] → [num_tokens, intermediate]
        ↑ M（token 维）
```

- **decode**：每个正在生成的请求每步吐 1 个 token → M ≈ 当前并发请求数
- **prefill**：整段 prompt 一起算 → M ≈ chunked prefill 大小

`--max-running-requests` / `--cuda-graph-max-bs-decode` 只是「M 允许到多大」的上限；config 里的 M 才是 kernel 实际看到的 token 数，两者相关但不是一回事。

### tune 调的是什么：6 个 tiling 参数

| 参数 | 控制什么 |
|---|---|
| `BLOCK_SIZE_M` | 一个 thread-block 处理多少 token（M 维块大小）|
| `BLOCK_SIZE_N` | 输出特征维（intermediate/hidden）块大小 |
| `BLOCK_SIZE_K` | 输入特征维（K）块大小 |
| `GROUP_SIZE_M` | 多少个 M 块共用一份权重 tile（L2 缓存复用）|
| `num_warps` | 每 block 用多少 warp（并行度）|
| `num_stages` | 软件流水线深度（预取下一块、掩盖访存延迟）|

triton 是 JIT 的：**不同 tile 参数 → 编译出不同 CUDA kernel → 不同共享内存/寄存器压力/访存模式 → 不同性能**。tune 就是把 1280 种组合都编译+跑一遍，选每个 M 下最快的那组。

**为什么 M 不同最优参数就不同**：M 小（token 少）块切小（M=16）避免 padding 浪费；M 大块切大（M=64）更好合并访存；N/K 切法、流水线深度都要跟着 M 在「共享内存够不够」和「并行够不够」之间重新平衡。

---

## 2. 调优项 ①：MoE triton up 投影 config

### 2.1 定位缺失的 shape（从 serving 日志）

```bash
docker logs sglang-step111 2>&1 \
  | grep -oE "triton_3_6_0/E=[0-9]+,N=[0-9]+,device_name=NVIDIA_B200[^ ]*\.json" \
  | sort -u
```

结果（去重后）：
```
triton_3_6_0/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json
triton_3_6_0/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
```

即缺两个：**up 投影**（`.json`）和 **down 投影**（`_down.json`）。

### 2.2 调优命令（tuning_fused_moe_triton.py）

工具在镜像内 `/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/`，需要 `ray`（镜像没装，先 `pip install ray`）。

```bash
# 起一个调优容器（模型挂载 + 输出落盘）
docker run -d --name moe-tune --gpus all --shm-size=8g \
  -v /data0/models:/data0/models \
  -v /root/tune_out:/out \
  --entrypoint bash \
  <ACR>/mindverse/sglang:v0.5.15.post1-cuda13-b200 \
  -c "pip install ray >/dev/null 2>&1 && cd /sgl-workspace/sglang && \
      python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
        --model /data0/models/glm52-step111-fp8 --tp-size 8 --dtype fp8_w8a8 --tune \
        > /out/full_tune.log 2>&1; \
      cp /sgl-workspace/sglang/E=257*.json /out/"
```

关键参数：
- `--tp-size 8`：与 serving 的 TP 一致（决定 shard_intermediate_size）。
- `--dtype fp8_w8a8`：与 serving 的 fp8 量化一致。
- `--batch-size 16`（可选）：只 tune 单个 batch size；不加则用默认清单 `[1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096]`（18 个，约 18h）。

### 2.3 ⚠️ 重要：config 写到哪了

`save_configs()` 用**相对路径** `open(filename, "w")`，写到**调优进程的 cwd**（`/sgl-workspace/sglang/`），**不是** config 目录。所以要手动把生成的文件挪进 `configs/triton_3_6_0/`（见上面的 `cp ... /out/` 落盘，然后挂载回 serving）。

### 2.4 bs=16 的实测结果（before/after，修正版）

⚠️ 之前用主脚本 benchmark 测出的"0 收益"是**误测**——主脚本 `get_moe_configs` 对缺失 shape 回退到的是**工具内置默认值**（N128/stages4），不是 serving 实际用的 3.5.1 fallback（N64/stages3）。用 sep 脚本 `--configs` 直接对比两个 config（单位 us）：

| 组件 | 3.5.1 fallback (N64/s3) | 3.6.0 tuned (N128/s4) | 提升 |
|---|---|---|---|
| up t0 非TMA | 71.63 | 68.78 | -4.0% |
| up t0 TMA | 72.06 | 65.98 | -7.9% |
| down t1 非TMA | 50.89 | 49.51 | -2.7% |
| down t1 TMA | 46.72 | **41.70** | **-18.1%** |
| **合计 t0+t1** | 122.52 | 107.68（TMA） | **-12.1%** |

### 2.5 全 batch 调优结果（✅ 完成）

full tune（18 batch）跑完，生成 `/root/tune_out/E=257,N=256,...json`（18 个 batch 全量）。**与 3.5.1 fallback 相比，13/14 个 batch 的 config 不同**（仅 bs=24 相同），主要差异：bs≤16 时 N64→N128、bs≥512 时 M16/M32→M64 等。

---

## 3. 调优项 ②：DeepGEMM 预编译

### 3.1 背景

DeepGEMM 是 dense 层 fp8 GEMM 的后端（B200 上 `fp8_gemm_runner_backend=auto` → deep_gemm）。首次 serve 时会对每个 GEMM shape 做 JIT 编译（10-20 分钟）。预编译 = 提前把 kernel 编译进缓存，让 serve 更快启动。

启动日志里的 shape 线索：
```
Try DeepGEMM JIT Compiling for <GEMM_NT_F8F8BF16> N=2048, K=2048 ...
Try DeepGEMM JIT Compiling for <GEMM_NT_F8F8BF16> N=3072, K=6144 ...
...（共 6 个 shape）
```

### 3.2 命令

```bash
docker run --rm --gpus all --ipc=host \
  -v /data0/models:/data0/models \
  -v /root/.cache/deep_gemm:/root/.cache/deep_gemm \
  --entrypoint python3 \
  <ACR>/mindverse/sglang:v0.5.15.post1-cuda13-b200 \
  -m sglang.compile_deep_gemm \
    --model-path /data0/models/glm52-step111-fp8 --tp-size 8 \
    --enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce
```

### 3.3 踩坑（四次才跑通）

| 报错 | 根因 | 修法 |
|---|---|---|
| `NCCL error: unhandled system error` | TP8 初始化 NCCL 需要 IPC | `--ipc=host` |
| `TypeError: trtllm_mnnvl_allreduce_fusion Expected 16 but got 21` | flashinfer allreduce tvm_ffi 版本不匹配 | `--enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce` |
| `TypeError: trtllm_fp8_block_scale_moe Expected 29 but got 33` | flashinfer TRTLLM MoE tvm_ffi 版本不匹配 | `--moe-runner-backend triton`（与 serving 一致）|

> 注意 flag 名：是 `--model-path` + `--tp-size`（不是日志示例里的 `--model`/`--tp`）。

### 3.4 收益（✅ 完成）

第 4 次（带齐 4 个 flag）成功：

```
The server is fired up and ready to roll!
DeepGEMM Kernels compilation finished successfully.
```

耗时约 **4.3 分钟**（15:25:23 → 15:29:38）。注：`.8` 的 `/root/.cache/deep_gemm` 此前已由官方 FP8 prefill 预热过，本次是复验+补齐；**真正的收益场景是“冷机器首次 serve”**——预编译后首次启动不再现场 JIT（省 10-20 分钟）。

---

## 4. 调优项 ③：MoE down 投影 config（TMA）

### 4.1 背景

serving 日志里 down 投影是「**reusing tuned up without TMA**」——即 down 投影复用 up 的 config 但**没用 TMA**（Tensor Memory Accelerator，Blackwell 的异步访存加速）。独立 tune down 投影可能开 TMA 拿到个位数 % 收益。

### 4.2 命令（tuning_fused_moe_triton_sep.py）

需要先准备 topk_ids（随机路由即可，benchmark GEMM 时具体路由不影响耗时）：

```bash
# 生成 100 个随机 topk_ids（shape=(M,topk)，M=batch size，topk=9）
cat > /root/gen_topk_ids.py <<'EOF'
import torch, os
os.makedirs("/root/topk_ids", exist_ok=True)
num_layers, dense_layers = 61, 3   # sep 脚本硬编码
moe_layers = num_layers - dense_layers
M, topk, E = int(os.getenv("M","16")), int(os.getenv("TOPK","9")), 257
for i in range(100):
    layer = i % moe_layers + dense_layers
    idx = i // moe_layers
    torch.save(torch.randint(0, E, (M, topk), dtype=torch.int32),
               f"/root/topk_ids/topk_ids_layer{layer}_idx{idx}.pt")
EOF
docker run --rm -v /root/topk_ids:/root/topk_ids -v /root/gen_topk_ids.py:/gen_topk_ids.py \
  --entrypoint python3 <IMG> /gen_topk_ids.py

# 跑 sep tune（down 投影独立，含 TMA）
docker run -d --name moe-sep-tune --gpus all --shm-size=8g \
  -v /data0/models:/data0/models -v /root/topk_ids:/root/topk_ids -v /root/tune_out:/out \
  --entrypoint bash <IMG> -c "pip install ray && cd /sgl-workspace/sglang && \
    python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
      --model /data0/models/glm52-step111-fp8 --tp-size 8 --dtype fp8_w8a8 \
      --batch-size 16 --topk-ids-dir /root/topk_ids --tune > /out/sep_tune.log 2>&1"
```

### 4.3 收益（✅ 已量化）

用 `--configs` 模式直接对比 up/down 的 TMA 与非 TMA（bs=16）：

| 组件 | 非 TMA | TMA | TMA 收益 |
|---|---|---|---|
| down 投影 t1（3.6.0 tuned）| 49.51 us | 41.70 us | **-15.8%** |
| up 投影 t0（3.6.0 tuned）| 68.78 us | 65.98 us | -4.1% |

**down 投影开 TMA 收益最大（~16%）**，正好对应日志里"reusing up without TMA"的次优点。注：sep 脚本的 `--tune` 模式不落盘（代码里无 `save_configs`），要用 `--configs` 手动对比，或自己把 `_down` config 写进 `configs/triton_3_6_0/`。

---

## 5. 复现清单（从零到跑完）

```bash
# 0) 前置：确认镜像 + 模型
docker images | grep sglang
ls /data0/models/glm52-step111-fp8/config.json

# 1) 找缺失的 config shape
docker logs <serving容器> 2>&1 | grep -oE "triton_3_6_0/E=[0-9]+,N=[0-9]+[^ ]*\.json" | sort -u

# 2) tune up 投影（单 batch，快速验证）
docker run -d --name moe-tune --gpus all --shm-size=8g -v /data0/models:/data0/models \
  --entrypoint bash <IMG> -c "pip install ray && cd /sgl-workspace/sglang && \
  python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model /data0/models/glm52-step111-fp8 --tp-size 8 --dtype fp8_w8a8 --batch-size 16 --tune"

# 3) DeepGEMM 预编译
docker run --rm --gpus all --ipc=host -v /data0/models:/data0/models \
  -v /root/.cache/deep_gemm:/root/.cache/deep_gemm --entrypoint python3 <IMG> \
  -m sglang.compile_deep_gemm --model-path /data0/models/glm52-step111-fp8 --tp-size 8 \
  --enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce
```

---

## 6. 学习要点（怎么读懂这套调优）

1. **config 文件 = shape → batch → tiling 参数的映射**。`E` 是 expert 数（含 shared），`N` 是中间维/2，`dtype`/`block_shape` 由量化决定。
2. **"Miss" 不一定是坏事**：要看它回退到哪。回退到「旧版 triton 的 tuned config」≈ 已接近最优；回退到「default」才是真·未调优。
3. **batch size 是关键变量**：config 按 `M`（batch）分桶，decode（小 batch）和 prefill（大 batch）的最优 tiling 不同。只 tune 你关心的 batch。
4. **tune 的时间主要花在 triton JIT 编译**（每个新 tiling ~秒级），不是 kernel 跑的时间。1280 config × 单 batch ≈ 1h；全 batch ≈ 18h。
5. **config 写入是相对路径**：生成后要手动挪进 `configs/triton_3_6_0/` 或挂载回 serving 才能生效。

---

## 7. 收益总结（端到端实测版）

### kernel 级（micro-benchmark，仅供参考）

| 项 | kernel 级 | 说明 |
|---|---|---|
| MoE up 投影（tuned vs 3.5.1）| +3~8% | 合成随机张量测的，不反映真实负载 |
| MoE down 投影（开 TMA）| +8~16% | 同上 |
| MoE kernel 合计 | ~12% | t0+t1 = 122.5→107.7 us |

### 端到端（真实 serving 吞吐，tokens/sec）❌ 不成立

| 并发 | fallback 3.5.1 | tuned 3.6.0+TMA | 差异 |
|---|---|---|---|
| 4 | 193.8 | 187.7 | -3.1% |
| 8 | 326.3 | 323.0 | -1.0% |
| 16 | 565.4 | 559.7 | -1.0% |
| 32 | 1011.6 | 904.1 | **-10.6%** |

**结论**：kernel 级 ~12% 是 micro-benchmark 假象（合成张量 + 分开的 up/down kernel，非真实 fused kernel + 真实路由）。端到端实测 tuned config **无收益甚至略慢**。

### 最终决策

**保留官方 3.5.1 fallback，不使用重新 tune 的 config。**（`.13` 的 tuned config 挂载已移除，serving 已回退 fallback）

唯一保留价值的：**DeepGEMM 预编译**（冷机器省 10-20min JIT，与 config 无关）。

---

**教训**：micro-benchmark 的 kernel 加速 ≠ 端到端加速，必须端到端实测才算数。本次调优的最大产出是这个方法论结论，而非性能收益。

### 方法复盘：调优方式有问题吗？

**方法本身（官方 `tuning_fused_moe_triton.py` autotune）是标准做法，没错。但有三处问题：**

1. **流程错误**：拿到 kernel 级结果后没先端到端验证就下了“12% 收益”结论。正确流程应是「tune → 立刻端到端 benchmark → 再谈收益」。
2. **工具固有局限**：autotune 在「合成随机张量 + 独占 GPU + 分离 kernel」条件下找最优，与真实服务（真实 activation + 多请求竞争 + 融合 kernel）存在系统性偏差——天然高估“大 tile + TMA”的收益、低估“低 occupancy 在竞争下”的代价。
3. **脚本用错（次要）**：down 投影用了 sep 脚本（给“分离 kernel”调），但 serving 跑的是“融合 kernel”，上下文不匹配。

**为什么 tune 完反而慢**：不是调优动作错，而是「微基准 proxy ≠ 端到端目标」+「没做端到端验证」两条叠加——把 kernel 级 proxy 当成了端到端结果。

---

## 8. 附录：每个性能数值怎么测的（可复现）

> 所有数值的来源脚本 + 精确命令 + 测量条件都列在这里。每个数字对应报告里的哪张表也标了。

### A.1 端到端吞吐（§7 的 before/after 表）

**来源**：自定义脚本 `bench_e2e.py`，直接打在真实 serving 的 `/generate` 上。

```python
import urllib.request, json, time, concurrent.futures
URL = "http://127.0.0.1:10100/generate"
PROMPT = "请详细解释一下什么是张量并行和专家并行，以及它们的区别。"
MAX_NEW_TOKENS = 128
def run(concurrency, num_req):
    def one(_):
        body = json.dumps({"text": PROMPT,
            "sampling_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0}}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        return d["meta_info"]["completion_tokens"], time.time() - t0
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        rs = list(ex.map(one, range(num_req)))
    el = time.time() - start
    tot = sum(r[0] for r in rs)
    print(f"conc={concurrency} throughput={tot/el:.1f} tok/s avg_lat={sum(r[1] for r in rs)/num_req*1000:.0f} ms")
run(4, 8)          # 预热
for c in [4, 8, 16, 32]:
    run(c, 32)
```

运行：`docker run --rm --network host -v /root/bench_e2e.py:/bench.py --entrypoint python3 <镜像> /bench.py`

**条件**：固定 prompt（~30 token）+ `max_new_tokens=128`（decode 为主）+ `temperature=0`；4 个并发档各 32 请求（另有 8 个预热）。吞吐 = 总 output tokens / 总墙钟时间。测了两遍：一遍 serving 挂 tuned config（after），一遍去掉挂载回退 fallback（before）。

### A.2 kernel 级 up/down 时间（§2.4 表）

**来源**：官方 `tuning_fused_moe_triton_sep.py` 的 `--configs` 模式（测单个 config 的 up/down 时间）。

前置——生成 topk_ids（sep 脚本必需）：
```python
import torch, os
os.makedirs("/root/topk_ids", exist_ok=True)
num_layers, dense_layers = 61, 3
moe_layers = num_layers - dense_layers
for i in range(100):
    torch.save(torch.randint(0, 257, (16, 9), dtype=torch.int32),
               f"/root/topk_ids/topk_ids_layer{i % moe_layers + dense_layers}_idx{i // moe_layers}.pt")
```

命令（两个 config 各跑一次）：
```bash
# 3.5.1 fallback 的 bs=16 参数 (M16 N64 K128 G1 w4 s3)
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
  --model /data0/models/glm52-step111-fp8 --tp-size 8 --dtype fp8_w8a8 \
  --batch-size 16 --topk-ids-dir /root/topk_ids --configs 16 64 128 1 4 3
# tuned 的 bs=16 参数 (M16 N128 K128 G1 w4 s4)
  ... --configs 16 128 128 1 4 4
```
输出 `t0=.. t0_tma=.. t1=.. t1_tma=..`（单位 us；t0=up 非TMA、t0_tma=up TMA、t1=down 非TMA、t1_tma=down TMA）。

**条件**：`num_iters=100`（每 config 跑 100 次取均值）；`torch.randn(16, 6144)` 合成输入 + 随机 gating。⚠️ **测的是「分离 kernel + 合成张量」，不是 serving 的融合 kernel**——这正是 kernel 级数字端到端不成立的根源。

### A.3 "13/14 个 batch 不同"（§2.5）

**来源**：`diff_cfg.py` 逐 batch 对比 tuned 3.6.0 config 与镜像内 3.5.1 config 的 6 个 tiling 参数，统计不同的 batch 数。

### A.4 DeepGEMM 预编译耗时 "~4.3 分钟"（§3.4）

**来源**：`python -m sglang.compile_deep_gemm --model-path ... --tp-size 8 --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce` 的日志时间戳差（`fired up and ready` 到 `compilation finished` 之间的墙钟时间）。

### A.5 "~12% MoE kernel 合计"（§2.4 表最末行）

**来源**：A.2 的 `t0 + t1` **算术求和**（非单次测量）：122.52 = 71.63 + 50.89，107.68 = 65.98 + 41.70。注意这是"把 up/down 分开测的时间相加"，不代表真实融合 kernel 的一次端到端时间。
