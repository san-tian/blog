# sglang MoE 内核调优报告（GLM-5.2 step111 FP8 / B200）

> 日期：2026-08-24
> 机器：38.255.28.13（.13）+ 38.255.28.8（.8），各 8×B200（179G/卡）
> 镜像：`b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200`
> 模型：`/data0/models/glm52-step111-fp8`（E=257 expert，topk=8，hidden=6144，fp8_w8a8，block[128,128]）

---

## 0. 结论速览（TL;DR）

| 调优项 | 状态 | 收益 |
|---|---|---|
| ① MoE up 投影 config（full 18 batch）| ✅ 完成 | **13/14 个 batch 与 fallback 不同**；bs=16 up 投影 **~4-8% 快** |
| ② DeepGEMM 预编译 | ✅ 完成 | 冷机器首 serve 省 10-20min JIT |
| ③ MoE down 投影 TMA | ✅ 量化 | **down 投影开 TMA 快 ~8-16%**（bs=16）|

**核心发现（修正了中途的误判）**：`Config file not found` 回退到的是**旧版 triton 3.5.1 的 tuned config**，它不是最优的——对当前 triton 3.6.0，重新 tune 出的 config **比 3.5.1 快 3-8%**；且 down 投影原本"reusing up without TMA"，开 TMA 后**再快 8-16%**。合计 MoE kernel 约 **12% 提升**。

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
  b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200 \
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
  b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200 \
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

## 7. 收益总结

| 项 | 收益 | 说明 |
|---|---|---|
| MoE up 投影（tuned 3.6.0 vs 3.5.1）| **+3~8%** | 13/14 batch 的 config 不同 |
| MoE down 投影（开 TMA）| **+8~16%** | 日志"reusing without TMA"处 |
| **MoE kernel 合计（bs=16）**| **~+12%** | 122.5→107.7 us |
| DeepGEMM 预编译 | ✅ 跑通 | 冷机器省 10-20min JIT |

**落地方法**：把 `/root/tune_out/E=257,N=256,...json`（tuned up）和生成的 `_down` config 拷回镜像 `configs/triton_3_6_0/`（或挂载），重启 serve 即生效。

### ✅ 已落地（2026-08-25 收尾）

- up 投影 tuned config + down 投影 `USE_TMA:true` config 已生成到 `.13:/root/tuned_moe_configs/`
- serving 脚本加挂载 `-v /root/tuned_moe_configs:/sgl-workspace/.../configs/triton_3_6_0`，已重启
- 重启后 `Config file not found` 行数 = **0**，冒烟 `1+1=` → `2` 正常

**修正说明**：本报告中途曾误判"bs≤16 = 0 收益"——根因是主脚本 benchmark 回退到工具内置默认值而非 3.5.1 fallback。最终用 sep 脚本 `--configs` 直接对比两个 config 得出上面的真实收益。
