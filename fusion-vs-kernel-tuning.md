# 算子融合 vs Kernel 调优：GLM-5.2 端到端吞吐 +51%

> 本文是《sglang MoE 内核调优报告》的续篇。前一篇调 `fused_moe_triton` 的 tiling 参数，端到端 0 收益；这一篇换两个 flag 打开算子融合，端到端输出吞吐 +51%、TPOT -36%。两篇合起来回答一个问题：**在 GLM-5.2 这种大 MoE 上，什么优化才真正划算。**

## TL;DR

| 结论 | 数据 |
|---|---|
| 算子融合（allreduce 融合 + FlashInfer TRTLLM MoE）端到端 +51% | 输出吞吐 524 → 793 tok/s |
| TPOT -36%，TTFT -16% | 26.7 → 17.0 ms / 3884 → 3255 ms |
| 融合此前一直被一个 flashinfer 版本错配卡死 | python 0.6.14 vs cubin/jit-cache 0.6.12 |
| 修复只需两行 pip | 见下 |
| kernel tune（tiling 搜索）端到端 0 收益 | 前一篇结论 |

## 核心结论

**算子融合的收益来自"删掉整条通信/访存往返"，kernel tune 的收益来自"同一个 kernel 内部换 tile"。** 前者省的是 kernel launch、中间 DRAM 往返、allreduce 通信本身，后者省的只是 kernel 内部的访存模式——在大 MoE + TP8 的解码场景里，瓶颈在前者不在后者。

用 profile 数据可以一眼看清：decode 阶段 `ncclDevKernel_AllReduce` 占 **23.6%**（头号），而 `fused_moe_kernel` 只占 19.1%；prefill 阶段 allreduce 更是占到 **95.9%**。也就是说，TP8 的通信是头号瓶颈，而 allreduce 融合恰好就是干掉这部分开销的开关。

> **📐 数据来源**：`llm-torch-profiler-analysis` skill 对 .13 现役 serve 的 trace 三表分析（`analyze_llm_torch_profile.py --input`，8×rank，DECODE/EXTEND 两 stage）。kernel 占比为 GPU 时间份额。

## 完整环境

复现本文所有数字需要的环境，一处不省：

### 硬件

| 项 | 值 |
|---|---|
| 机器 | 2 台，`yotta-arc-b200-13`（38.255.28.13）/ `yotta-arc-b200-8`（38.255.28.8） |
| GPU | 每台 8× NVIDIA B200，179G/卡 |

### 镜像与软件栈

| 组件 | 版本 |
|---|---|
| 镜像 | `b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200`（digest `1839d8f3c89b`） |
| sglang 代码 | `sgl-project/sglang` release/v0.5.15，HEAD `0b3bb0c`（cherry-pick GLM-5.2 MTP prefill CP #30992） |
| torch | 2.11.0+cu130 |
| apache-tvm-ffi | 0.1.11 |
| 模型 | `/data0/models/glm52-step111-fp8`：E=257 MoE，topk=8，hidden=6144，fp8_w8a8，block_shape[128,128]，MLA + DSA 稀疏注意力 |

### flashinfer 三件套（修复前 → 后）

flashinfer 拆成三个独立包，**三者必须同版本**。镜像里错配：

| 包 | 修复前 | 修复后 |
|---|---|---|
| `flashinfer-python`（调用层） | 0.6.14 | 0.6.14 |
| `flashinfer-cubin`（native） | **0.6.12** ❌ | **0.6.14** ✅ |
| `flashinfer-jit-cache`（native AOT） | **0.6.12+cu130** ❌ | **0.6.14+cu130** ✅ |

> **📐 数据来源**：`import flashinfer_cubin as c; c.__version__` / `import flashinfer_jit_cache as j; j.__version__`，在镜像内跑出。错配由 `flashinfer/jit/env.py` 的 `_verify_flashinfer_cubin_version()` 与 `_get_aot_dir()` 两处版本检查负责——它们分别对 cubin 和 jit-cache 做一致性校验。

## 根因：融合被一个版本错配卡死

镜像里 `FLASHINFER_DISABLE_VERSION_CHECK=1` 这个环境变量，把 flashinfer 自身的「cubin ↔ python」「jit-cache ↔ python」两次版本检查关掉了。于是镜像能 load 起来，但一跑到融合 kernel 的 forward，python 0.6.14 的 API 去调 0.6.12 的 native kernel，直接 tvm_ffi 签名错位：

```
TypeError: trtllm_fp8_block_scale_moe(...) Expected 29 but got 33 arguments
TypeError: trtllm_mnnvl_allreduce_fusion(...) Expected 16 but got 21 arguments
```

这就是前一篇报告里那排「tvm_ffi Expected N but got M」坑的同一个根因——**不是调优取舍，是版本错配导致融合根本跑不起来**，于是只能被迫 `--moe-runner-backend triton` + 全 disable。

## 修复：两行 pip

```bash
pip install --force-reinstall "flashinfer-cubin==0.6.14"     --index-url https://flashinfer.ai/whl
pip install --force-reinstall "flashinfer-jit-cache==0.6.14" --index-url https://flashinfer.ai/whl/cu130
```

注意第二个的索引是 `cu130` 不是 `cu13`（CUDA 13.0.1 → `CUINDEX=130`）。修完后 `import flashinfer` 的版本检查自然通过，不再需要 `FLASHINFER_DISABLE_VERSION_CHECK=1`。

## 对比的两套 serve 参数

只差三处，其余完全一致（TP8、fp8 KV、hicache、page_size 64 等）：

| 开关 | baseline（融合关闭） | fusion（融合开启） |
|---|---|---|
| MoE 后端 | `--moe-runner-backend triton` | `--moe-runner-backend flashinfer_trtllm` |
| allreduce 融合 | `--enforce-disable-flashinfer-allreduce-fusion` | `--flashinfer-allreduce-fusion-backend auto` |
| custom allreduce | `--disable-custom-all-reduce` | （去掉） |

> **📐 数据来源**：完整命令见文末附录。baseline 跑在 .13 现役容器 `sglang-step111`，fusion 跑在 .8 容器 `sglang-fusion-b8`（启动脚本先执行上面两行 pip 再 launch_server）。

## 端到端 benchmark 结果

同参数 `bench_serving`（8192 输入 + 1024 输出，64 prompts，concurrency 16）：

| 指标 | baseline | fusion | 差异 |
|---|---:|---:|---:|
| **输出吞吐 (tok/s)** | 524.62 | **793.20** | **+51%** |
| 峰值输出吞吐 (tok/s) | 656 | 1072 | +63% |
| **TPOT 均值 (ms)** | 26.72 | **17.00** | **-36%** |
| TPOT 中位 (ms) | 26.58 | 16.95 | -36% |
| **TTFT 均值 (ms)** | 3884.02 | **3254.78** | **-16%** |
| E2E 均值 (ms) | 31215 | 20645 | -34% |
| 总 token 吞吐 (tok/s) | 4721 | 7138 | +51% |

> **📐 数据来源**：`python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 10100 --model glm52-step111-fp8 --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts 64 --max-concurrency 16 --warmup-requests 64 --flush-cache`。baseline 实测 2026-08-25 于 .13，fusion 实测 2026-08-25 于 .8。

## 为什么这次有用、上次没用

| 维度 | kernel tune（前一篇） | 算子融合（本篇） |
|---|---|---|
| 调的对象 | `fused_moe_triton` 内部 6 个 tiling 参数 | 整条 allreduce + MoE 计算链路 |
| 省掉的开销 | kernel 内部访存（micro-benchmark 可见，端到端被稀释） | kernel launch + 中间 DRAM 往返 + **allreduce 通信本身** |
| profile 天花板 | MoE 占 19%，调它上限低 | allreduce 占 23.6%（decode 头号）/ 95.9%（prefill），上限高 |
| 端到端结果 | 0 收益（大 tile 伤 occupancy 抵消） | **+51%** |

方法上的差别一句话：**先 profile 定位真正瓶颈，再决定调什么**。前一篇是先选了 MoE kernel 去 tune，profile 一看才发现瓶颈根本不在那；这一篇是 profile 先指出 allreduce 是头号，才去动融合开关，一击命中。

## 附录：完整 serve 命令

fusion 变体（.8，改动处以注释标出）：

```bash
python3 -m sglang.launch_server \
  --model-path /data0/models/glm52-step111-fp8 \
  --served-model-name glm52-step111-fp8 \
  --host 0.0.0.0 --port 10100 --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-cache-report --page-size 64 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend flashinfer_trtllm \          # baseline 这里是 triton
  --flashinfer-allreduce-fusion-backend auto \      # baseline 这里是 --enforce-disable-flashinfer-allreduce-fusion
  --model-impl sglang \                              # baseline 还有 --disable-custom-all-reduce
  --mem-fraction-static 0.90 --max-total-tokens 5000000 \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-write-policy write_back --hicache-mem-layout page_first \
  --hicache-storage-backend file --file-storage-path /root/hicache \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 --enable-metrics
```

> **📐 数据来源**：baseline 命令取自 .13 `docker inspect sglang-step111` 的 `Cmd` 字段，fusion 命令由 baseline 只改三处生成。
