# Expected 29 but got 33：sglang 两个 flashinfer 融合报错的根因与修复

> 开 `--flashinfer-allreduce-fusion-backend auto` 崩 `Expected 16 but got 21`，开 `--moe-runner-backend flashinfer_trtllm` 崩 `Expected 29 but got 33`。两个报错同一个根因：flashinfer 三件套版本错配被一个环境变量掩盖。这篇讲怎么从报错一步步定位到根因、怎么修。

## TL;DR

| 结论 | 数据 |
|---|---|
| 两个报错同根因 | flashinfer 三个独立包版本不一致 |
| 错配 | `flashinfer-python` 0.6.14 vs `flashinfer-cubin` 0.6.12 vs `flashinfer-jit-cache` 0.6.12+cu130 |
| 掩盖机制 | `FLASHINFER_DISABLE_VERSION_CHECK=1` 关掉了 flashinfer 自己的两次版本检查 |
| 修复 | 两行 pip 把 cubin / jit-cache 对齐到 0.6.14 |
| 修复后收益 | 融合真正生效，端到端输出吞吐 524 → 793 tok/s（+51%） |

## 现象：两个 tvm_ffi 签名报错

开融合开关后，模型能 load、能起服务，但一到融合 kernel 的 forward 就崩，报错都指向 `tvm_ffi` 的 `Function.__call__`：

```
TypeError: trtllm_mnnvl_allreduce_fusion(...) Expected 16 but got 21 arguments
TypeError: trtllm_fp8_block_scale_moe(...) Expected 29 but got 33 arguments
```

「Expected N but got M」是 tvm_ffi 的典型签名错位：python 层调用时传了 M 个参数，但 native kernel 只声明了 N 个。两个报错里 `M > N`，说明**调用方（python）比 native 层新**。

> 📐 **数据来源**：.8 容器启动日志（`docker logs`）。第一个来自 `--flashinfer-allreduce-fusion-backend auto`，第二个来自 `--moe-runner-backend flashinfer_trtllm`。

## 根因：flashinfer 三件套版本错配

flashinfer 拆成三个独立安装包，三者必须同版本：

| 包 | 作用 | 实际版本 |
|---|---|---|
| `flashinfer-python` | python 调用层 | 0.6.14 |
| `flashinfer-cubin` | 预编译 kernel 二进制（CUDA 无关） | **0.6.12** ❌ |
| `flashinfer-jit-cache` | AOT 编译缓存（CUDA 相关） | **0.6.12+cu130** ❌ |

python 是 0.6.14，native（cubin + jit-cache）是 0.6.12。0.6.12 → 0.6.14 之间 flashinfer 给 MoE/allreduce kernel 加了参数，所以 0.6.14 的 python 去调 0.6.12 的 native，签名就对不上。

> 📐 **数据来源**：镜像内 `import flashinfer_cubin as c; c.__version__` 与 `import flashinfer_jit_cache as j; j.__version__`。

## 定位路径

四步从报错走到根因，每步都有源码证据。

### 1. 报错堆栈指向 native 层

```
File "flashinfer/fused_moe/core.py", line 3500, in trtllm_fp8_block_scale_moe
  result = get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(...)
File "python/tvm_ffi/cython/function.pxi", line 968, in tvm_ffi.core.Function.__call__
TypeError: ... Expected 29 but got 33 arguments
```

报错发生在 `tvm_ffi` 的 `Function.__call__`，即 native kernel 的入口。这一层不归 python 管，是 flashinfer 的 native 库签名。

### 2. 版本对不上的来源

flashinfer 的 MoE kernel 是 JIT 生成的——`gen_trtllm_gen_fused_moe_sm100_module()` 用 wheel 里的 csrc 模板现场编译出 native module，签名由模板决定。python 调用层（`core.py`）与这个模板分属两个包，各自带版本号。

### 3. 版本检查被环境变量关掉了

flashinfer 自己会检查这三件套版本一致性，位置在 `flashinfer/jit/env.py`：

- `_verify_flashinfer_cubin_version()`：检查 `flashinfer_cubin.__version__ == flashinfer.__version__`
- `_get_aot_dir()`：检查 `flashinfer_jit_cache.__version__.startswith(flashinfer.__version__)`

两处都有一行：

```python
if (
    not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
    and flashinfer_version != "0.0.0+unknown"
    and ...  # 版本不一致
):
    raise RuntimeError("... does not match ... Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check.")
```

而部署环境里恰好设了 `FLASHINFER_DISABLE_VERSION_CHECK=1`，把这两次检查都压掉了——所以 load 阶段不报错，到 forward 才崩。

### 4. 三件套版本为什么不同源

sglang 的 Dockerfile 里，三件套由不同来源控制（`docker/Dockerfile`）：

```dockerfile
ARG FLASHINFER_VERSION=0.6.14
# pyproject.toml 里硬编码：flashinfer_python[cu13]==0.6.14
# flashinfer_cache stage 里：
pip install flashinfer-cubin==${FLASHINFER_VERSION}      --index-url https://flashinfer.ai/whl
pip install flashinfer-jit-cache==${FLASHINFER_VERSION}  --index-url https://flashinfer.ai/whl/cu${CUINDEX}
```

python 版由 `pyproject.toml` 锁定，cubin/jit-cache 版由 build-arg `FLASHINFER_VERSION` 控制。构建镜像时这个 arg 传成了 0.6.12，而 pyproject 里 python 已经是 0.6.14，于是三件套错配。

> 📐 **数据来源**：`flashinfer/jit/env.py`（版本检查逻辑）、`docker/Dockerfile:22,381,385`（`ARG FLASHINFER_VERSION` 与三件套安装）、镜像内 `pip show` 结果。sglang 代码为 `sgl-project/sglang` release/v0.5.15，HEAD `0b3bb0c`。

## 修复：两行 pip

把 cubin 和 jit-cache 对齐到 python 的 0.6.14：

```bash
pip install --force-reinstall "flashinfer-cubin==0.6.14"     --index-url https://flashinfer.ai/whl
pip install --force-reinstall "flashinfer-jit-cache==0.6.14" --index-url https://flashinfer.ai/whl/cu130
```

注意第二个索引是 `cu130` 不是 `cu13`（CUDA 13.0.1 → `CUINDEX=130`，见 Dockerfile 的映射）。

修完后 `import flashinfer` 的版本检查自然通过，不再需要 `FLASHINFER_DISABLE_VERSION_CHECK=1`，也不需要 `--enforce-disable-flashinfer-allreduce-fusion` / `--disable-custom-all-reduce`。

## 验证

三层验证，从轻到重：

1. **版本一致**：`flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache` 三者均 0.6.14。
2. **import + build**：`import flashinfer` 通过版本检查；`gen_trtllm_gen_fused_moe_sm100_module().build_and_load()` 成功构建 native module，签名对齐。
3. **端到端**：开融合后正常 serve，不再崩。

> 📐 **数据来源**：验证脚本在 .8 容器内跑（`--gpus all`，`TVM_FFI_CUDA_ARCH_LIST=10.0a`），输出 `import flashinfer: OK` + `BUILD_OK`。

## 修复后的端到端收益

修复前融合被错配卡死，只能 `triton` + 全 disable；修复后融合真正生效：

| 指标 | 修复前（triton + disable） | 修复后（flashinfer_trtllm + allreduce 融合） | 差异 |
|---|---:|---:|---:|
| 输出吞吐 (tok/s) | 524.62 | 793.20 | +51% |
| TPOT 均值 (ms) | 26.72 | 17.00 | -36% |
| TTFT 均值 (ms) | 3884.02 | 3254.78 | -16% |

> 📐 **数据来源**：`python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 10100 --model glm52-step111-fp8 --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts 64 --max-concurrency 16 --warmup-requests 64 --flush-cache`。两套 serve 只差融合开关（MoE backend、allreduce fusion、custom allreduce 三处），其余完全一致。

## 完整环境

| 组件 | 版本 |
|---|---|
| 硬件 | 8× NVIDIA B200（179G/卡），`yotta-arc-b200-13` / `yotta-arc-b200-8` |
| 镜像 | `b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200`（digest `1839d8f3c89b`） |
| sglang | `sgl-project/sglang` release/v0.5.15，HEAD `0b3bb0c` |
| torch | 2.11.0+cu130 |
| apache-tvm-ffi | 0.1.11 |
| 模型 | `/data0/models/glm52-step111-fp8`（E=257 MoE，topk=8，hidden=6144，fp8_w8a8，MLA + DSA） |

## 方法论

这类「Expected N but got M」的 tvm_ffi 签名报错，本质是**调用方与 native 层版本不一致**。定位套路固定：

1. 看报错堆栈落到 `tvm_ffi Function.__call__` → 确认是 native 签名问题，不是 python 逻辑。
2. 查这个库拆了几个包、各自的版本号 → 找不一致。
3. 看这个库自己的版本检查逻辑在哪、有没有被环境变量关掉。
4. 看它的打包方式（Dockerfile / pyproject）里各包版本分别由谁控制 → 找到错配来源。

对 flashinfer 来说，就是三件套（python / cubin / jit-cache）+ 一个 `FLASHINFER_DISABLE_VERSION_CHECK` 环境变量。
