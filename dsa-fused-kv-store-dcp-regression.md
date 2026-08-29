# DSA 融合 KV 写 kernel 丢失 DCP 映射：dcp>1 乱码根因排查

> 📑 **逐页卡片版（建议先看）**：[dsa-fused-kv-store-dcp-regression.html](dsa-fused-kv-store-dcp-regression.html) —— 14 页漏斗式幻灯片（现象 → 变量隔离 → 虚拟 id → 契约 → 根因 → 定位），每页一张图 + 排查进度条。本文是详细证据版。

> 排查对象：[`0a60805df4`](https://github.com/san-tian/sglang/commit/0a60805df4)（DSA fused quant+store，即 indexer KV share 优化）。现象：dcp=4 下服务开头输出正常，跑一段时间突然乱码；关掉该优化则 dcp=4/8 均正常；dcp=8 下开优化暂时未复现乱码。结论一句话：新融合 kernel 丢掉了旧写入路径里的 DCP 虚拟 id → 本地行号映射（owner mask + `loc // dcp`），DCP 下把虚拟 id 直接当物理行号写，先错位、后越界。**这不是 dcp 值特异的 bug，kernel 对任何 dcp>1 都不正确，dcp=8 的"正常"是水位没越过越界阈值的假象。**

## TL;DR

- **根因**：`fused_dsa_quant_store` 直接以 `cache_loc = loc` 为行号写入 KV 池，而 DCP 下 `loc` 是虚拟 id 域（0 ~ size×dcp）。旧路径 `set_mla_kv_buffer_triton` 有 `loc % dcp == rank` 归属过滤 + `loc // dcp` 行号换算，新 kernel 两者皆无。
- **两种失效模式**：虚拟 id < size 时写入"错位行"（读侧按 `v//dcp` 取数取不到正确数据）；虚拟 id ≥ size 时**越界写**，打穿本 layer buffer 之后的相邻 GPU 内存（其他 layer 的 KV、相邻 pool），表现为"开头正常、突然乱码"。
- **为什么 dcp=8 看起来没事**：allocator 从低位顺序发虚拟页，越界阈值是累计分配超过 `size`（与 dcp 值无关）。dcp=8 的运行没跑到阈值，不是正确。
- **修复**：kernel 内补回 DCP mask + divide（与旧 kernel 逐行对齐），或 `dcp_enabled()` 时回退旧两步路径。另需补回 `reserved_skip_index`（slot 0 跳过）行为。

## 排查路径（narrow down）

从现象到根因的五层收敛，正文按此展开：

| 层 | 问题 | 答案 | 对应章节 |
|---|---|---|---|
| ① 变量隔离 | 什么变了？ | 唯一变更 = [commit 0a60805df4](https://github.com/san-tian/sglang/commit/0a60805df4) 的写路径（两步→一步，读侧零改动） | 背景 |
| ② loc 语义 | DCP 下写入的 loc 是什么？ | 虚拟 id（0 ~ size×dcp），不是物理行号 | DCP 下 KV 写入的既有契约 |
| ③ 正确写法 | 全库怎么做？ | 归属过滤 v%dcp==rank + 行号换算 v//dcp | DCP 下 KV 写入的既有契约 |
| ④ 根因 | 新 kernel 做了吗？ | 两样都没做（还丢了 slot 0 跳过） | 新 kernel 丢掉了什么 |
| ⑤ 定位 | 错误落在哪、何时发生？ | v<size 错位写、v≥size 越界写；拐点 = 水位过 size | 失效机制 / dcp=8 假象 |

现象的三条线索：「关掉就正常」指向变更隔离（①）；「开头正常→突然乱码」指向状态积累型错误（⑤ 的时间形态）；「dcp=8 暂未复现」是误导项（拐点与 dcp 无关，见失效机制一节）。

## 背景：fp8 下 indexer 与 attention 共享的 KV 布局

GLM-5.2 开 `--kv-cache-dtype fp8_e4m3` 后，DSA 层的 KV 行变成 656 字节混合布局，indexer 的 topk 打分和稀疏 attention 读同一份 KV（即本次的 "indexer KV share"）：

```
[k_nope fp8(512B) | per-128 块的 fp32 scale(16B) | k_rope bf16(128B)] = 656B
```

行内偏移在 kernel 里是硬编码常量：`_DIM_NOPE = 512`、`_SCALE_BYTES = 16`、`_ROPE_BYTES = 128`、`_TOTAL_BYTES = 656`（`python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py:15-24`）。布局公式与池配置一致：`kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2`（`python/sglang/srt/mem_cache/kv_cache_configurator.py:2235-2250`，512+16+128=656）。

本次 commit 只动了写路径，读侧零改动：

```
commit 0a60805df4 [kernel] DSA fused quant+store: ...
 python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py | 173 +++++
 python/sglang/srt/mem_cache/memory_pool.py                       |  20 +--
 2 files changed, 181 insertions(+), 12 deletions(-)
```

所以排查范围天然收敛：**写入路径的语义变化**。

## DCP 下 KV 写入的既有契约（全部有源码证据）

DCP（decode context parallelism）下，token 的 KV 按轮转方式分布在 dcp 个 rank 上。四条独立源码证据构成同一契约：

**证据 1：allocator 发放的是虚拟 id，上限 size×dcp**

```python
# python/sglang/srt/mem_cache/kv_cache_configurator.py:1775-1783
token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
    sizes.max_total_num_tokens * get_parallel().attn_dcp_size,   # 容量 ×dcp
    page_size=get_schedule().page_size * get_parallel().attn_dcp_size,  # 64×dcp
    ...
)
```

**证据 2：物理池每个 rank 只有 `size` 行**

```python
# python/sglang/srt/mem_cache/memory_pool.py:4105-4113
# The padded slot 0 is used for writing dummy outputs from padded tokens.
self.kv_buffer = [
    torch.zeros(
        (self.size + self.page_size, 1, self.kv_cache_dim),   # size+64 行，不乘 dcp
        dtype=self.store_dtype, device=self.device,
    )
    for _ in range(self.layer_num)
]
```

**证据 3：写侧约定——"kernel 自己除 dcp"（注释原话）**

```python
# python/sglang/srt/mem_cache/memory_pool.py:4240-4245
# loc is widened under DCP; the kernel divides by the world size itself.
maybe_detect_oob(
    loc, 0,
    (self.size + self.page_size) * get_parallel().attn_dcp_size,   # 允许虚拟 id 进来
    "set_mla_kv_buffer (MLA)",
)
```

旧路径（被替换掉的那个）正是按这个约定实现的：

```python
# python/sglang/kernels/ops/kvcache/mla_buffer.py:39-43（set_mla_kv_buffer_kernel）
loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
is_valid = (loc != reserved_skip_index) & (loc % DCP_WORLD_SIZE == DCP_RANK)
safe_loc = tl.where(is_valid, loc, 0)
safe_loc = safe_loc // DCP_WORLD_SIZE          # 虚拟 id → 本地物理行
```

DCP 参数取自真实运行时配置，不是写死：

```python
# python/sglang/kernels/ops/kvcache/mla_buffer.py:173-174
DCP_RANK=get_parallel().attn_dcp_rank,
DCP_WORLD_SIZE=get_parallel().attn_dcp_size,
```

**证据 4：读侧同样按 `v // dcp` 换算（官方 DCP kernel）**

```python
# python/sglang/kernels/ops/attention/dcp_kernels.py:98-105（create_mla_kv_page_table_for_dcp）
global_positions = DCP_RANK + page_offsets * PHYSICAL_PAGE_SIZE * DCP_SIZE
virtual_locs = tl.load(
    req_to_token_ptr + req_pool_index * req_to_token_stride + global_positions,
    mask=mask, other=0,
)
physical_pages = virtual_locs // DCP_SIZE // PHYSICAL_PAGE_SIZE   # v//dcp 再换页
```

**旁证：全库其他 KV 写路径都遵守这条契约**

```python
# python/sglang/srt/mem_cache/memory_pool.py:4172-4176（set_kv_buffer）
if parallel.dcp_enabled:
    valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank
    if not valid_mask.all():
        loc = loc[valid_mask]; cache_k = cache_k[valid_mask]
```

```python
# python/sglang/srt/layers/attention/trtllm_mla_backend.py:1030-1033
# DCP cyclic KV sharding: virtual loc -> owner mask + loc//world
# (identity when attn_dcp_size == 1).
dcp_world_size=parallel.attn_dcp_size,
dcp_rank=parallel.attn_dcp_rank,
```

## 新 kernel 丢掉了什么（逐条对比）

| 行为 | 旧路径 `set_mla_kv_buffer_kernel` | 新路径 `_fused_dsa_quant_store_kernel` |
|---|---|---|
| 归属过滤 | `loc % DCP_WORLD_SIZE == DCP_RANK`（mla_buffer.py:40） | ❌ 无 |
| 虚拟→本地行号 | `safe_loc = safe_loc // DCP_WORLD_SIZE`（mla_buffer.py:42） | ❌ 无，`cache_loc = loc` 原样使用 |
| DCP 参数 | `DCP_RANK/DCP_WORLD_SIZE` constexpr，launch 时取 `get_parallel()`（mla_buffer.py:173-174） | ❌ kernel 签名（fused_dsa_quant_store.py:31-47）与 launch（:104-120）均无任何 dcp 参数 |
| slot 0 跳过 | `loc != reserved_skip_index`（mla_buffer.py:40） | ❌ 无 |

新 kernel 的关键代码：

```python
# python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py:51
cache_loc = tl.load(loc_ptr + token_id).to(tl.int64)   # 虚拟 id，原样

# :60-61 —— nope/scale 直接以 cache_loc 为行号
tl.store(nope_buf_ptr + cache_loc * nope_buf_stride_0 + offs, y_q)
tl.store(scale_buf_ptr + cache_loc * scale_buf_stride_0 + block_id, y_s)

# :70-71 —— rope 同样
tl.store(rope_buf_ptr + cache_loc * rope_buf_stride_0 + offs, data, mask=mask)
```

而调用点不再经过任何 DCP 处理：

```python
# python/sglang/srt/mem_cache/memory_pool.py:4204-4216（_write_mla_kv_buffer）
elif self.dsa_kv_cache_store_fp8:
    from sglang.kernels.ops.attention.dsa.fused_dsa_quant_store import (
        fused_dsa_quant_store,
    )
    fused_dsa_quant_store(
        cache_k_nope.view(-1, 512),
        cache_k_rope.view(-1, 64),
        dst_buffer,
        loc,          # ← DCP 下的虚拟 id，直接透传
    )
```

commit 之前的同一位置（[`git show 0a60805df4^`](https://github.com/san-tian/sglang/commit/0a60805df4)）走的是 DCP 感知的两步路径：

```python
cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(cache_k_nope, cache_k_rope)
set_mla_kv_buffer_triton(    # ← 内部做 loc % dcp == rank 过滤 + loc // dcp 换算
    dst_buffer, loc, cache_k_nope_fp8, cache_k_rope_fp8
)
```

> ⚠ 诚实标注：新 kernel 的量化部分（per-block scale、fp8 clamp、bf16 rope 直拷）与旧路径字节一致，commit 自带的 `test_correctness` 也验证了这一点——但该测试只在 **dcp=1** 下等价，因为它只对比"同一 loc 下两路径写同一行"；DCP 下 loc 语义变化，测试覆盖不到。

## 失效机制：错位写 + 越界写

DCP 下虚拟 id 的几何：`v = p·(64·dcp) + i`（p 为虚拟页号，i 为页内偏移），归属 rank = `v % dcp`，本 rank 物理行 = `v // dcp`。新 kernel 直接写第 `v` 行：

- **模式 A（错位写）**：`v < size` 时写入物理行 `v`，而读侧（[dcp_kernels.py:105](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/attention/dcp_kernels.py#L98-L105)、trtllm 路径、以及一切按契约实现的消费方）按 `v // dcp` 取数 → 读到别的 token 或从未写入的行。此阶段读侧如果恰好也用原始 `v` 取值（本分支 DSA 读路径没有任何 DCP 换算代码，`dsa_backend.py` 与 `dsa/` 目录 grep 不到 dcp 引用），读写可以暂时自洽，输出正常。
- **模式 B（越界写）**：`v ≥ size` 时写入超过该 layer buffer 的 `size+64` 行范围，打穿后面紧邻的 GPU 内存（同一 `kv_buffer` 列表里后续 layer 的 buffer、indexer 缓存、其他 pool）。allocator 从低位顺序发页，**累计分配超过 `size` 即进入模式 B** —— 服务跑一段时间必达，这就是"开头正常、突然乱码"的拐点。

两个模式对 dcp=4 和 dcp=8 完全对称（越界阈值都是累计分配 > size），所以 "dcp=8 能跑" 只说明那组实验没跑到水位：

> ⚠ 诚实标注：dcp=8 未复现乱码的具体原因（读侧巧合自洽的持续区间、实验负载/时长差异）无法纯静态分析确定，需要压测验证；但 kernel 对 dcp>1 不正确的结论不依赖这一点——写侧丢失 mask+divide 是确定性事实，只有 dcp=1 时 `v//1 == v` 才数学上成立。

## 旁证：同一 bug 类在生产分支有前科

生产分支（b300-glm52）的 [`37231e4884`](https://github.com/MindLab-Research/sglang/commit/37231e4884) fix(dcp): decode correctness — index_k full-domain mapping + topk -1-lane redirect 记录了同构 bug：DSA index_k 写路径曾只转换 `>= pool.size` 的 id，低位虚拟 id 原样落位 → index_k 错位 → 稀疏 topk 选错页 → 乱码。修复方式是"无条件转换 + owner 检查"。本次 fused kernel 是同一类错误的更强版本：**连高 id 的转换都没有**，高 id 直接变成越界写。

另外注意：生产分支还有本分支（kernel/dsa-fused-quant-store）缺失的一批 DCP 读侧修复（`_localize_index_k_cache_locs`、`_repair_global_kv_slots_`、page-table read repair，见 [`3a6d3f281f`](https://github.com/MindLab-Research/sglang/commit/3a6d3f281f) / [`37231e4884`](https://github.com/MindLab-Research/sglang/commit/37231e4884)）。如果要把这个 fused kernel 往生产分支 rebase，写侧补 DCP 只是第一步，读侧这批修复必须一并核对，否则乱码会以另一种形式重现。

## 修复建议

**方案 A（推荐）：kernel 内补回 DCP，与旧 kernel 逐行对齐**

在 dispatch 层取运行时配置传入（参照 mla_buffer.py:173-174），kernel 内：

```python
# 参照 mla_buffer.py:39-43 的既有写法
loc = tl.load(loc_ptr + token_id).to(tl.int64)
is_valid = (loc != reserved_skip_index) & (loc % DCP_WORLD_SIZE == DCP_RANK)
safe_loc = tl.where(is_valid, loc, 0) // DCP_WORLD_SIZE
# 三个 region（nope/scale/rope）的 tl.store 都加 mask=is_valid
```

**方案 B（保守）：DCP 下回退旧路径**

`_write_mla_kv_buffer` 里 `if get_parallel().dcp_enabled` 走 `quantize_k_cache_separate + set_mla_kv_buffer_triton`，fused 仅用于 dcp=1。

**附带修复：**

- 补回 `reserved_skip_index`：旧 kernel 跳过 slot 0（CUDA-graph padding 预留槽，`memory_pool.py:4105` 注释），新 kernel 会写它。
- HIP 分支同样缺 DCP：`set_mla_kv_buffer_fp8_quant_kernel`（[mla_buffer.py:206-208](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/kvcache/mla_buffer.py#L206-L208)）也只有 `loc != reserved_skip_index` 过滤、没有 `// dcp` 换算——AMD 上开 DCP+fp8 share 会踩同一个坑（存量问题，非本次 commit 引入）。

**验证建议**：dcp=4 + 长时压测，监控累计分配 token 数越过 `size` 后的输出质量（预期在拐点处出现乱码）；对照 dcp=1 应全程正常。

## 证据索引

| 断言 | 证据位置 | 关键代码 |
|---|---|---|
| allocator 发虚拟 id（容量 size×dcp，页 64×dcp） | [`kv_cache_configurator.py:1775-1783`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1775-L1783) | `PagedTokenToKVPoolAllocator(max_total_num_tokens * attn_dcp_size, page_size=page_size * attn_dcp_size)` |
| 物理池每 rank 仅 size+64 行 | [`memory_pool.py:4105-4113`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/srt/mem_cache/memory_pool.py#L4105-L4113) | `torch.zeros((size + page_size, 1, kv_cache_dim))` |
| 旧写 kernel 有 DCP mask+divide | [`mla_buffer.py:39-43`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/kvcache/mla_buffer.py#L39-L43) | `is_valid = (loc != reserved) & (loc % DCP_WORLD_SIZE == DCP_RANK)`；`safe_loc // DCP_WORLD_SIZE` |
| 旧写 kernel 的 DCP 参数来自运行时 | [`mla_buffer.py:173-174`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/kvcache/mla_buffer.py#L173-L174) | `DCP_RANK=get_parallel().attn_dcp_rank` |
| 读侧契约：v//dcp 换算 | [`dcp_kernels.py:98-105`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/attention/dcp_kernels.py#L98-L105) | `physical_pages = virtual_locs // DCP_SIZE // PHYSICAL_PAGE_SIZE` |
| 新 kernel 原样使用 loc | [`fused_dsa_quant_store.py:51,60-71`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py#L51-L71) | `cache_loc = tl.load(loc_ptr + token_id)`；`store(buf + cache_loc * stride ...)` |
| 新 kernel 无任何 dcp 参数 | [`fused_dsa_quant_store.py:31-47,104-120`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py#L31-L47) | kernel 签名与 launch 均无 DCP_RANK/WORLD_SIZE |
| 调用点直接透传 loc | [`memory_pool.py:4204-4216`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/srt/mem_cache/memory_pool.py#L4204-L4216) | `fused_dsa_quant_store(..., dst_buffer, loc)` |
| commit 前旧路径走 DCP 感知 kernel | [`git show 0a60805df4^`](https://github.com/san-tian/sglang/commit/0a60805df4) | `quantize_k_cache_separate + set_mla_kv_buffer_triton` |
| commit 只改写侧（读侧零改动） | [`git show 0a60805df4 --stat`](https://github.com/san-tian/sglang/commit/0a60805df4) | 2 files, +181/-12 |
| 656 字节布局公式 | [`kv_cache_configurator.py:2235-2250`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/srt/mem_cache/kv_cache_configurator.py#L2235-L2250)；[`fused_dsa_quant_store.py:15-24`](https://github.com/san-tian/sglang/blob/0a60805df4/python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py#L15-L24) | 512 + 512//128*4 + 64*2 = 656 |
| 同类 bug 前科（生产分支） | [`37231e4884`](https://github.com/MindLab-Research/sglang/commit/37231e4884) commit message | "index_k landed at slot x instead of its local slot ... probabilistic confident-wrong garbling" |

> 📐 数据来源：本报告全部证据来自本地 sglang 仓库（分支 `kernel/dsa-fused-quant-store`，HEAD=`0a60805df4`）的源码静态核对，2026-08-27；无上机实验。dcp=8 未复现乱码的机制解释为推断（文中已标注 ⚠）。源码链接锚定：fork [san-tian/sglang](https://github.com/san-tian/sglang/commit/0a60805df4) @ 0a60805df4（该 commit 即分支 HEAD）；生产分支 commit 锚定 [MindLab-Research/sglang](https://github.com/MindLab-Research/sglang)（b300-glm52）。
