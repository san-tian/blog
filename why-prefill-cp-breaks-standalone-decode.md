# 为什么 prefill 的 CP 优化，不能带到单机 decode？

## 一句话总结

sglang 里 `--enable-prefill-cp`（配合 `--cp-strategy interleave`）和 `--enable-dsa-prefill-cp-layersplit` 是两个 **prefill 专属**的并行优化。在 PD 分离架构里它们各安其位；可一旦把服务以**单机一体（prefill+decode 同进程）**的方式起，这两个开关就会把 decode 阶段搞坏——一个让输出退化成重复，另一个让输出直接变成随机 token。根子在于：**这两个优化从语义上就跟 decode 的自回归特性冲突**，只能整个去掉。

---

## 导读：这篇文章由浅入深讲清楚 8 层

顺着读，每一层都建立在上一层之上，不需要任何并行训练/推理背景：

1. **先看现象** —— 一个"关两个参数就能修好乱码"的真实案例，先有个感性认识
2. **三个基础概念** —— 注意力与 KV cache、prefill vs decode、三种切分维度
3. **GLM-5.2 的特殊结构** —— 什么是 DSA 稀疏注意力、indexer、top-k
4. **两个 flag 的原理** —— 它们各自切的是什么
5. **一个关键事实** —— prefill CP 和 decode CP 的 token 排列是反的
6. **为什么坏** —— 三种坏法（重复 / 乱码 / 对称错位）
7. **源码证据** —— 用源码把结论钉死
8. **要不要改代码** —— 工程判断 + 结论

文末附「相关知识点速查」，可当索引用。

---

## 第 1 层：先看现象

在 8×B200 上，用 sglang 以**单机（standalone，不做 PD 分离）**跑 GLM-5.2 时，发现输出是乱的：`1+1=` 答成 `2a&apos in'ss…` 这种随机 token，或者"一句话一句话一句话…"的无限重复。

排查一圈后，把启动参数里的这两条去掉，输出立刻正常（`1+1=` → `2`）：

```bash
# ❌ 单机里不要带
--enable-prefill-cp --cp-strategy interleave
--enable-dsa-prefill-cp-layersplit
```

**为什么这两个参数在别的场景是合理的、在单机却是毒药？** 这就是全文要讲清楚的。先从地基开始。

---

## 第 2 层：三个基础概念

### 2.1 注意力与 KV cache

Transformer 的注意力：每个 token 生成一个 **Query（Q）**，去和所有历史 token 的 **Key（K）** 算相似度，再用相似度加权 **Value（V）** 得到输出。历史 token 的 K、V 会被缓存下来，就是 **KV cache**——decode 每生成一个 token，都要回头读一遍全部 KV cache。

### 2.2 prefill 和 decode 是两个完全不同的阶段

```
prefill（预填充）：一次性把整段 prompt 喂进去
  [t0 t1 t2 ... tN] ──▶ 所有 token 并行前向，边算边写 KV
  特点：token 多、一次性、可并行

decode（解码）：自回归逐 token 生成
  每 1 步：1 个新 token ──attend──▶ 全部历史 KV ──▶ 出下 1 个
  特点：每步只有 1 个 token、要读全量 KV、逐步串行
```

一句话记住区别：**prefill 是"多 token 一次性并行"，decode 是"单 token 反复读全量"。** 这个区别是后面一切冲突的根源。

### 2.3 并行的三个切分维度

模型大、放不下，就得"切"。可切的维度有三个，切的是完全不同的东西：

| 切法 | 切的是什么 | 8 张卡各拿 | 解决什么 |
|---|---|---|---|
| **TP（张量并行）** | 权重 | 1/8 权重，算整条序列 | 权重太大放不下 |
| **CP（上下文并行）** | 序列长度 | 1/8 token，算全部权重 | 序列太长算不动 |
| **layersplit（层切分）** | 78 层 | own ~10 层，其它层靠临时广播 | prefill 层间流水线 |

关键：**TP 和 CP 是正交的**。TP8 让每卡算 1/8 权重，CP8 再让每卡算 1/8 序列，两者可叠加。

---

## 第 3 层：GLM-5.2 的特殊结构（DSA）

GLM-5.2 不是普通 Transformer，它的注意力是 **DSA（Dynamic Sparse Attention，动态稀疏注意力）**：

```
GLM-5.2（78 层，hidden=6144，64 注意力头）

输入 tokens ──▶ 第 i 层（i=0..77）：
                ① QKV 投影（MLA）
                ② 索引器 indexer（32 头 × 128 维）—— 动态选出 top-2048 个相关 token
                ③ 稀疏注意力（只 attend 这 2048 个，而非全部）
                ④ MLP（MoE 专家）
            ──▶ 输出 logits
```

**DSA 的关键**：不做全注意力（每个 token 看所有 token，O(N²)），而是让 indexer 动态挑最相关的 **`index_topk=2048`** 个 token 来 attend，复杂度降到 O(N×2048)。indexer 每 4 层共享一次（`index_topk_freq=4`）。

记住：**indexer 和 top-k 是 DSA 的核心**，下面两个 flag 都是在"并行化 / 存放"这一套东西。

---

## 第 4 层：两个 flag 的原理

### 4.1 `--enable-prefill-cp --cp-strategy interleave`：切序列

打开后 `attn_cp_size` = TP 大小（TP8 → CP8），prefill 把长序列切成 8 段，每卡算 1/8 段的 DSA 稀疏注意力再归并。

```
整段 prompt: [══════ t0 … tN ══════]
                   切 8 段
   卡0      卡1      卡2   …   卡7
```

对 prefill 这是对的。**但 `attn_cp_size` 是全局量**——decode 也带着"序列已切 8 段"的布局去跑。

### 4.2 `--enable-dsa-prefill-cp-layersplit`：切层 KV

这个切的是 **KV cache 的存放**，按**层**切。启动日志写明：

```
CP layer-split rank 0/8: owns layers [0, 10) (10 layers).
  Allocates 1 transient slot to hold each non-owned layer's prefix KV
  broadcast from its owner rank.
```

含义：78 层拆给 8 卡，每卡只"拥有"约 10 层的 KV；其它层靠一个 **transient slot（临时槽位）** 接对方广播。因为 prefill 前向从第 0 层顺序走到第 77 层，这套广播能"算一次、广播一次、用一次"地流水往下传。

### 4.3 顺带引出：decode 也有自己的 CP（dcp_size）

sglang 其实**还实现了 decode 端自己的 CP**，叫 DCP，由 `--decode-context-parallel-size`（内部 `dcp_size`）控制，默认是 1（关）。它是下一层的关键对照。

---

## 第 5 层：一个关键事实——两种 CP 的 token 排列是反的

这是全文最容易被忽略、却最核心的一点：**prefill CP 和 decode CP 把 token 分给 rank 的方式是相反的**。

源码 `srt/layers/dcp/layout.py` 里 decode CP 的 owner 规则是 `pos % dcp_size == dcp_rank`（轮询分条）；而 prefill CP 是连续分块。同一批 token，两种布局的归属完全不同：

```
prefill CP（attn_cp_size=4，连续分块）:
  rank0 = [0 1 2]   rank1 = [3 4 5]   rank2 = [6 7 8]   rank3 = [9 10 11]

decode CP（dcp_size=4，轮询 pos%4）:
  rank0 = [0 4 8]   rank1 = [1 5 9]   rank2 = [2 6 10]  rank3 = [3 7 11]
```

比如 **token 4**：prefill 在 rank1，decode 却要在 rank0——同一个 token 落在不同 rank。

**在 PD 架构里，这个错位靠一次「DCP reshard」桥接**：跨机传输时 prefill 把自己连续块里的 KV 重新切条、按 `1/N token-shard` 分发给 decode 各 rank（源码 `srt/disaggregation/common/conn.py:620`）。**而 standalone 单机没有这次跨机 reshard**——这就是所有问题的根源。

这里要澄清一个常见误解：**这不是“开了 prefill CP 就不能开 decode CP”的二选一开关**。本质是**同一份 KV cache 的写方（prefill）和读方（decode）必须用同一套布局**——PD 里 prefill 机和 decode 机各用各的布局、靠 reshard 对接（所以两台机器可以同时各自开一个）；单机里没有 reshard，写与读的布局不一致就必然错位。换句话说，不是“谁把谁关掉”，而是“一份 KV 装不下两种布局，单机又缺了那个重排步骤”。

---

## 第 6 层：为什么 standalone 会坏（四种坏法）

| 只开…… | 发生了什么 | 结果 | 证据 |
|---|---|---|---|
| prefill CP（不配 decode CP） | prefill 按连续块写 KV，decode 按本地连续读 | 破坏 decode | **实测**：输出重复 |
| 只开 decode CP（不配 prefill CP） | prefill 写本地 KV，decode 按 pos%N 轮询读 | 对称错位 | 源码推断，B200 未验证 |
| 两个 CP 都开 | 写按连续块、读按轮询条，仍不一致 | 错位 | 源码推断 |
| layersplit | decode 读非拥有层时广播 gate 关闭 | 随机 token | **实测** + 源码 |

具体拆两个 flag：

**`--enable-prefill-cp`：decode 端「没接上」**。decode 每步只有 1 个 token，却要在"序列已切 8 段"的布局里读全量 KV，需要 decode-CP 来兜，DSA 的 decode 端没接上 → 注意力分值算错 → 输出一层层重复。

**`--enable-dsa-prefill-cp-layersplit`：语义上容不下 decode**。decode 是逐 token 反复读全 78 层 KV，而每卡只存 ~10 层、transient 槽只有一层容量，想读的"其它 68 层"不在卡上；且广播 gate（`is_extend`）只在 prefill 触发，decode 永远不广播 → 读到空槽 → 随机 token。

---

## 第 7 层：源码证据

（sglang 源码树 `python/sglang/`，带文件:行号）

**1. layersplit 是 prefill-only**

`srt/layers/cp/utils.py:52` docstring：「Layer split is a **prefill-CP-only optimization** for DSA...」

`srt/server_args.py:4427` 非 prefill 直接抛异常：

```python
raise ValueError(
    "--enable-dsa-cache-layer-split is only supported on PD prefill workers. "
    "Non-PD workers also run decode and require ordinary local decode cache semantics."
)
```

广播 gate `srt/layers/utils/cp_utils.py:540`：

```python
return (
    get_global_server_args().enable_dsa_prefill_cp_layersplit
    and forward_batch.forward_mode.is_extend_without_speculative()  # 只有 prefill 才广播
)
```

**2. prefill CP 与 decode CP 是两套独立 flag**

| | prefill CP | decode CP（DCP）|
|---|---|---|
| flag | `--enable-prefill-cp` | `--decode-context-parallel-size` |
| 内部字段 | `attn_cp_size` | `dcp_size` |
| 默认 | 关 | 1（关） |

**3. DCP 在 B200 没验证过**：`srt/server_args.py:3153` 注释「DCP is validated on both HIP and CUDA (**B300 SM103**)」，B200（SM100）不在范围内。

---

## 第 8 层：要不要改代码 + 结论

**layersplit：不要，也没法"实现"**。它不是缺代码，是语义上只属于 prefill。源码明说 decode 需要 ordinary local cache。任何情况 standalone 都关它。

**prefill-CP：技术上可行，但投入产出不划算**。decode-CP（DCP）代码已存在，真正缺的是在 B200(SM100) 上验证/适配；而 prefill-CP 的收益只在「单条超长 prompt + 低并发」才明显，且 **PD 分离本来就是这么设计的**（prefill 机开 CP、decode 机不开）。standalone 硬要单机同时吃，等于重造 PD 已经解决的轮子。

**结论**：

```bash
# 单机一体服务，三个开关必须全去掉
# ❌ --enable-prefill-cp --cp-strategy interleave
# ❌ --enable-dsa-prefill-cp-layersplit
```

去掉后输出恢复正常。哪天单机发现"输出变成重复 / 乱码"，先查这两条——它们比采样参数、fp8 scale 都更可能是元凶。

---

## 附录：相关知识点速查

| 概念 | 一句话 | 与本文关系 |
|---|---|---|
| **注意力 / KV cache** | token 用 Q 找 K 算相似度、加权 V；历史 K/V 缓存 | 第 2 层地基 |
| **prefill / decode** | 多 token 并行 vs 单 token 自回归 | 冲突根源 |
| **TP（张量并行）** | 切权重 | CP 的对照 |
| **CP（上下文并行）** | 切序列长度 | 本文主角之一 |
| **DCP（decode CP）** | decode 端切 KV，轮询条 | 主角之二（对照） |
| **layersplit（层切分）** | 切层 + transient 广播 | 主角之三 |
| **DSA（动态稀疏注意力）** | indexer 挑 top-k，稀疏 attention | 模型特性 |
| **MLA（多头潜在注意力）** | 压缩 KV 的注意力变体 | GLM-5.2 用 |
| **PD 分离** | prefill/decode 分机 | reshard 发生的场所 |
| **DCP reshard** | 跨机传输时重排 KV 布局 | 单机缺失的关键步骤 |

相关阅读：

- vLLM 的 decode-CP（DCP）：《DCP 实测：消除张量并行下的 KV Cache 重复存储》（本站）
- sglang 源码：`srt/layers/dcp/`、`srt/layers/attention/dsa/`、`srt/mem_cache/cp_layersplit_pool.py`