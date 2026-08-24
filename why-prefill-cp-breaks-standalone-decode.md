# 为什么 prefill 的 CP 优化，不能带到单机 decode？

## 一句话总结

sglang 里 `--enable-prefill-cp`（配合 `--cp-strategy interleave`）和 `--enable-dsa-prefill-cp-layersplit` 是两个 **prefill 专属**的并行优化。在 PD 分离架构里它们各安其位；可一旦把服务以**单机一体（prefill+decode 同进程）**的方式起，这两个开关就会把 decode 阶段搞坏——一个让输出退化成"一句话重复"式的循环，另一个让输出直接变成随机 token。问题不在"哪个值没调好"，而在**这两个优化从语义上就跟 decode 的自回归特性冲突**，只能整条去掉。

---

## 基础：三个"切分维度"是三种完全不同的东西

要理解这两个 flag，得先分清模型可以被"切"的三个维度——它们切的是**完全不同的东西**。

一次 attention 要算的是：每个 token 的 `Q` 去和所有历史 token 的 `K` 做点积，再乘 `V`。历史 token 的 `K/V` 就是 **KV cache**（每个 token 一份）。以 GLM-5.2（78 层、hidden 6144、TP8 = 8 张卡）为例：

| 切法 | 切的是什么 | 8 张卡各拿 | 什么时候有用 |
|---|---|---|---|
| **TP（张量并行）** | 权重（每个头的 `Wq/Wk/Wv`） | 每张卡 1/8 的权重，但**算整条序列** | 权重太大单卡放不下 |
| **CP（上下文并行）** | **序列长度**（token） | 每张卡 1/8 的 token，但算**全部权重** | 序列太长、attention 算不动 |
| **layersplit（层切分）** | **层**（78 层） | 每张卡 own ~10 层，其它层的 KV 靠临时广播 | prefill 的层间流水线 |

关键：**TP 和 CP 是正交的**。TP8 让每张卡只算 1/8 的权重；CP8 再让每张卡只算 1/8 的序列。合起来，prefill 阶段"长 prompt + 全权重"就被切成了"1/8 序列 × 1/8 权重"，这正是 prefill 能加速的原因。

---

## 背景：为什么 prefill 能吃 CP，decode 不能

补一个最关键的认知：**attention 的复杂度是 O(序列长度²)**。

- **prefill**：一次性把整条 prompt（比如 8 万 token）喂进去，attention 要算 8万² = 64 亿次点积。这时把序列切成 8 段，每张卡只算 1万² = 1 亿次，**8 倍加速**——所以 CP 对 prefill 是真金白银的收益。
- **decode**：每步只生成 1 个新 token，这个 token 去 attend 所有历史 KV。它的 attention 是 O(1 个新 token × 全部历史)，本来就很小。**切序列对它没有意义**——你不可能把"1 个 token"再切成 8 份去并行。所以 decode 压根不需要、也不该用"序列切分"。

这就是根子上"prefill 专属"的意思。

---

## 两个 flag 到底做了什么

### `--enable-prefill-cp --cp-strategy interleave`：沿序列维切分

`--enable-prefill-cp` 打开后，sglang 会把 `attn_cp_size` 设成等于 TP 的大小（TP8 → CP8），于是 prefill 阶段把一条长 prompt 切成 8 段，8 个 rank 各算自己的 1/8，再通信归并。对长 prompt 的 prefill 加速这是合理的。

但这里藏着一个关键事实：**这个 `attn_cp_size` 是全局生效的，不只作用于 prefill**。decode 阶段一旦也带着 CP=8 的序列切分布局去跑，就会出问题。

### `--enable-dsa-prefill-cp-layersplit`：沿层维切分

`DSA` 是 GLM-5.2 那套动态稀疏注意力（Dynamic Sparse Attention）。`--enable-dsa-prefill-cp-layersplit` 切的是**另一个维度**——不切序列，而是**把 78 层 attention 层拆给 8 个 rank，每个 rank 只"拥有"约 10 层**。

这个设计的直接证据在启动日志里：

```
CP layer-split rank 0/8: owns layers [0, 10) (10 layers).
  Allocates 1 transient slot to hold each non-owned layer's prefix KV
  broadcast from its owner rank.
```

每个 rank 只负责自己拥有的层；对其它 rank 拥有的层，它只留一个 **transient slot（临时槽位）**去接对方广播过来的 KV 前缀。这套"只算自己层 + 对方广播"的机制，是给 prefill 那种**从第 0 层到第 77 层顺序往下灌**的转发路径设计的。

---

## 实测：它们在单机下是怎么坏的

把这三个开关在单机（不带 PD 分离）里做隔离测试，三种配置的输出差异非常鲜明（都用 `temperature=0` 贪心解码，排除采样噪声）：

| 配置 | `1+1=` 的输出 | 现象 |
|---|---|---|
| 无 CP | `2` | ✅ 正常 |
| 只开 `--enable-prefill-cp --cp-strategy interleave` | `1111+1++++` | decode 退化重复 |
| 两者都开 | 随机、多语言混杂 | decode 读到错 KV |

> ⚠️ 诚实标注：其中"只开 prefill-CP"一行是本次直接隔离实测；"两者都开"的随机乱码来自最初的完整参数集观测。下面对**机制**的解释是逻辑推断，不是逐行源码确认。但即便不看原因，只看这三行对照，结论已经硬。

---

## 为什么坏：两类不同的问题

两个 flag 虽然都会在单机下坏 decode，但坏的原因不同，值得分开讲。

### prefill-CP：decode 端「没接上」

`--enable-prefill-cp` 本质是"prefill 有、decode 没有"的一类问题。

decode 每一步只有 1 个 query token，它要 **attend 到全部已生成 token** 上。而在"序列切 8 段"的 CP 布局下，这 1 个 query 要对分散的 8 段 KV 各自算、再归并。prefill 靠"所有 token 一并算 + 通信归并"行得通；但 decode 的"单 query 读全量 KV"跟"序列已被切成 8 份"之间，需要一个专门的 decode-CP 实现来兜底，**而在这个 sglang 版本里，DSA 模型的 decode 端并没有正确接上这套**。

结果就是 decode 每一步的注意力分值算错，模型在原地打转——典型表现是输出一层层重复同一个短语（`一句话一句话一句话…`）。

### layersplit：语义上就只该给 prefill 用

`--enable-dsa-prefill-cp-layersplit` 的问题更深，属于"**逻辑上就不该对 decode 开**，而不是某段代码没写好"。

回到那个 transient slot 机制：它成立的前提是 **prefill 是一次性顺序前向**——rank A 算完它拥有的第 0~9 层，把结果广播给拥有第 10~19 层的 rank B，流水线往下灌，广播完就完事。

但 decode 是自回归：每一个 token 都要**反复读全部 78 层的 KV**。layersplit 把层拆走、只留 transient 槽位后，decode 需要的那份"每层完整、稳定在位的 KV"并不存在——它读到的是切来切去、跟"全量 KV"对不上的布局。于是 logits 直接乱掉，输出变成随机 token。

**一句话区分**：

- `--enable-prefill-cp` 坏在 **decode 端这个能力没实现 / 不兼容**；
- `--enable-dsa-prefill-cp-layersplit` 坏在 **它从设计上就容不下 decode 的语义**。

---

## 源码证据：坐实上面的判断

上面是逻辑推断，下面用 sglang 源码（`python/sglang/`）把关键结论钉死。

### 1. layersplit 就是「prefill-only」，源码白纸黑字

`srt/layers/cp/utils.py:52` 的 docstring 直接写：

> "Layer split is a **prefill-CP-only optimization** for DSA..."

更硬的是 `srt/server_args.py:4427` 在非 prefill 模式下会直接抛异常：

```python
raise ValueError(
    "--enable-dsa-cache-layer-split is only supported on PD prefill workers. "
    "Non-PD workers also run decode and require ordinary local decode cache semantics."
)
```

翻译：**「非 PD 的 worker 还要跑 decode，需要的是普通的、本地的 decode cache 语义」**——作者自己就告诉你，layersplit 和 decode 的语义不兼容。

而"transient slot 广播"那段（`srt/layers/utils/cp_utils.py:540` 的 `cp_layersplit_should_broadcast_prefix`）被一个 gate 卡死：

```python
return (
    get_global_server_args().enable_dsa_prefill_cp_layersplit
    and forward_batch.forward_mode.is_extend_without_speculative()  # 只有 prefill(extend) 才广播
)
```

`is_extend` = prefill 阶段。**decode 阶段这个广播永远不触发**，所以 decode 去读非自己拥有的层时，读到的是没被填过的槽位 → 乱码。这印证了：layersplit 从设计上就是 prefill 的，不是"decode 端没实现"，是"decode 端本就不该有它"。

### 2. prefill-CP 和 decode-CP 是两套独立的 flag

源码里有两个东西，这是最关键的盲点：

| | prefill CP | decode CP（DCP）|
|---|---|---|
| flag | `--enable-prefill-cp` | `--decode-context-parallel-size` / `--dcp-size` |
| 内部字段 | `attn_cp_size` | `dcp_size` |
| 默认 | 关 | **1（关）** |

sglang 其实**已经实现了 decode 端的 CP**（`srt/layers/dcp/` 整个模块 + dsa_backend 里的 "DCP decode path"）。所以 decode-CP 不是"要从零写"。

### 3. 但 DCP 在 B200 上没验证过

`srt/server_args.py:3153` 的 `_handle_dcp_validation` 注释：

> "Decode context parallel (DCP) is currently implemented and validated only on AMD HIP/ROCm... DCP is validated on both HIP and CUDA (**B300 SM103**)."

**B200（SM100）不在验证范围内。**

---

## 要不要实现代码，让 prefill-CP 能开？

分两个开关分别答。

### layersplit：不要，也没法"实现"

它不是"缺代码"，是"语义上只属于 prefill"。源码直接告诉你 decode 需要 ordinary local cache。想在 standalone 里开它，等于要重写 decode 的 KV cache 语义去迁就一个 prefill 优化——方向就错了。**结论：任何情况下 standalone 都关掉它。**

### prefill-CP：技术上可行，但投入产出不划算

- decode 端的 CP（DCP）**代码已经存在**，不是从零写。真正缺的是**在 B200(SM100) 上把 DCP 验证/适配好**。
- 而 prefill-CP 的收益，只在**「单条超长 prompt + 低并发」**这个场景才明显（长 prompt 的 O(seq²) attention 被 8 段切分）。短 prompt、高并发下，prefill-CP 的通信开销甚至会倒贴。
- 最关键的是：**PD 分离已经解决了这个问题**。PD 架构里，prefill 机开 `--enable-prefill-cp` 拿满收益，decode 机不开 CP 跑干净 decode——这就是这套参数**被设计出来的用法**。standalone 硬要单机同时吃 prefill-CP，等于自己去适配一个"PD 已经解决好了"的问题。

**建议：不为 B200 standalone 去实现/适配 DCP。** 如果确实有"长上下文 + 单机"的硬需求，两条路都比改代码便宜：

1. 直接用 **PD 分离**（prefill 机带 CP + decode 机不带）；
2. 或者接受 standalone 不开 CP，靠 `max-total-tokens` + hicache 把长上下文 KV 撑住。

---

## 结论：单机部署怎么处理

PD 分离（prefill 机 + decode 机分开）里，这两个 flag 是 prefill 机的合理优化，**保留**；但要跑**单机一体**服务（prefill+decode 同进程），三个开关必须全去掉：

```bash
# ❌ 单机里不要带这几条
--enable-prefill-cp --cp-strategy interleave
--enable-dsa-prefill-cp-layersplit
```

去掉之后，模型输出才恢复正常（`1+1=` → `2`）。

如果你在单机里发现"模型输出突然变成只有重复或乱码"，第一步先查启动参数里有没有这两条——**它们比什么采样参数、fp8 scale 都更可能是元凶**。

---

## 参考

- 本文结论来自 sglang PD 容器化部署（GLM-5.2 FP8 / B200）的 standalone 排障实录，源码证据来自 sglang 源码树（`python/sglang/`）。
- 相关但不同的主题：vLLM 的 decode-CP（DCP）是另一个维度的事，见本站《DCP 实测：消除张量并行下的 KV Cache 重复存储》。