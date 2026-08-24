# 为什么 prefill 的 CP 优化，不能带到单机 decode？

## 一句话总结

sglang 里 `--enable-prefill-cp`（配合 `--cp-strategy interleave`）和 `--enable-dsa-prefill-cp-layersplit` 是两个 **prefill 专属**的并行优化。在 PD 分离架构里它们各自安好；但一旦把模型以**单机一体（prefill+decode 同进程）**的方式起服务，这两个开关就会把 decode 阶段搞坏——一个让输出退化成"一句话一句话"式的重复，另一个让输出直接变成随机 token。问题不在"哪个值没调好"，而在**这两个优化从语义上就跟 decode 的自回归特性冲突**，只能整个关掉。

---

## 背景：prefill 和 decode 是两种完全不同的计算

要理解为什么"prefill 的优化"不能套到 decode 上，得先看清两者根本不是一回事。

| | prefill（预填充） | decode（解码） |
|---|---|---|
| 输入 | 整段 prompt（几十到几万 token） | 上一个生成的 token |
| 计算方式 | 所有 token **一次性并行**前向 | 每步只算 **1 个 token** |
| 读 KV cache | 边算边写 KV | 每步**读全部历史 KV** |
| 瓶颈性质 | 计算密集，可并行 | 访存/带宽密集，逐 token 串行 |

关键在最后一行：prefill 是一次性、可并行的；decode 是一步一步、严重依赖"完整 KV"的自回归循环。**凡是靠"把东西切开来并行"的优化，天然适合 prefill，天然不适合 decode。**

---

## 两个 flag 到底做了什么

### `--enable-prefill-cp --cp-strategy interleave`：沿序列维切分

CP（Context Parallelism，上下文并行）是把 **序列长度维度**切到多个 rank 上并行算，和 TP（张量并行，切权重）是正交的。

`--enable-prefill-cp` 打开后，sglang 会把 `attn_cp_size` 设成等于 TP 的大小（比如 TP8 → CP8），于是 prefill 阶段把一条长 prompt 切成 8 段，8 个 rank 各算自己的那 1/8 序列，再通过通信归并。这对长 prompt 的 prefill 加速是合理的。

但这里藏着一个关键事实：**这个 `attn_cp_size` 是全局生效的，不只是 prefill**。decode 阶段一旦也带着 CP=8 的序列切分布局去跑，就会出问题（见下一节为什么要）。

### `--enable-dsa-prefill-cp-layersplit`：沿层维切分

`DSA` 是 GLM-5.2 那套动态稀疏注意力（Dynamic Sparse Attention）。`--enable-dsa-prefill-cp-layersplit` 做的是另一件事——它不切序列，而是**把 78 层 attention 层拆给 8 个 rank，每个 rank 只"拥有"约 10 层**。

这么做的直接证据在启动日志里：

```
CP layer-split rank 0/8: owns layers [0, 10) (10 layers).
  Allocates 1 transient slot to hold each non-owned layer's prefix KV
  broadcast from its owner rank.
```

每个 rank 只负责自己拥有的层；对于其它 rank 拥有的层，它只留一个 **transient slot（临时槽位）**去接对方一次广播过来的 KV 前缀。这套"只算自己层 + 临时广播"的机制，是给 prefill 那种**从 0 层到 77 层顺序往下灌**的转发路径设计的。

---

## 实测：它们在单机下是怎么坏的

把这三个开关在单机（不带 PD 分离）里做隔离测试，三种配置的结果差异非常鲜明（都用 `temperature=0` 贪心解码，排除采样噪声）：

| 配置 | `1+1=` 的输出 | 现象 |
|---|---|---|
| 无 CP | `2` | ✅ 正常 |
| 只开 `--enable-prefill-cp --cp-strategy interleave` | `1111+1++++` | decode 退化重复 |
| 两者都开 | 随机、多语言混杂 | decode 读到错 KV |

⚠️ 诚实标注：其中"只开 prefill-CP"一行是本次直接实测；"两者都开"的随机乱码来自最初的完整参数集观测。**两者的"机制"推断见下，是逻辑推断而非逐行源码确认**。但就算不看原因，只看这三行对照，结论已经硬。

---

## 为什么坏：两类不同的问题

这两个 flag 虽然都在单机下坏 decode，但坏的原因不一样，值得分开讲。

### prefill-CP：decode 端「没接上」

`--enable-prefill-cp` 本质是"prefill 有、decode 没有"的一类问题。

理由在于 decode 的形态：decode 每步 A 只有 1 个 query token，它要**attend 到全部已生成的 token**上。而把序列切 8 段的 CP 布局下，这 1 个 query 要对 8 段分散的 KV 各自算、再归并。这个流程 prefill 靠"所有 token 一并算 + 通信归并"是行得通的；但 decode 的"单 query 读全量 KV"跟"序列已经被切成 8 份"之间，需要一个专门的 decode-CP 实现来兜，**而在这个 sglang 版本里，DSA 模型的 decode 端并没有正确接上这套。**

结果就是 decode 每一步的注意力算出来是错的分值，模型行为被拖成"在原地打转"——典型表现就是输出一层层重复同一个短语（`一句话一句话一句话…`）。

### layersplit：语义上就只该给 prefill 用

`--enable-dsa-prefill-cp-layersplit` 的问题更深，属于"**逻辑上就不该对 decode 开**，而不是某段代码没写好"。

回到那个 transient slot 机制：它成立的前提是 **prefill 是一次性的顺序前向**——rank A 算完它拥有的第 0~9 层，把结果广播给拥有第 10~19 层的 rank B，流水线往下灌，广播完就完事。

但 decode 是自回归：每一个 token 都要**反复读全部 78 层的 KV**。layersplit 把层拆走、只留 transient 槽位后，decode 需要的那份"每层完整的、稳定在位的 KV"并不存在——它读到的是切来切去、跟"全量 KV"对不上的布局。于是 logits 直接乱掉，输出变成随机 token。

**一句话区分**：

- `enable-prefill-cp` 坏在 **decode 端这个能力没实现/不兼容**；
- `dsa-prefill-cp-layersplit` 坏在 **它从设计上就容不下 decode 的语义**。

---

## 结论：单机部署怎么处理

PD 分离（prefill 机 + decode 机分开）里，这两个 flag 是 prefill 机的合理优化，**保留**；但要跑**单机一体**服务（prefill+decode 同进程），三个开关必须全去掉：

```bash
# ❌ 单机里不要带这几条
--enable-prefill-cp --cp-strategy interleave
--enable-dsa-prefill-cp-layersplit
```

去掉之后，模型的输出才恢复正常（`1+1=` → `2`）。

如果你在单机里发现"模型输出突然变成只有重复或乱码"，第一步先查启动参数里有没有这两条——**它们比什么采样参数、fp8 scale 都更可能 是元凶**。

---

## 参考

- 本结论来自 sglang PD 容器化部署（GLM-5.2 FP8 / B200）的 standalone 排障实录。
- 相关但不同的主题：vLLM 的 decode-CP（DCP）是**另一个维度**的事，见本站《DCP 实测：消除张量并行下的 KV Cache 重复存储》。