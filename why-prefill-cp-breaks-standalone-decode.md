# 为什么 prefill 的 CP 优化，不能带到单机 decode？

## 一句话总结

sglang 里 `--enable-prefill-cp`（配合 `--cp-strategy interleave`）和 `--enable-dsa-prefill-cp-layersplit` 是两个 **prefill 专属**的并行优化。在 PD 分离架构里它们各安其位；可一旦把服务以**单机一体（prefill+decode 同进程）**的方式起，这两个开关就会把 decode 阶段搞坏——一个让输出退化成"一句话"式的重复，另一个让输出直接变成随机 token。原理根子在**这两个优化从语义上就跟 decode 的自回归特性冲突**，只能整个去掉。

---

## 模型结构：先看清要切的东西

讲切分之前，先看 GLM-5.2 本身。它是一个用 **DSA（Dynamic Sparse Attention，动态稀疏注意力）的 MoE 模型**：

```
GLM-5.2（78 层，hidden=6144，64 个注意力头）

输入 token 序列：  t0  t1  t2  ...  tN
                      │ 逐层往下传
        ┌─────────────────────────────────┐
        │ 第 i 层（i = 0 .. 77，共 78 层）   │
        │                                 │
        │   ① QKV 投影（MLA 多头潜在注意力）  │
        │   ② 索引器 indexer（32 头 × 128 维）│
        │        —— 动态选出 top-2048 个 token│
        │   ③ 稀疏注意力（只 attend 这 2048） │
        │   ④ MLP（MoE 专家混合）            │
        └─────────────────────────────────┘
                      │
                      ▼
                  输出 logits
```

**DSA 的关键**：它不做传统全注意力（每个 token 看所有 token，O(N²)），而是用**索引器（indexer）**这个小网络，动态给每个 token 挑最相关的 **`index_topk=2048` 个** token，attention 只在这 2048 个里算。所以复杂度是 O(N × 2048)，远低于全注意力。索引器每 4 层共享一次（`index_topk_freq=4`）。

下面两个 flag 做的都是"并行化 / 存放"这一套东西。

---

## 三个"切分维度"：切的是完全不同的东西

| 切法 | 切的是什么 | 8 张卡各拿 | 什么时候有用 |
|---|---|---|---|
| **TP（张量并行）** | 权重（`Wq/Wk/Wv`） | 1/8 权重，算**整条序列** | 权重太大单卡放不下 |
| **CP（上下文并行）** | **序列长度**（token） | 1/8 token，算**全部权重** | 序列太长算不动 |
| **layersplit（层切分）** | **78 层** | own ~10 层，其它层靠临时广播 | prefill 层间流水线 / 每卡只存 1/8 层 KV |

关键：**TP 和 CP 是正交的**。TP8 让每张卡算 1/8 权重；CP8 再让每张卡只算 1/8 序列。两者叠加，prefill 的"长 prompt × 全权重"就被切成了"1/8 序列 × 1/8 权重"。

---

## 为什么 prefill 能吃 CP，decode 不能

prefill 和 decode 是两种完全不同的计算，这是整套理解的根：

```
prefill（预填充）：一次性把整段 prompt 喂进去
   [t0 t1 t2 ... tN] ──────▶ 所有 token 并行前向，边算边写 KV
   特点：token 一次性并行 → 可切分、可加速

decode（解码）：自回归生成
   每 1 步只有 1 个新 token ──attend──▶ 全部历史 KV ──▶ 再出下 1 个
   特点：每步只有 1 个 token、要读全量 KV、逐步串行
```

- **prefill**：一次处理整段 prompt（比如 8 万 token），切成 8 段、每卡算 1 万段 → **接近 8 倍加速**。这是 CP 的真金收益。
- **decode**：每步只出 1 个 token。"1 个 token" 没法再切 8 份并行；而且它要读的是**全部历史 KV**，切序列反而添乱。

**这就是"prefill 专属"的根。** 下面两个 flag 违反这一条的两种方式各不相同。

---

## 两个 flag 的原理

### `--enable-prefill-cp --cp-strategy interleave`：沿序列维切

打开它，sglang 把 `attn_cp_size` 设成等于 TP（TP8 → CP8），prefill 把长序列切 8 段：

```
整段 prompt:   [════════ t0 … tN ════════]
                    切成 8 段
   ┌─────┐  ┌─────┐  ┌─────┐      ┌─────┐
   卡 0      卡 1      卡 2   …     卡 7
每张算自家 1/8 段的 DSA 稀疏注意力 + 索引器，再通信归并
```

对 prefill 这是对的。**但 `attn_cp_size` 是全局量，不只 prefill**——decode 也带着"序列已切 8 段"的布局去跑，于是出问题。

### `--enable-dsa-prefill-cp-layersplit`：沿层维切 KV

这个切的是 **KV cache 的存放**，按**层**切。启动日志直接写明：

```
CP layer-split rank 0/8: owns layers [0, 10) (10 layers).
  Allocates 1 transient slot to hold each non-owned layer's prefix KV
  broadcast from its owner rank.
```

含义：78 层拆给 8 张卡，**每张卡只"拥有"约 10 层的 KV**；对其它 7/8 的层，它只留一个 **transient slot（临时槽位）**，等对方广播时临时占用。因为 prefill 前向是从第 0 层到第 77 层顺序往下跑，这套广播能"接一次、用一次、再传下一个"：

```
prefill 顺序前向（transient 槽位有效）：
  卡0 算 L0-9 的 KV ──广播──▶ 卡1 的 transient slot ──▶ 卡1 接着算 L10-19
  卡1 算 L10-19    ──广播──▶ 卡2 ──▶ 算了 L20-29 …
  …（算一次、广播一次、用一次，流水线往后传）
```

这套机制是给 prefill 的**一次性顺序前向**设计的。decode 撞上它就坏（下面讲）。

---

## 实测：单机下它们坏了什么

隔离测试（`temperature=0` 贪心，排除采样噪声）：

| 配置 | `1+1=` 的输出 | 现象 |
|---|---|---|
| 无 CP | `2` | ✅ 正常 |
| 只开 `--enable-prefill-cp --cp-strategy interleave` | `1111+1++++` | decode 退化重复 |
| 两者都开 | 随机、多语言混杂 | decode 读到错 KV |

> ⚠️ 诚实标注："只开 prefill-CP" 是本次直接隔离实测；"两者都开"的乱码来自最初完整参数集观测。

---

## 为什么坏：两类不同的问题

### `--enable-prefill-cp`：decode 端「没接上」

decode 每步只有 1 个 token，要 attend 全部历史；而"序列已切 8 段"后，这 1 个 token 要去对 8 段分散 KV 各自算再归并。prefill 用"所有 token 一并算 + 通信归并"能行，但 decode 的"单 query + 已切段"这个组合，需要专门的 **decode-CP** 实现来兜——**而 DSA 的 decode 端没接上** → 注意力分值算错 → 输出一层重复（`一句话一句话…`）。

### `--enable-dsa-prefill-cp-layersplit`：语义上容不下 decode

layersplit 的 transient 槽位只在 prefill 顺序前向时有意义。**decode 是逐 token 反复读全 78 层 KV**，而每卡只存了 ~10 层、transient 槽位只有一层的容量，decode 想读的"其它 ~68 层"根本不在卡上 → 读到没被填充过的错位数据 → logits 全乱 → 随机 token。

```
decode 流程（layersplit 下坏）：
  1 个新 token 的 forward 要过全 78 层：
    卡0 只存 L0-9，读到 L10 时：
      广播 gate = is_extend=False → 不广播 → 读到空 transient slot → 错
```

**一句话区分：**

- `--enable-prefill-cp` 坏在 **decode 端能力（decode-CP）没实现 / 不兼容**；
- `--enable-dsa-prefill-cp-layersplit` 坏在 **从设计上就不该对 decode 开**。

---

## 源码证据：坐实上面两个判断

（sglang 源码树 `python/sglang/`，带文件:行号）

**1. layersplit 就是 prefill-only**

`srt/layers/cp/utils.py:52` docstring：

> "Layer split is a **prefill-CP-only optimization** for DSA..."

`srt/server_args.py:4427` 非 prefill 模式直接抛异常：

```python
raise ValueError(
    "--enable-dsa-cache-layer-split is only supported on PD prefill workers. "
    "Non-PD workers also run decode and require ordinary local decode cache semantics."
)
```

即作者白纸黑字：**「非 PD worker 还要跑 decode，需要普通本地 decode cache 语义」**。

更硬的是 `srt/layers/utils/cp_utils.py:540` 的广播 gate：

```python
return (
    get_global_server_args().enable_dsa_prefill_cp_layersplit
    and forward_batch.forward_mode.is_extend_without_speculative()  # 只有 prefill(extend) 才广播
)
```

`is_extend` = prefill 阶段。**decode 时这个广播永不触发**——印证"读到空槽位"。

**2. prefill-CP 和 decode-CP 是两套独立 flag**

| | prefill-CP | decode-CP（DCP）|
|---|---|---|
| flag | `--enable-prefill-cp` | `--decode-context-parallel-size` / `--dcp-size` |
| 内部字段 | `attn_cp_size` | `dcp_size` |
| 默认 | 关 | **1（关）** |

sglang 其实**已经实现了 decode 端 CP**（`srt/layers/dcp/` 整个模块 + dsa_backend 里的 "DCP decode path"）。

**3. DCP 在 B200 上没验证过**

`srt/server_args.py:3153` 注释：

> "…DCP is validated on both HIP and CUDA (**B300 SM103**)."

**B200（SM100）不在验证范围内。**

---

## 要不要实现代码，让 prefill-CP 能开？

### layersplit：不要，也没法"实现"

它不是"缺代码"，是"语义上只属于 prefill"。源码明说 decode 需要 ordinary local cache。想在 standalone 开它，等于重写 decode 的 KV cache 语义去迁就一个 prefill 优化——方向就错。**结论：任何情况下 standalone 都关掉它。**

### prefill-CP：技术上可行，但投入产出不划算

- decode 端 CP（DCP）**代码已存在**，不是从零写。真正缺的是**在 B200(SM100) 上把 DCP 验证/适配好**。
- prefill-CP 的收益只在**「单条超长 prompt + 低并发」**才明显（长序列 prefill 被切 8 段并行）；短 prompt 高并发下通信开销甚至倒贴。
- **PD 分离已经解决了这个问题**：prefill 机开 CP 拿满收益，decode 机不开 CP 干净跑——这就是这套参数的设计用法。standalone 硬要单机同时吃 prefill 的 CP，等于重新造一个 PD 已经解决的轮子。

**建议：不为 B200 standalone 去适配 DCP。** 真有"长上下文 + 单机"硬需求，两条更便宜的路：

1. 直接用 **PD 分离**（prefill 机带 CP + decode 机不带）；
2. 或 standalone 不开 CP，靠 `max-total-tokens` + hicache 撑住长上下文。

---

## 结论

PD 分离（prefill 机 + decode 机分开）里这两个 flag 是 prefill 机的合理优化，**保留**；但**单机一体**服务这三个开关必须全去掉：

```bash
# ❌ 单机里不要带
--enable-prefill-cp --cp-strategy interleave
--enable-dsa-prefill-cp-layersplit
```

去掉后输出恢复正常。哪天单机发现"输出变成重复 / 乱码"，先查启动参数里有没有这两条——它们比采样参数、fp8 scale 都更可能是元凶。

---

## 参考

- sglang PD 部署（GLM-5.2 FP8 / B200）的 standalone 排障实录；源码证据来自 sglang 源码树 `python/sglang/`。
- 相关但不同：vLLM 的 decode-CP（DCP）见本站《DCP 实测：消除张量并行下的 KV Cache 重复存储》。