# 在 H200 上服务 GLM-5.2 智能体负载:llm-d 的工程实践与成本测算

> **转载说明**:本文核心内容编译自 llm-d 官方博客《[Serving GLM-5.2 for Agentic Workloads on llm-d](https://llm-d.ai/blog/serving-glm-5-2-agentic-workloads-on-llm-d)》(2026 年),结合 SemiAnalysis 发布的 219 个生产环境 Claude Code 会话追踪语料库。所引用的实验拓扑、吞吐数据、延迟数据与租赁成本测算均来自原文。

---

## 引言:为什么需要重新思考 LLM 服务架构

智能体(agentic)工作负载和传统的聊天补全(chat completion)在结构上有根本性差异。编码智能体的一次轮次以"读"为主:输入端是仓库量级的上下文,输出端是一次工具调用或一段较短的 diff;下一轮再把整份上下文原样喂回去,只是稍微长一点。

SemiAnalysis 的 [Weka 追踪语料库](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-051926)给出了这背后清晰的统计画像:219 个生产会话、78,099 次模型请求。

| 主智能体指标 | p50 | p90 | p99 |
| --- | --- | --- | --- |
| 单次请求输入 token | 195K | 521K | 849K |
| 单次响应输出 token | 317 | 1,705 | 7,551 |
| 相对上一轮的输入长度增长 | 632 | 3,735 | 148K |
| 轮次之间的墙钟间隔 | 2 s | 24 s | 11.4 min |
| 单会话轮次 | 107 | 354 | 992 |

从中可以提炼出三个关键事实:

1. **绝大部分 GPU 工作发生在第一个输出 token 之前** —— 中位请求要处理 195K 输入 token、却只产生 317 输出 token。输入占全部被服务 token 的 98.6%。
2. **几乎所有内容之前都已读过** —— 把每次请求与最近的前序轮次匹配,96% 复用至少 90% 的输入,中位复用率高达 99.6%。
3. **会话会暂停,然后突发** —— 中位会话持续 1 小时,但调用间隔从 2 秒到 11 分钟不等;78,099 次请求中 40,367 次以"组"形式到达,中位 7 次/组、p90 51 次/组。

这意味着传统的"把模型装在 N 卡上"的部署方式,既无法装下这么长的 KV 缓存,也无法在请求突发的瞬间保持低延迟。

## 架构选择:六项 llm-d 能力的组合

llm-d 的 agentic-serving 路径是 **"从工作负载出发,把所需能力组合成部署"**。针对 GLM-5.2 在 H200 上的智能体负载,组合了六项能力:

| 能力 | 解决的问题 |
| --- | --- |
| **MLA**(多头潜在注意力) | 将每 token KV 状态压缩到约 44 KB(对照 64 KV 头 MHA 可省 ~57 倍) |
| **宽专家并行(Wide-EP)** | 用数据并行注意力 + 专家并行 MoE 替代 TP=8,让 KV 不再被复制 8 份 |
| **前缀感知路由** | 把每个轮次送回已持有其 KV 的副本,避免重算 |
| **CPU 卸载** | 当 HBM 装满时,把淘汰的 KV 保留在 DRAM 中,恢复比重建更快 |
| **Prefill/Decode 分池** | 让"长读取"与"token 生成"独立扩缩容 |
| **MTP(多 token 预测)** | 每次 decode 步额外起草 3 个 token,提升生成吞吐 |

### 关键架构决策:MDA 与 Wide-EP

GLM-5.2 的 MLA 只存储一份压缩后的潜在向量,而不是分开的 KV 头,因此张量并行无法对其进行分片 —— TP=8 会保留 8 份 KV 缓存副本;同时跨节点 TP 的 all-reduce 开销昂贵,每层都要执行两次"列并行 + 行并行 + all-reduce"。

llm-d 的 Wide-EP 路径对此的解法是: **数据并行注意力 + 专家并行 MoE**。在 H200 上,实测显示这能换来数量级的 KV 容量提升:

| 拓扑 | 单节点 KV 容量(逻辑 token) |
| --- | --- |
| 聚合 TP=8 | 598K |
| DEP8 | 1.77M |
| DEP16(节点份额) | 约 5.2M |

聚合 TP=8 布局以 3×–9× 的会话容量为代价,换取了单节点调度单元与避免 NVSHMEM/all-to-all 流量 —— 对于 KV 容量是主要约束的智能体负载,这不是划算的取舍。

### 拓扑记号

本文使用 `xPyD` 表示 `x` 个 prefill 实例 + `y` 个 decode 实例;`DEP16` 表示 TP=1、DP=16、专家并行 16 GPU。`3P1D` 即"3 个 prefill 实例 + 1 个 decode 实例,合计 64 GPU"。

## 深度解析:MLA 与 Wide-EP 是怎么做的

前面提到 MLA 和 Wide-EP 是整套部署的根基 —— 没有它们,仓库级上下文根本装不下,单 GPU 上也没有足够容量去支撑高并发。下面把这两件事拆开讲清楚。

### 一、MLA —— 为什么 KV 缓存能压到这么小

#### 标准注意力的 KV 长什么样

在标准多头注意力(MHA)里,每个 token 在每一层生成两个张量:**K**(key)和 **V**(value),形状都是 `[num_heads, head_dim]`。它们的"尺寸"由两个超参决定 —— 头数(如 64)和每头维度(如 128)。如果 KV 用 FP16 存储,每 token 每层的体积就是 `2 × num_heads × head_dim × 2 bytes`。

GLM-5.2 是 78 层、64 KV 头的模型,若用标准 MHA,单 token 单层约 `64 × 128 × 2 × 2 = 32 KB`(FP16),全 78 层叠加约 2.5 MB。KV 还没算权重,光这部分就足以吞噬宝贵的 HBM。

GQA(分组查询注意力)用更少的 KV 头共享 K/V,可把 KV 降到约 1/8,但仍然以"每个 KV 头一份完整 K/V 张量"的形式存储。

#### MLA 的做法:压到一个潜在向量

MLA(Multi-head Latent Attention,DeepSeek-V2 提出)的核心想法是 —— **不直接存 K 和 V,而是存一个压缩后的"潜在向量" `c_t`**,推理时再通过低秩投影把它还原成 K 和 V。

GLM-5.2 的 MLA 每 token 每层只存:

| 组件 | 大小(FP8 元素) | 说明 |
| --- | --- | --- |
| 潜在向量 `c_t` | 512 | 压缩后的低维表征 |
| 共享 RoPE | 64 | 解耦的旋转位置编码,所有 head 共享 |
| **合计** | **576** | 每层每 token 576 字节 |

78 层叠加后,约 **44 KB/token**。下面按比例对比三种方案的 KV 占用:

| 方案 | 单 token 78 层累积 KV | 相对 MLA |
| --- | --- | --- |
| MHA(64 KV 头 × 256 维) | ~2.5 MB | 约 57× |
| GQA(8 KV 头 × 256 维) | ~313 KB | 约 7× |
| **MLA(512 潜在 + 64 RoPE)** | **~44 KB** | **1×** |

存储语义可以写成伪代码:

```python
# MLA:每次只缓存潜在向量 + 共享 RoPE
KV_cache[token][layer] = {
    c_t:        Tensor[512],   # 压缩后的潜在向量
    rope_share: Tensor[64],    # 所有 head 共享的 RoPE
}
# 推理时:W_K · c_t + rope_share → K,W_V · c_t → V
```

### 二、MLA 与张量并行的不兼容

MLA 让 KV 变小,但也让它**不能再被张量并行(TP)分片**。这是整篇文章的因果起点。

#### 为什么 TP 分片不了 MLA 的 KV

在标准 MHA/GQA 里,KV 缓存沿"头"维度是有结构的。比如 TP=8 时,可以按头把 64 个 KV 头分给 8 个 rank,每个 rank 只需缓存自己负责的那几个头的 K、V(约占总量的 1/8)。

MLA 的 KV 缓存是一个**完整的潜在向量 `c_t`**,以及**所有 head 共享的 RoPE 项**。这两个张量**都没有"沿 head 分片"的维度** —— 不管你分多少 rank,每个 rank 都需要 `c_t` 的全部 512 个元素和 RoPE 的全部 64 个元素。

> **关键后果**:TP=8 不再*分摊* KV 缓存,而是**复制 8 份**。跨节点 TP 的 all-reduce 开销也更贵 —— 每层要做两次"列并行 → 行并行 → all-reduce"(因为后面还接 MoE)。

实测上,聚合 TP=8 单节点 8 份副本,单节点总物理 KV 约 200 GiB,其中绝大部分是冗余副本。

### 三、Wide-EP —— 换一个分法

既然 TP 行不通,llm-d 选择的方案是 **Wide-EP(Wide Expert Parallelism)**。它不是单一技术,而是两种并行的组合:

- **注意力部分**:数据并行(DP attention)
- **MoE 部分**:专家并行(EP MoE)

#### 3.1 数据并行注意力:不复制 KV

在数据并行注意力下:

- 每个 GPU 都持有**完整的**注意力权重(W_Q、W_K、V 投影、MLA 的低秩矩阵等)
- 不同 GPU 处理**不同的 token batch**(类似经典的数据并行)
- 每个 GPU **只缓存自己处理过的 token 的 KV**

也就是说:KV 缓存在每个 rank 上是**本地且独立**的,完全不需要在 rank 之间复制或分片。

实测单 rank KV 容量:

| 配置 | 单 rank KV 容量 |
| --- | --- |
| 聚合 TP=8(单节点) | 598K token(每 rank 25 GiB 副本) |
| DEP8 prefill rank | 182K token |
| **DEP16 prefill rank** | **958K – 990K token** |
| DEP16 decode rank | 1,078K token |

从单节点总 KV 看:

| 布局 | 单节点总 KV |
| --- | --- |
| 聚合 TP=8 | 598K token(约 200 GiB 物理 KV,绝大部分是冗余副本) |
| DEP8 | 1.77M token |
| **DEP16(节点份额)** | **约 5.2M token** |

#### 3.2 专家并行 MoE:把专家分散到不同 GPU

GLM-5.2 是 744B 参数、激活 39B 的 MoE 模型。如果按 TP=8 把权重分给 8 个 rank,每个 rank 要承担约 **89.5 GiB** 的专家权重 —— 直接把留给 KV 的显存吃掉了。

EP MoE 的做法:

1. 不同 rank **持有不同的专家子集**(例如 256 个专家分给 16 个 rank,每个 rank 持 16 个)
2. 推理时,token 通过门控网络选专家,然后通过 **all-to-all** 通信把 token 送到持有相应专家的 rank 上做计算
3. 计算完成后,再通过 all-to-all 把结果送回原 rank

这样每个 rank 只需持有自己那份专家权重。在 Wide-EP 的实测布局中,**每个 GPU 只承担约 65 GiB 权重**(对比 TP=8 的 89.5 GiB),省下约 25 GiB 用于 KV。

GLM-5.2 的部署用 **DeepEP** 处理 token 的 all-to-all 派发,用 `deep_gemm` 跑专家计算。通信走 NVLink(节点内)+ InfiniBand(节点间)。

### 四、"Wide" 的含义

历史上,EP 通常被限制在**单节点内**(节点内 NVLink 带宽充裕、延迟低)。"Wide" 则是把 EP 扩展到**跨多个节点**,借助 InfiniBand 通信。

实测中,宽度直接决定单 rank KV 容量上限:

| 布局 | 每 rank KV | 备注 |
| --- | --- | --- |
| DEP8(单节点,8 GPU) | ~182K token | 更早触及 HBM 压力点 |
| **DEP16(2 节点 × 8 GPU)** | **~990K token** | DEP8 的 5.4 倍 |

宽度更大,意味着每 rank 上需要持有的专家权重更少(权重被分得更散),**省下来的显存可以全部留给 KV 缓存**。这是为什么在限长输入扫描中,DEP8 比 DEP16 更早触发 CPU 卸载。

### 五、MLA + Wide-EP 如何协同

把上面两块拼起来,GLM-5.2 在 H200 上的 prefill 实例每层工作流大致是:

```python
# 单层 prefill 工作流(在 DEP16 实例上,每层重复 78 次)
def layer_forward(tokens, position_ids):
    # 1) MLA 注意力 —— 数据并行,KV 写入本地 HBM
    attn_out = mla_attention_dp(
        tokens,
        local_kv_cache   # 本 rank 的 KV,不跨 rank 复制
    )

    # 2) MoE 门控 —— 为每个 token 选 top-k 专家
    routing = moe_gate(attn_out)    # [token → expert_id]

    # 3) DeepEP all-to-all —— 把 token 派发到持有对应专家的 rank
    dispatched = deep_ep.dispatch(attn_out, routing)

    # 4) deep_gemm —— 在目标 rank 上跑专家计算
    expert_out = deep_gemm(dispatched)

    # 5) DeepEP all-to-all —— 把结果送回原 rank
    output = deep_ep.combine(expert_out)

    return output   # 进入下一层
```

整个过程**没有任何跨 rank 的 attention all-reduce**(因为是 DP,不是 TP),KV 始终在本地;**每层只有一次 MoE 的 all-to-all**。

与 TP=8 的逐项对比:

| 操作 / 资源 | 聚合 TP=8 | Wide-EP |
| --- | --- | --- |
| 注意力 all-reduce | 每层 2 次(列并行 + 行并行) | **0 次** |
| 跨 rank 通信 | InfiniBand all-reduce × 2/层 | DeepEP all-to-all × 1/层(MoE) |
| KV 复制份数 | 8 份 | **1 份** |
| 单 rank 权重 | 89.5 GiB | **65.1 GiB** |
| 单 rank 可用 KV 容量 | 少(被权重挤压) | **多**(权重更少 + KV 不复制) |

代价是 MoE 的 all-to-all。但 DeepEP 对 NVLink + InfiniBand 做了专门优化,且 all-to-all 的数据量与 **token 数**线性相关,与 **KV 总量**无关 —— 恰好适合长上下文、短 batch 的智能体负载。

### 六、因果链:一句话总结

1. **MLA** 把 KV 缓存压到 44 KB/token,但也让 KV 不再能沿 head 维度分片 —— **TP 不再省 KV 显存**。
2. **Wide-EP** 用 DP 注意力(让 KV 本地独立、不复制)+ EP MoE(把专家权重分散到各 rank),既省了权重显存、又避开了 TP 的 all-reduce 开销。
3. **二者结合** 让每个 GPU 都能在 HBM 里塞下大量 KV 缓存(DEP16 rank 约 1M token),为仓库级上下文 + 高并发的智能体服务提供了物理基础。

## 容量与并发:输入侧的瓶颈

### 64 GPU `3P1D` 参考扫描

| 并发 | 请求数 | TTFT p50 / p90 | ITL p50 / p90 | 端到端 p50 | 输出吞吐 |
| --- | --- | --- | --- | --- | --- |
| 64 | 3,543 | 1.71 s / 6.36 s | 15.5 ms / 18.9 ms | 9.6 s | 3,679 tok/s(57/GPU) |
| 128 | 6,242 | 2.18 s / 15.82 s | 19.1 ms / 23.1 ms | 14.0 s | 6,281 tok/s(98/GPU) |

- **并发翻倍,输出提升 71%** (3,679 → 6,281 tok/s),中位输出 token 间隔从 15.5 ms 仅上升到 19.1 ms —— 单个活跃流的流式速度基本不受影响。
- **输入吞吐从 86K 升至 431K tok/s**,增长速度在比例上快于 decode,意味着 prefill 容量越来越成为首 token 时间的主导因素。

> 关键洞察:这个工作负载上,**集群主要是在为 prefill 容量买单**,而不是 decode 容量。

### Prefill/Decode 容量分配的两难

| 指标 | `3P1D`(3 prefill + 1 decode) | `2P2D`(2 prefill + 2 decode) |
| --- | --- | --- |
| 输出吞吐 | 6,281 tok/s | 6,781 tok/s |
| 端到端单用户 | 36.6 tok/s/user | 40.9 tok/s/user |
| TTFT 中位 | ~2.1 s | ~2.7 s |
| 输出 token 间隔中位 | 19.1 ms | 15.5 ms |
| CPU 恢复流量 | 较少 | 较多 |

- **新增一个 decoder**:输出 +21%、单轮完成时间 -18%、流式间隔从 19.1 → 15.5 ms。
- **新增一个 prefill**:中位 TTFT -21%、CPU 恢复流量 -94%(把活跃 KV 集合搬回 HBM)。

在同样 64 GPU 预算下,`2P2D` 偏向流式吞吐与整轮完成,`3P1D` 偏向首 token 时间与更低的恢复流量。

## 前缀缓存与 CPU 卸载

### 路由捕获的复用率

启用 MTP 与 CPU 卸载后,在 `DEP16` `1P1D` 与 `2P1D` 拓扑上,GPU 内存供给 90–92% 的可复用 prefill 前缀(可达上限 93.9–94.4%)。路由器对 prefill 与 decode 采用不同调度:prefill 侧同时考虑"前缀复用度"、"队列长度"、"缓存剩余空间";decode 侧则在 GPU/CPU 局部性与活跃请求数之间权衡。

在暂停对话实验中,带缓存前缀的轮次达到首 token 速度比无前缀的首轮快 **2.8 倍**(2.68 s vs 7.46 s,后者输入约 20K token)。

### CPU 卸载的临界点

24 GPU `1P1D` 上,固定输入分布(匹配度 99%)的对比:

- **并发 32**:输出 +13%,中位 TTFT -32%,中位单轮完成时间 -27%,中位输出 token 间隔不变。
- **并发 128**:基线输出停留在约 951 tok/s,CPU 卸载达到 2,776 tok/s。

CPU 卸载在 HBM 装满前几乎不起作用;一旦压力到来,它提供容量与延迟的同步收益。

### 恢复 vs 重算

在混合宽度 `2P1D` 拓扑、GPU KV 限制在每 rank 约 230K token 的受控实验下:

- **启用 CPU 卸载**:一次未命中恢复约 59 ms,并发 128 时中位 TTFT 8.3 s,吞吐保持完整。
- **关闭 CPU 卸载**:一次未命中需重算 ~45K token,客户端因 97 个请求排队而等待 59 s,**吞吐下降 78%,完成的轮次减少到 1/4.5**。

两组实验下输出 token 间隔均维持在 31–36 ms —— 区别只发生在 prefill 侧,decode 侧不受影响。

整个实验期间,引擎计数器记录到 12.7 TB KV(即 2.83 亿 token)以 26 ms/GB 的速率被恢复。CPU 缓存在服务 2.17 亿 prompt token 的同时恢复了 2.22 亿 token,比例 >100% —— 这暗示存在抖动,**比 LRU 更好的留存策略可以进一步减少数据搬运**。

## MTP:同样的 GPU,8 倍的并发

32 GPU `1P1D` `DEP16`、均开启 CPU 卸载,在维持"每用户至少 30 tok/s"的平均输出速率下限前提下:

| 配置 | 满足下限的最高并发 | 聚合输出 |
| --- | --- | --- |
| 仅 CPU 卸载 | c16 | 561 tok/s |
| CPU 卸载 + MTP | c128 | 5,647 tok/s |

也就是说,MTP 让同样的 GPU **至少服务 8 倍的并发、产出 10.1 倍的输出**。

每次 MTP 步平均产出 2.95 个 token,草稿接受率介于 64.0%–70.5%;按草稿位置看,接受率约为 80%、65%、52%。

| 并发 | TTFT p50(卸载 / MTP) | ITL p50(卸载 / MTP) | 输出吞吐(卸载 / MTP) | 端到端单用户(卸载 / MTP) |
| --- | --- | --- | --- | --- |
| 32 | 1.1 s / 1.3 s | 28.6 / 13.5 ms | 1,145 / 2,159 tok/s | 28.7 / 51.5 tok/s/user |
| 64 | 1.3 s / 1.6 s | 30.7 / 14.7 ms | 2,178 / 3,788 tok/s | 25.9 / 45.1 tok/s/user |
| 128 | 1.6 s / 2.4 s | 35.1 / 17.1 ms | 3,505 / 5,647 tok/s | 21.6 / 35.0 tok/s/user |

MTP 把中位输出 token 间隔降低约 52%,但**首 token 时间并未改善** —— MTP 加速的是 decode,不是 prefill。

## 成本测算:自建 GPU 租赁 vs 托管 API

按 H200 租赁价 $2.71/GPU·小时 计算,持续满载场景下:

| 拓扑 | 每百万输入 token | 每百万输出 token |
| --- | --- | --- |
| 64 GPU `3P1D` | $0.08 | $1.92 |
| 64 GPU `2P2D` | $0.05 | $3.55 |

对比公开 API 价格:

| 服务 | 输入 | 输出 |
| --- | --- | --- |
| 自建 `2P2D` | $0.05 | $3.55 |
| Z.ai GLM-5.2 | $1.40 | $4.40 |
| Anthropic Claude Opus 4.8 | $5.00 | $25.00 |

- 自建 `2P2D` 的输出成本约为 Z.ai GLM-5.2 的 81%、Opus 4.8 的 1/7。
- 自建 `2P2D` 的输入成本是 Z.ai 价的 3.5%、Z.ai 缓存输入价的 20%。

需要注意的是:这些数字**仅覆盖 GPU 租赁**,不含存储、出口带宽、工程、运维、税费;且假设集群 24/7 满载运行,利用率越低,等效成本反比上升。P/D 分池的成本拆分将每个池的成本分配给它处理的 token,改变拆分方式会改变两列数字,但不影响总集群成本。

## 经验法则:为智能体负载调参

把全文的发现凝练为几条工程经验:

1. **KV 容量是主要部署约束**。MLA + Wide-EP 是基础,没有它们任何长上下文服务都不可行。
2. **HBM 装满前不要急着加 CPU 卸载**;但一旦预见到峰值会超 HBM,CPU 卸载几乎是必选项。
3. **Prefill 与 Decode 必须分池**。它们的工作节奏、瓶颈、扩缩容参数都不同 —— 把它们绑在一起注定取次优解。
4. **MTP 是性价比最高的 decode 加速**。在 decode batch 受长上下文限制的场景下,它能 8× 并发。
5. **路由的复用率上限是"工作负载的可复用前缀占比"**。当上限达到 93.9% 时,即使把路由器做到极致也只到 91% —— 不要再为剩余几个百分点调路由。
6. **容量规划要看"瓶颈在哪里"**,而不是看"配置是否对称"。`3P1D` 与 `2P2D` 在同 64 GPU 下输出仅差 8%,但它们的延迟画像和恢复流量完全不同。

## 下一步

llm-d 团队计划:对比 P/D 与配对的聚合拓扑、在高并发下测试更宽的 decode 池、并验证 MTP 下的工具调用。Agentic Inference SIG 在 llm-d Slack 的 [#sig-agentic-inference](https://llm-d.slack.com/archives/C0ALHNZJCFJ) 频道公开推进这项工作,欢迎带上你的追踪数据参与。

完整的部署配置、可组合的 MTP/CPU 卸载/上下文长度组件,见 [GLM-5.2 H200 配置指南](https://github.com/llm-d/llm-d/blob/main/guides/agentic-serving/glm-5-2-h200.md)。

---

**参考与延伸阅读**

- 原文:[Serving GLM-5.2 for Agentic Workloads on llm-d](https://llm-d.ai/blog/serving-glm-5-2-agentic-workloads-on-llm-d)
- 部署指南:[GLM-5.2 H200 configuration](https://github.com/llm-d/llm-d/blob/main/guides/agentic-serving/glm-5-2-h200.md)
- 工作负载语料:[cc-traces-weka-with-subagents-051926](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-051926)(219 会话) / [cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)(393 追踪)
- 同模式 cookbook:[Nemotron-3-Ultra-550B on H200](https://github.com/llm-d/llm-d/blob/main/guides/agentic-serving/nemotron-3-ultra-550b-h200.md)、[Qwen3-Coder-480B on TPU v7](https://github.com/llm-d/llm-d/blob/main/guides/agentic-serving/qwen3-coder-480b-tpu.md)

---

### 转载说明

- 本文以 llm-d 官方博客 2026 年发布的原文为基础,在保留核心技术结论与数据的前提下进行了结构重组与精简;未引入原文以外的论断或数据。
- 文中所有数字、拓扑标签(`3P1D` / `DEP16` 等)、术语(MLA / MTP / Wide-EP / ITL / TTFT / NIXL / DeepEP 等)均与原文保持一致。
- 转载时建议保留本文顶部的转载说明与底部的参考链接,便于读者溯源。