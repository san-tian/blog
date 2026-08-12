# DCP: 动态上下文并行 —— 破解长上下文训练的输入动态性难题

## 核心问题

长上下文训练面临**输入动态性**问题：序列长度和 token 关系在样本间存在巨大变异。现有的 Context Parallelism (CP) 方法采用静态配置，为所有 batch 使用固定的并行度和数据分区方案，导致：

1. **序列长度差异**：训练数据中短序列远多于长序列（图 2 显示 LongAlign 和 LongDataCollection 数据集中，大部分序列长度 < 20K tokens，但也有 100K+ 的长序列）
2. **注意力模式多样**：不同任务使用不同的注意力掩码（causal、shared question、lambda-shaped、block-wise 等，图 6）

静态 CP 配置在处理这些动态输入时会产生：
- **冗余通信**：短序列也按最长序列的通信模式传输 KV cache
- **计算不均衡**：不同设备间的计算负载严重失衡（图 5c 显示长短序列混合时，某些设备空闲而其他设备过载）

以 8B GPT 模型在 EC2 p4d.24xlarge 集群（8 节点 × 8 GPUs）上训练为例，16-way CP 的通信开销占总迭代时间的 **36.7%**（图 1）。

## DCP 方案

DCP (Dynamic Context Parallelism) 通过**细粒度块级分区 + 动态映射**解决输入动态性问题。

### 核心设计

#### 1. 块级数据和计算分区

将注意力计算的四个可并行维度（batch、head、SeqQ、SeqKV）划分为细粒度的块：

- **数据块（Data Block）**：Q、K、V、O 张量的连续切片
  - Q、K、V 形状 [B, H, L, D]，按 batch 和 SeqLen 维度划分为 `H × B_b × B_q` 个 Q 块和 `B_b × B_kv` 个 KV 块
  - 块大小是超参数（如 Q/KV 块 256 tokens）

- **计算块（Computation Block）**：描述一对 Q 块和 KV 块之间的注意力计算
  - 一个 Q_i 块和 KV_j 块的计算产生输出块 O_i
  - 图 9 展示了不同注意力掩码下的计算块结构

#### 2. 超图分区优化映射

将块到设备的映射建模为**超图分区问题**（Hypergraph Partitioning）：

- **顶点**：数据块（Q、KV、O）和计算块
  - 顶点权重：数据块大小（bytes）+ 计算块 FLOPS
- **超边**：连接一个计算块与其输入/输出数据块
  - 超边权重：数据块大小（跨设备传输成本）

优化目标（图 12）：
```
minimize Σ s_e(λ_e - 1)  # 跨设备通信量
subject to:
  w(P_i) ≤ [1+ε, 1] ⊙ w(N)/R  # 负载均衡约束
```

使用层级超图分区：
1. **机器级分区**：优先最小化跨机器通信（带宽低、延迟高）
2. **设备级分区**：在每台机器内优化设备间通信

采用 multi-level KaHyPar 等高效启发式算法求解（NP-hard 问题）。

#### 3. 多路分治调度

计算和通信调度采用多路分治策略（Listing 3）：

1. 将每个设备的计算块按通信需求分为 T 个 division
2. 第 1 个 division 优先调度最少通信的块（先执行不依赖远程数据的计算）
3. 后续 division 按通信量递增调度
4. 使用 fused kernel（Triton）执行每个 division 内的所有操作（attention + reduction + copy + comm launch），隐藏通信延迟

### 系统架构（图 8）

- **Data Loader**：预取序列长度和掩码，触发 Planner
- **Planner**：
  - 为每个 batch 生成块（Block Generation）
  - 超图分区求解最优映射（Placement w/ Hypergraph）
  - 生成每设备执行计划（schedules with comm/comp blocks）
- **Executor**：
  - 块级缓冲区管理（local buffers，类型索引复用）
  - 执行 DCP instructions（blockwise attention、reduction、copy、comm、wait）
  - 使用 FlashAttention/Triton 实现高性能 kernel

### 用户接口（Listing 2）

```python
# 替换 attention 实现
core_attn_out = DCPAttn.apply(dcp_executor, q, k, v)

# 定义掩码函数（输入序列信息 → 掩码矩阵）
def mask_fn(seqlens, ...):
    return mask

# 训练脚本
dcp_dataloader = DCPDataloader(dataset, mask_fn)
dcp_executor = DCPExecutor(grouped_ranks)
for local_data, execution_plan in dcp_dataloader:
    loss = model(local_data, dcp_executor)
```

DCP 在训练迭代 i 时，Planner 可预取 i+κ 迭代的输入信息并行规划，与 GPU 训练流水线重叠，消除规划开销。

## 性能数据

### Attention 微基准测试（图略，论文 §7.1）

8× NVIDIA A100 (80GB)，NVSwitch 600GB/s，4x100 Gbps NIC：

| 场景 | Baseline | DCP 加速比 |
|---|---|---|
| Causal mask | RingFlashAttention | 1.19×–2.45× |
| Sparse mask | - | 2.15×–3.77× |

**关键优势**：
- RingFlashAttention 对所有序列使用固定 ring size（R 块），DCP 为每个序列动态选择最优分区
- ZigZag 只支持 2^R 块大小，无法适配任意长度；DCP 支持变长输入
- LoongTrain 不支持序列长度维度变长，需 padding 到 batch 内最长序列

### 端到端训练（表略，论文 §7.2）

| 框架 | Mask | 加速比 |
|---|---|---|
| TransformerEngine | Causal | 0.94×–1.16× |
| TransformerEngine | Sparse | 1.00×–1.46× |

端到端加速低于 attention 层加速（图 1 显示 attention 占 44.6%，CP 通信占 36.7%）：
- Causal mask 下，静态 CP 通信开销已较优化，DCP 提升有限
- Sparse mask 下，静态 CP 产生大量冗余通信（图 7b 显示 38/48 个 KV 块未被使用但仍传输），DCP 消除冗余获得显著提升

## 技术亮点

1. **统一表示**：数据块 + 计算块的抽象统一描述了所有 CP 配置（pure CP、DP+CP 混合、图 5），使动态规划成为可能

2. **负载感知分区**：计算块权重 = 2×(head_dim)×(Q_size)×(KV_size) FLOPS，精确建模不同掩码下的计算负载（图 7a causal mask 中上三角块被 mask 掉）

3. **通信避免**：masked 计算块不生成（如 M_{ij} = 0 的 Q_i × KV_j），对应数据块无需传输

4. **与其他并行正交**：
   - **Tensor Parallelism (TP)**：可在 head 维度联合使用，DCP 占用传统 DP 的设备等级
   - **Pipeline Parallelism (PP)**：不同 stage 可用不同 CP 配置，建议 TP-CP-DP-PP 的等级顺序（§6.2）
   - **Data Parallelism (DP)**：DCP 在 DP group 间应用（图 5c 长序列用 CP，短序列用 DP）

## 适用场景

- **长上下文预训练**：Llama 3 tuning 阶段，128K 上下文样本仅占 0.1%，但训练成本极高
- **指令微调**：数据集包含大量短序列（<4K）+ 少量长序列（>100K），如 LongAlign
- **多任务训练**：不同任务使用不同注意力模式（RLHF 的 shared question mask、retrieval 的 lambda-shaped、causal LM 训练）
- **推理优化**：虽然论文聚焦训练，但 DCP 的动态分区思想可扩展到批处理推理（不同请求长度差异大）

## 实现要点

- **代码量**：核心模块 14K LOC Python（dataloader、planner、executor）+ 300 LOC C++（通信/计算调度加速）
- **依赖**：FlashAttention（block attention kernel）、Triton（fused kernel）、KaHyPar（超图分区）
- **开销**：
  - Planning：在 CPU 异步执行，与 GPU 训练重叠（lookahead κ 迭代）
  - Executor：块级缓冲区按类型索引复用，避免频繁分配

---

## 参考资料

- 论文：Chenyu Jiang et al., "DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism", SOSP'25
- 代码：https://github.com/chenyu-jiang/dcp
- arXiv：https://arxiv.org/abs/2510.10620
