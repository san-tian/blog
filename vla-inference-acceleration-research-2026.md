# VLA 推理加速：研究进展与现状（截至 2026-08-30）

## 摘要

VLA（Vision-Language-Action）推理加速已经从“把模型做小”发展为一个跨算法、模型、运行时和硬件的系统问题。当前瓶颈通常同时出现在四个位置：视觉 token/跨帧特征重复计算，VLM/LLM 主干的深度与 KV-cache，动作生成的自回归或多步 flow/diffusion 解码，以及机器人闭环中的同步等待和动作块陈旧。

本文把“推理加速”限定为 inference efficiency，不展开提升 CoT/空间推理能力本身；仅在 reasoning token 直接造成延迟时涉及。检索以 arXiv 元数据、论文正文和项目页为主，最后核验日期为 2026-08-30。所有性能数字均视为原作者报告；“已接收”状态以 arXiv comment/论文版本为依据，并不等价于独立复现。

截至 2026 年 8 月，最可靠的进展可以概括为：

1. **低风险、无需重新训练的加速**已经稳定在约 1.4–1.7x：跨帧 KV-cache（[VLA-Cache](https://arxiv.org/abs/2502.02175)）、视觉 token 剪枝（[LightVLA](https://arxiv.org/abs/2509.12594)）和动作感知的 speculative decoding（[Spec-VLA](https://arxiv.org/abs/2507.22424)）等。
2. **需要微调/蒸馏的解码加速**可达到约 2–4x：action chunking（[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)）、并行 fixed-point/Jacobi 解码（[PD-VLA](https://arxiv.org/abs/2503.02310)、[CEED-VLA](https://arxiv.org/abs/2506.13725)）、一致性蒸馏和一步 flow 解码（[FreqPolicy](https://arxiv.org/abs/2506.08822)）等。
3. **架构重设计或多技术叠加**已经出现 4–15x 的作者报告（例如 [FlashDrive](https://arxiv.org/abs/2608.12932)、[WAM-Diff2](https://arxiv.org/abs/2608.01035)），但大多是预印本、特定硬件/任务和特定测量口径，不能视为统一 SOTA。
4. **边缘部署正在从“可行性演示”走向可用**：0.2–0.45B 的小 VLA（[SmolVLA](https://arxiv.org/abs/2506.01844)、[TurboVLA](https://arxiv.org/abs/2607.27205)）已能在消费级 GPU 上达到约 30 Hz；但动态、接触丰富、长时程真机任务的稳定 20–50 Hz 闭环仍未解决。

最重要的判断是：VLA 的“action throughput（每秒产生多少动作）”不能替代“端到端闭环控制频率”（[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 对这一区分给出了典型测量）。报告性能时必须同时给出单次 observation-to-action/chunk 延迟、实际 observation refresh Hz、动作块长度、成功率、显存和功耗。

## 1. 先定义问题：VLA 推理栈的计算在哪里

典型 VLA 的一次控制调用可以抽象为：

```text
相机/本体状态 -> 视觉编码 -> 多模态融合/VLM 主干 -> 动作解码器 -> 动作块 -> 执行与重新观测
```

不同范式的主要成本不同：

| 范式 | 主要计算 | 典型问题 |
|---|---|---|
| AR 离散动作（[RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246)） | LLM/VLM prefill + 逐 token 解码 | token 串行，动作维度和 chunk 长度会放大解码成本 |
| AR + 连续动作头（[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)） | 一次主干前向 + 连续动作回归 | 主干仍重，但可以把多个时间步放入一个 chunk |
| Flow/diffusion 动作头（[π0](https://arxiv.org/abs/2410.24164)、[SmolVLA](https://arxiv.org/abs/2506.01844)） | 主干条件下的多次积分/去噪 | 每个动作块需要多次 forward，实时性受采样步数限制 |
| 分层/双频策略（[HiRT](https://arxiv.org/abs/2410.05273)、[TIDAL](https://arxiv.org/abs/2601.14945)、[SmolVLA async](https://arxiv.org/abs/2506.01844)） | 慢语义路径 + 快控制路径 | 需要处理 stale feature、动作连续性和调度 |
| 自动驾驶 VLA | 视频编码、长上下文、推理 token、轨迹 flow | 多阶段级联，单点优化收益受其他阶段限制 |

一次闭环的端到端延迟可写成：

\[
T_{e2e}=T_{capture}+T_{transfer}+T_{vision}+T_{backbone}+T_{decode}+T_{queue}+T_{actuation}.
\]

许多论文只测 `T_decode` 或 CUDA kernel latency；这对算子研究有用，但不能直接推断真机控制频率。

### 1.1 技术演进时间线

| 时间 | 关键转折 | 含义 |
|---|---|---|
| 2023 | [RT-2](https://arxiv.org/abs/2307.15818) 建立 VLA/action-token 范式；[SARA-RT](https://arxiv.org/abs/2312.01990) 尝试线性注意力 up-training | 大型 VLM 成为能力来源，也成为延迟瓶颈 |
| 2024 | [OpenVLA](https://arxiv.org/abs/2406.09246) 开源 7B 基线；[HiRT](https://arxiv.org/abs/2410.05273) 双频控制；[DeeR-VLA](https://arxiv.org/abs/2411.02359) 动态早退；[TinyVLA](https://arxiv.org/abs/2409.12514)/Smol-style 小模型萌芽 | 研究从“能控制”转向“如何实时控制” |
| 2025 H1 | [FAST](https://arxiv.org/abs/2501.09747)、[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)、[VLA-Cache](https://arxiv.org/abs/2502.02175)、[PD-VLA](https://arxiv.org/abs/2503.02310)、[MoLe-VLA](https://arxiv.org/abs/2503.20384)、[SmolVLA](https://arxiv.org/abs/2506.01844) | action representation、chunk、跨帧复用、动态层和异步系统形成主线 |
| 2025 H2 | [FreqPolicy](https://arxiv.org/abs/2506.08822)、[Spec-VLA](https://arxiv.org/abs/2507.22424)、[SpecPrune-VLA](https://arxiv.org/abs/2509.05614)、[LightVLA](https://arxiv.org/abs/2509.12594) | 一步 flow、speculative decoding、动作感知 token pruning 走向成熟 |
| 2026 H1 | [QVLA](https://arxiv.org/abs/2602.03782)/[QuantVLA](https://arxiv.org/abs/2602.20309)、[Shallow-π](https://arxiv.org/abs/2601.20262)、[TIDAL](https://arxiv.org/abs/2601.14945)、[AC²-VLA](https://arxiv.org/abs/2601.19634)、[ActQuant](https://arxiv.org/abs/2605.24011) | 动作敏感低比特、flow 蒸馏、跨时间自适应计算成为热点 |
| 2026 H2 | [TurboVLA](https://arxiv.org/abs/2607.27205)、[Embodied.cpp](https://arxiv.org/abs/2607.02501)、[FlashDrive](https://arxiv.org/abs/2608.12932)、[WAM-Diff2](https://arxiv.org/abs/2608.01035)、[streaming FlashVLA](https://arxiv.org/abs/2608.27384) | 架构、runtime、kernel、异步和硬件开始联合设计 |

## 2. 技术路线总览

### 2.1 Action chunking、并行解码和动作表示

**[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)（RSS 2025）**是目前最有影响力的基线级工程配方：parallel decoding、action chunking、continuous action representation 和 L1 regression。它把 [OpenVLA](https://arxiv.org/abs/2406.09246) 在 LIBERO 的平均成功率从 76.5% 提升到 97.1%，作者报告 action generation throughput 提升 26x。正文的 A100 测量更能说明问题：原生 OpenVLA 约 4.2 action/s、单 query 239.6 ms；OFT（chunk=8）约 109.7 action/s、单 chunk 72.9 ms。也就是说，26x 是“每秒动作数”，而不是 26x 的端到端单次调用降延迟。

**[PD-VLA](https://arxiv.org/abs/2503.02310)（IROS 2025）**把带 action chunk 的 AR 解码重写为 fixed-point 并行迭代，training-free，7-DoF 执行频率作者报告提升 2.52x。它的价值在于不改主干架构，但收益依赖 fixed-point 收敛步数和动作容忍度。

**[CEED-VLA](https://arxiv.org/abs/2506.13725)（2025 预印本）**进一步使用 consistency distillation、mixed-label supervision 和 early-exit。作者在 [OpenVLA](https://arxiv.org/abs/2406.09246) 上报告约 4.1x speedup，在 [LLaVA](https://arxiv.org/abs/2304.08485)-VLA 自建基线上约 2x；真机实验的平均频率从 3.3 Hz（LLaVA-VLA）提升到 13.0 Hz，平均成功率从 33.25% 提升到 77.5%。其关键观察是 Jacobi 的最后若干迭代 token 变化很小，严格 fixed-point 收敛条件浪费了延迟。

**[FAST](https://arxiv.org/abs/2501.09747)（2025）**用离散余弦变换把连续动作压缩为频域 token，主要解决高频灵巧动作的 tokenization 和训练问题。作者报告配合 [π0](https://arxiv.org/abs/2410.24164) 可将训练时间最多降低 5x，但这不是部署期推理加速；而且 π0 的 flow 版本与 FAST 的 AR 版本之间不能直接做速度比较。

### 2.2 Flow/diffusion 动作头：多步采样变少步或一步

[π0](https://arxiv.org/abs/2410.24164)/[π0.5](https://arxiv.org/abs/2504.16054) 代表的 flow-matching VLA 可以生成连续、多模态动作，但一次动作块需要多次积分。当前路线有三种：

- **Consistency / frequency consistency**：[FreqPolicy](https://arxiv.org/abs/2506.08822)（NeurIPS 2025）在 flow policy 上加入频域一致性，支持一步 action generation；集成 VLA 后作者在 LIBERO 40 任务上报告无性能下降，真机推理频率 93.5 Hz。
- **Normalizing flow**：[NinA](https://arxiv.org/abs/2508.16845)（2025 预印本）把 diffusion action decoder 换成可逆的 normalizing flow，一步采样；作者报告在 [FLOWER](https://arxiv.org/abs/2509.04996) 上与 diffusion 对手性能相当但更快。
- **范式蒸馏**：[WAM-Diff2](https://arxiv.org/abs/2608.01035)（2026 预印本）把自回归 VLA 蒸馏到并行离散 diffusion；作者报告解码 2.8x，加 FlashInfer/CUDA Graph 后总加速 15.1x。该结果来自自动驾驶场景，不能直接外推到操作臂。

### 2.3 Backbone 压缩：层跳过、动态深度、蒸馏和小模型

- **[SARA-RT](https://arxiv.org/abs/2312.01990)（2023 预印本）**用 up-training 将既有机器人 Transformer 的二次注意力替换为线性注意力；论文正文中 [RT-2](https://arxiv.org/abs/2307.15818) TPU forward 从 53.2 ms 降到 45.7 ms，约 14%，性能近似保持。它说明只改 attention 往往只能拿到局部收益，无法消除整个 VLA 栈的其他瓶颈。
- **[DeeR-VLA](https://arxiv.org/abs/2411.02359)（NeurIPS 2024）**使用多出口动态 early-exit，根据延迟、功耗和显存约束选择激活深度；CALVIN 上作者报告 LLM 计算量降低 5.2–6.5x、LLM GPU memory 降低 2–6x，任务性能基本保持。
- **[MoLe-VLA](https://arxiv.org/abs/2503.20384)（2025 预印本）**将 LLM 层视为可路由 experts，用空间-时间 router 动态激活层，并以 CogKD 补偿被跳过的认知能力；作者报告计算成本最多降 5.6x、10 任务平均成功率提升 8 个百分点。
- **[RoboMamba](https://arxiv.org/abs/2406.04339)（NeurIPS 2024）**以 Mamba/SSM 替代 Transformer 处理视觉-语言序列，作者报告推理约 3x，且只需微调约 0.1% 参数；它更像重新设计骨干，而非即插即用压缩。
- **[Shallow-π](https://arxiv.org/abs/2601.20262)（2026 预印本）**把 flow VLA 深度从 18 层蒸馏到 6 层，作者报告超过 2x 推理加速、成功率绝对下降小于 1 个百分点，并在 Jetson Orin/Thor 上做了真机验证。
- **[TinyVLA](https://arxiv.org/abs/2409.12514) / [SmolVLA](https://arxiv.org/abs/2506.01844)**走“直接训练小模型”路线。TinyVLA 的当前预印本版本在同一 A6000 上报告 TinyVLA-1B 约 14 ms、[OpenVLA-7B](https://arxiv.org/abs/2406.09246) 约 292 ms，约 20x 延迟差，但版本演进较多，适合作为早期证据而非统一 SOTA。SmolVLA 约 0.45B 参数、使用冻结 SmolVLM-2 和 flow action expert，作者报告可在消费级 GPU 甚至 CPU 部署；其 LIBERO 0.45B 版本平均成功率 87.3%，并且较 [π0](https://arxiv.org/abs/2410.24164) 训练约快 40%、显存约少 6x。
- **[TurboVLA](https://arxiv.org/abs/2607.27205)（2026 预印本）**直接绕开 LLM-centric 的 `V -> L -> A` 路径，改为轻量 `V + L -> A`。作者报告 0.2B 参数、RTX 4090 上 31.2 ms、0.9 GB inference VRAM、约 32 Hz，LIBERO 平均成功率 97.7%。这是很激进的架构结果，尚需跨任务和跨机器人复现。

### 2.4 量化、剪枝和 token/KV 压缩

#### 量化

- **[QuantVLA](https://arxiv.org/abs/2602.20309)（CVPR 2026，arXiv 元数据显示会议归属）**是首个面向 VLA 的 PTQ 框架之一，并首次处理 DiT action head。它采用 selective quantization、attention temperature matching 和 output-head balancing；作者报告量化模块相对显存节省约 70%，任务成功率不低于全精度基线。
- **[QVLA](https://arxiv.org/abs/2602.03782)（ICLR 2026，arXiv 元数据显示会议归属）**强调“动作敏感”而非单纯权重/激活重建误差，按通道测量最终 action-space sensitivity，并把 0-bit pruning 纳入统一优化。[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 上作者报告仅使用原始 VRAM 的 29.2%，保留 98.9% 性能、1.49x speedup，比 [SmoothQuant](https://arxiv.org/abs/2211.10438) 高 22.6% 性能。
- **[BitVLA](https://arxiv.org/abs/2506.07530)（2025 预印本/WIP）**是原生 ternary/1-bit VLA，结合 1.58-bit 视觉编码器蒸馏；作者报告相对 [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) memory 降 11x、端到端 latency 降 4.4x，但目前不应与已接收工作同等看待。
- **[ActQuant](https://arxiv.org/abs/2605.24011)（2026 预印本）**把 action-guided mixed-precision PTQ 与原生 C/C++ 低比特 runtime 结合。作者报告 [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 在不高于 3 bpw 时保留 95.0% 性能；2.5 bpw 时保留 90.1%，backbone 从 14.3 GB 压到 2.7 GB（5.3x）。在 UR3 真机上，量化 [π0.5](https://arxiv.org/abs/2504.16054) 保持 baseline 成功率、内存降低 2.5x。

VLA 量化的独特难点是：小的动作误差会沿闭环累积，不能只用 perplexity、重建误差或单步 MSE 判断。未来更合理的量化校准目标应包括动作空间敏感度、接触状态、长时程成功率和安全约束。

#### 剪枝、视觉 token 与 KV-cache

- **[VLA-Cache](https://arxiv.org/abs/2502.02175)（NeurIPS 2025）**跨帧复用变化很小的视觉 token 的 KV，只重算任务相关 token；作者报告 CUDA latency 最多 1.7x、控制频率提升 15%，成功率几乎不变。
- **[Gated VLA-Cache](https://arxiv.org/abs/2608.10824)（IROS 2026，arXiv comment 标注接收）**用 action-token top-2 logit margin 作为置信门控，低置信时失效缓存并完整重算。在 LIBERO Goal/Long 上作者报告保留约 80% compute savings，同时恢复盲缓存损失的 100% 以上。
- **[LightVLA](https://arxiv.org/abs/2509.12594)（2025 预印本）**用可微 query + Gumbel softmax 学习视觉 token 选择，作者报告 FLOPs 降 59.1%、latency 降 38.2%、成功率提升 2.6 个百分点。
- **[SpecPrune-VLA](https://arxiv.org/abs/2509.05614)（ICML 2026，arXiv comment 标注接收）**结合历史/局部信息、层级动态剪枝和动作粗细粒度控制；作者报告 LIBERO 1.57x、真机 1.70x，成功率近乎无损。
- **[EfficientVLA](https://arxiv.org/abs/2506.10100)（2025 预印本）**同时做语言层剪枝、任务感知视觉 token 选择和 diffusion 中间特征缓存；在 [CogACT](https://arxiv.org/abs/2411.19650)/SIMPLER 上作者报告 1.93x、FLOPs 只剩 28.9%、成功率下降 0.6 个百分点。
- **[FlashVLA（Think Twice, Act Once）](https://arxiv.org/abs/2505.21200)（2025 预印本）**复用稳定时间步的动作并选择视觉 token，作者报告 FLOPs 降 55.7%、latency 降 36.0%、成功率下降 0.7 个百分点。它与 2026 年同名的 [streaming FlashVLA](https://arxiv.org/abs/2608.27384) 不是同一篇论文。
- **[Learned adaptive visual-token caching](https://arxiv.org/abs/2602.00686)（2026 预印本）**把 cache ratio 和 token 选择变成可学习决策；作者报告 wall-clock 1.76x，LIBERO 平均成功率由 75.0% 升到 76.9%，真机提升 5 个百分点。
- **[DySta](https://arxiv.org/abs/2602.03983)（2026 预印本）**把多帧视觉拆成静态/动态 token，只保留一份静态上下文并用 gate 决定何时重建 KV；作者报告仿真 2.0x 且成功率 +2.3 个百分点，真机 2.2x 且成功率 +10.6 个百分点。

2026 年的新趋势是从“删 token”转向“保留 token 但压缩 value 表示”：[RoleSub](https://arxiv.org/abs/2608.18410) 报告在匹配 visual-KV 预算时多数设置优于 token-only control；[Action-JND](https://arxiv.org/abs/2608.21247) 则直接以“token 扰动对闭环动作的可容忍度”决定压缩强度。这些结果仍处于预印本阶段。

### 2.5 Speculative decoding、流式和双频控制

- **[Spec-VLA](https://arxiv.org/abs/2507.22424)（EMNLP 2025）**把 speculative decoding 引入 VLA，并用 action-token 的相对距离放宽接受条件；作者报告接受长度提升 44%、相对 [OpenVLA](https://arxiv.org/abs/2406.09246) 1.42x，成功率无明显损失。
- **[HiRT](https://arxiv.org/abs/2410.05273)（CoRL 2024）**是双频控制的早期代表：低频 VLM 提取暂时不变的语义特征，高频视觉策略负责反应；静态任务控制频率翻倍，动态真机任务成功率由 48% 提升到 75%。
- **[TIDAL](https://arxiv.org/abs/2601.14945)（2026 预印本）**缓存低频 macro-intent，在高频 micro-control 中做单步 flow integration；作者报告边缘设备约 9 Hz，相比约 2.4 Hz baseline，动态拦截性能约 2x，但静态任务有轻微回退。
- **[SmolVLA asynchronous stack](https://arxiv.org/abs/2506.01844)（2025）**将执行、观测和策略预测并行化。SO100 实验中，同步完成一个任务平均 13.75 s，异步 9.7 s（约快 30%）；固定时间内成功 pick-and-place 循环从 9 次增至 19 次。
- **[FlashVLA: Streaming Action Decoding](https://arxiv.org/abs/2608.27384)（2026-08 预印本）**维护不同噪声水平的 action-chunk buffer，用 chunk-wise causal attention 每次产生一个可执行 chunk；作者报告单 GPU 真机异步控制频率至少 30 Hz。

这条路线的核心不是单纯提高 GPU utilization，而是让机器人在等待慢路径时仍能执行安全、连续、不过时的动作。难点是缓存语义何时失效、动作块如何重叠、观测延迟如何进入训练，以及异常时如何立即切回慢路径。

### 2.6 运行时和硬件协同

- **[Embodied.cpp](https://arxiv.org/abs/2607.02501)（2026 预印本）**提供面向异构机器人的 C++ runtime，包含输入适配、序列构建、backbone、head plugin 和 deployment adapter 五层；跨 3 个 VLA 和 2 个 WAM，作者报告相对 Python baseline 1.05–2.70x、VRAM 降 7–77%。这是工程落地很重要的一类结果，因为它减少了 Python glue、数据拷贝和多速率执行不一致。
- **[FlashDrive](https://arxiv.org/abs/2608.12932)（2026 预印本）**同时优化四阶段：视频重叠的 streaming KV reuse、LM prefill context reuse、低熵 reasoning token 的 diffusion drafter、flow velocity 的 adaptive step cache，再叠加 CUDA Graph/kernel fusion。Alpamayo 1.5-10B + W4A8 上作者报告端到端 717 ms 降到 151 ms（4.7x），单 GPU 1.4 Hz 提升到 6.6 Hz。
- **[SpecVLA](https://arxiv.org/abs/2608.15636)（2026 预印本）**利用机器人“活跃/非活跃”状态，非活跃段长动作 speculative prediction，活跃段用小验证模型选择性验证；同时设计 GPU + 专用模块的异构数据流。
- **[Deltoris](https://arxiv.org/abs/2608.04428)（2026 预印本）**针对 diffusion VLA 使用 temporal-aware bit sparsity、speculative inference 和 bit-serial systolic accelerator；作者报告相对 mobile GPU 最高 34.2x、相对已有 accelerator 6.1x，但属于强硬件协同结果，不能与通用 GPU 软件加速横比。
- **[EcoVLA（device-edge co-inference）](https://arxiv.org/abs/2608.15502)（2026 预印本）**根据网络和设备状态选择在哪个阶段切分模型；20 Hz action-output 约束下，作者报告系统能效相对既有 co-inference 方法最高提升 236%。这个指标是能效而非延迟倍数。

## 3. 证据分层与代表性结果

| 成熟度 | 代表工作 | 主要结果（作者报告） | 适合的结论 |
|---|---|---|---|
| 已接收/同行评审 | [HiRT](https://arxiv.org/abs/2410.05273)、[DeeR-VLA](https://arxiv.org/abs/2411.02359)、[RoboMamba](https://arxiv.org/abs/2406.04339) | 约 2x 控制频率；5.2–6.5x LLM compute 降；约 3x 推理 | 双频、早退、SSM 路线有较早实证基础 |
| 已接收/同行评审 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)、[VLA-Cache](https://arxiv.org/abs/2502.02175)、[PD-VLA](https://arxiv.org/abs/2503.02310)、[FreqPolicy](https://arxiv.org/abs/2506.08822)、[Spec-VLA](https://arxiv.org/abs/2507.22424) | 26x action throughput；1.7x cache；2.52x parallel；93.5 Hz one-step；1.42x speculative | 目前最值得复现的通用操作基线 |
| 已接收/同行评审 | [QuantVLA](https://arxiv.org/abs/2602.20309)、[QVLA](https://arxiv.org/abs/2602.03782)、[SpecPrune-VLA](https://arxiv.org/abs/2509.05614)、[Fast-ThinkAct](https://arxiv.org/abs/2601.09708) | 约 1.49–1.57x；量化 VRAM 29.2%；reasoning latency 最多降 89.3% | 动作敏感压缩和 latent planning 已成为主流方向 |
| 预印本/WIP | [TurboVLA](https://arxiv.org/abs/2607.27205)、[BitVLA](https://arxiv.org/abs/2506.07530)、[Shallow-π](https://arxiv.org/abs/2601.20262)、[TIDAL](https://arxiv.org/abs/2601.14945)、[FlashDrive](https://arxiv.org/abs/2608.12932)、[WAM-Diff2](https://arxiv.org/abs/2608.01035)、[2026 FlashVLA](https://arxiv.org/abs/2608.27384) | 2–15x 或 30 Hz 级别 | 方向很有潜力，但需跨硬件、协议和真机复核 |

截至检索日，2026 年新方法的 arXiv 预印本比例很高。表中“已接收”主要依据 arXiv 页面 comment/论文版本信息，正式会议论文集和独立复现实验仍应作为最终引用前的二次核验。

## 4. 为什么论文速度不能直接排名

至少有五个口径差异会改变结论：

1. **action throughput vs query latency**：`K / T_chunk` 会随动作块长度 K 增大，即使机器人只能在 chunk 结束后重新观测。
2. **GPU kernel latency vs sensor-to-action latency**：前者忽略相机、PCIe、预处理、网络、队列和执行器。
3. **control Hz vs observation refresh Hz**：异步策略可以保持高控制频率，但慢语义特征可能已过时。
4. **单视角 vs 多视角**：[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) LIBERO 常见单张 224×224；ALOHA 表中是三张图和 14-D 状态，不能混用。
5. **硬件和 batch**：A100/H100/4090/Jetson 的 kernel、显存带宽和量化支持差异很大；batch=1 才接近闭环机器人。

建议统一报告以下指标：

```text
P50/P95 端到端 latency
单 action 与单 chunk latency
实际 observation refresh Hz、执行 Hz、jitter
chunk 长度 K 与重规划策略
成功率/碰撞率/近接触失败率
峰值 VRAM、模型参数、功耗和温度
硬件、batch、输入相机和软件栈版本
```

## 5. 目前的研究现状判断

### 已经相对成熟

- 在 [OpenVLA](https://arxiv.org/abs/2406.09246)/[OFT](https://arxiv.org/abs/2502.19645) 这类 AR VLA 上，action chunking + continuous action head + parallel decoding 是最成熟的第一步。
- 跨帧视觉 token KV-cache（[VLA-Cache](https://arxiv.org/abs/2502.02175)）、轻量 token pruning（[LightVLA](https://arxiv.org/abs/2509.12594)）和 relaxed speculative decoding（[Spec-VLA](https://arxiv.org/abs/2507.22424)）已能在不重训或少量校准下获得约 1.4–1.7x，适合作为现有系统的低风险优化。
- 双频/异步执行已经从概念变成可运行系统；[SmolVLA](https://arxiv.org/abs/2506.01844)、[HiRT](https://arxiv.org/abs/2410.05273)、[TIDAL](https://arxiv.org/abs/2601.14945) 说明“慢语义 + 快控制”比强行让一个 7B/10B 模型每个控制周期完整运行更现实。

### 正在快速发展

- **动作感知压缩**：量化和剪枝目标从权重误差转向最终动作、接触状态和任务成功率（[QVLA](https://arxiv.org/abs/2602.03782)、[SpecPrune-VLA](https://arxiv.org/abs/2509.05614)）。
- **解码范式改变**：AR→parallel/diffusion、multi-step flow→consistency/normalizing flow，一次 forward 取代多次迭代（[FreqPolicy](https://arxiv.org/abs/2506.08822)、[NinA](https://arxiv.org/abs/2508.16845)）。
- **动态计算**：根据环境阶段、动作粗细、置信度和过去 action 决定层数、token 数、精度和是否调用慢模型（[DeeR-VLA](https://arxiv.org/abs/2411.02359)、[MoLe-VLA](https://arxiv.org/abs/2503.20384)、[TIDAL](https://arxiv.org/abs/2601.14945)）。
- **小型原生 VLA**：0.2–0.45B 级模型在消费级 GPU 30 Hz 左右已出现可信结果（[SmolVLA](https://arxiv.org/abs/2506.01844)、[TurboVLA](https://arxiv.org/abs/2607.27205)），研究问题从“能否压缩”转为“如何保持跨 embodiment 和长时程泛化”。
- **运行时/硬件协同**：CUDA Graph、FlashInfer、kernel fusion、C++ runtime、专用 bit-serial accelerator 开始与模型设计共同优化（[Embodied.cpp](https://arxiv.org/abs/2607.02501)、[Deltoris](https://arxiv.org/abs/2608.04428)）。

### 仍然没有解决

1. **没有统一 latency–success benchmark**：不同论文的 GPU、chunk、同步协议和任务集不一致。
2. **闭环稳定性缺乏理论**：量化/剪枝误差如何沿动力学和接触事件放大，尚无通用稳定性界。
3. **缓存与异步的陈旧性**：视觉变化检测不等于动作重要性检测；静态场景有效的 cache 可能在抓取、碰撞和快速目标处失效。
4. **动作多模态与 L1 回归冲突**：L1 速度快且稳定，但可能平均化多个可行动作模式；flow/diffusion 表达力强，却更贵。
5. **真实边缘设备证据少**：Jetson、移动机器人、网络抖动、功耗/热约束下的长期闭环测试仍明显少于 A100/4090 仿真。
6. **操作 VLA 与驾驶 VLA 的方法迁移不足**：自动驾驶的长视频/KV reuse 和操作臂的接触感知、动作敏感剪枝还没有统一框架。
7. **安全与效率尚未联合设计**：低比特权重、动态跳层和 speculative acceptance 对故障、攻击和异常观测的影响缺乏系统评估。

## 6. 推荐的工程路线

### 场景 A：已有 [OpenVLA](https://arxiv.org/abs/2406.09246)，目标是尽快提速

1. 先采用 [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 的连续动作 + action chunk + parallel decoding 配方，建立真实 batch=1 的 P50/P95 端到端基线。
2. 加 [VLA-Cache](https://arxiv.org/abs/2502.02175)/[Gated VLA-Cache](https://arxiv.org/abs/2608.10824)；缓存失效门控用动作 logit margin 或观测变化双重触发。
3. 再评估 [Spec-VLA](https://arxiv.org/abs/2507.22424)/[PD-VLA](https://arxiv.org/abs/2503.02310)；接受条件必须改为动作空间/安全空间，而不是纯 token 精确匹配。
4. 最后做 [QVLA](https://arxiv.org/abs/2602.03782)/[QuantVLA](https://arxiv.org/abs/2602.20309) 风格的动作敏感混合精度，先量化语言层，再单独校准 action head。

### 场景 B：需要真机 20–50 Hz 动态控制

1. 将语义规划与控制拆成双频路径：慢路径 1–5 Hz，快路径 20–50 Hz。
2. 快路径使用 [SmolVLA](https://arxiv.org/abs/2506.01844) 这类小 VLA、flow expert 或蒸馏 student，慢路径只在任务阶段改变、置信度下降或接触事件附近触发。
3. action chunk 不要固定整段 open-loop 执行；采用可中断队列、chunk overlap 和 stale-state compensation。
4. 重点优化 P95 latency、jitter、观测刷新率和异常恢复，而不是只追求平均 action/s。

### 场景 C：Jetson/移动端受显存和功耗约束

1. 优先考虑 0.2–0.5B 小模型或蒸馏 student，再做 W8A8/W4A8；不要一开始把 7B 主干硬塞进端侧。
2. 用 [Shallow-π](https://arxiv.org/abs/2601.20262)/[DeeR-VLA](https://arxiv.org/abs/2411.02359)/[MoLe-VLA](https://arxiv.org/abs/2503.20384) 类动态深度，给任务阶段设计算力预算。
3. 采用 [Embodied.cpp](https://arxiv.org/abs/2607.02501) 的 C++ runtime 思路、CUDA Graph、固定 shape 和预分配 buffer，减少 Python、数据拷贝和 kernel launch 开销。
4. 用真实功耗、温度降频和网络断开测试验证“可持续 Hz”，不能只测短时峰值。

## 7. 推荐优先阅读的论文

### 已接收或基线价值高

- [RT-2](https://arxiv.org/abs/2307.15818)
- [SARA-RT](https://arxiv.org/abs/2312.01990)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [HiRT](https://arxiv.org/abs/2410.05273)
- [DeeR-VLA](https://arxiv.org/abs/2411.02359)
- [RoboMamba](https://arxiv.org/abs/2406.04339)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [VLA-Cache](https://arxiv.org/abs/2502.02175)
- [PD-VLA](https://arxiv.org/abs/2503.02310)
- [FreqPolicy](https://arxiv.org/abs/2506.08822)
- [Spec-VLA](https://arxiv.org/abs/2507.22424)
- [Fast-ThinkAct](https://arxiv.org/abs/2601.09708)
- [QuantVLA](https://arxiv.org/abs/2602.20309)
- [QVLA](https://arxiv.org/abs/2602.03782)
- [SpecPrune-VLA](https://arxiv.org/abs/2509.05614)

### 2026 年值得跟踪的预印本

- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [CEED-VLA](https://arxiv.org/abs/2506.13725)
- [EfficientVLA](https://arxiv.org/abs/2506.10100)
- [BitVLA](https://arxiv.org/abs/2506.07530)
- [ActQuant](https://arxiv.org/abs/2605.24011)
- [Shallow-π](https://arxiv.org/abs/2601.20262)
- [TIDAL](https://arxiv.org/abs/2601.14945)
- [Learned adaptive visual-token caching](https://arxiv.org/abs/2602.00686)
- [DySta](https://arxiv.org/abs/2602.03983)
- [TurboVLA](https://arxiv.org/abs/2607.27205)
- [Embodied.cpp](https://arxiv.org/abs/2607.02501)
- [FlashDrive](https://arxiv.org/abs/2608.12932)
- [WAM-Diff2](https://arxiv.org/abs/2608.01035)
- [FlashVLA streaming](https://arxiv.org/abs/2608.27384)

综述入口：

- [*A Survey on Efficient Vision-Language-Action Models*](https://arxiv.org/abs/2510.24795)
- [*Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey*](https://arxiv.org/abs/2510.17111)

## 结论

VLA 推理加速的主战场已经从单点算子优化转向“动作解码范式 + 动态计算 + 时序复用 + 异步系统”的联合设计。对今天要落地的团队，最稳妥的组合是：**小/中型 VLA 或蒸馏 student + action chunk/连续动作 + 双频异步执行 + 跨帧 KV/token 复用 + 动作敏感量化 + C++/CUDA runtime**。论文中 10x 以上的数字可以作为研究上限参考，但在统一的真机闭环协议出现之前，不应当作为跨论文的直接性能排名。
