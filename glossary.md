# 术语索引（glossary）

跨文章的术语知识库。**写新文章前先查这里**：

- 术语已登记 → 文中直接引用出处（链接原文），**不重复解释**
- 未登记 → 在文章里单出一页/一节仔细讲（直接定义，不用比喻），发布后登记到这里

同一概念全站沿用同一术语（不换说法）；本文用语与已有文章不一致时，以本表为准修正。

## 用法

1. 写作前先 `grep <术语> glossary.md`
2. 命中 → 直接使用该术语并链接出处，不再解释（例：DCP 概念见《DCP 实测》）
3. 未命中 → 文章内首次出现处详细解释（一页卡片/一节，直接定义），发布后把术语加进下表

## 术语表

| 术语 | 一句话定义 | 详细解释出处 |
|---|---|---|
| prefill / decode / PD 分离 | LLM 推理两阶段（处理整段 prompt / 逐 token 生成）及拆分部署 | pd-decode-kvcache-offload.md（PD 分离基础） |
| KV cache（键值缓存） | 注意力为每个 token 生成的中间状态缓存；decode 每步读全部历史 KV | dsa-fused-kv-store-dcp-regression.md（术语段） |
| kernel（GPU kernel / triton kernel） | GPU 上执行的并行计算函数 | dsa-fused-kv-store-dcp-regression.md（术语段）；triton kernel 调优见 sglang-moe-kernel-tuning.md |
| MLA（多头潜在注意力） | 每 token KV 压缩到 ~44KB 的注意力结构（对比 64 KV 头 MHA 省 ~57 倍） | glm52-llm-d-agentic-serving.md（§一） |
| **DSA（DeepSeek Sparse Attention）** | 稀疏注意力架构：先 indexer 选路、再对 top-k 条目计算（两段式） | dsa-fused-kv-store-dcp-regression.html（P2，首次详细解释） |
| **indexer（索引器）** | DSA 的选路模块：fp8 低精度打分器扫全部历史 KV、选出 top-k（GLM-5.2 约 2048） | dsa-fused-kv-store-dcp-regression.html（P2，首次详细解释） |
| indexer KV share | indexer 打分与 attention 计算共用同一份 K（656B 行） | dsa-fused-kv-store-dcp-regression.html（P3） |
| DCP（概念） | KV Cache 切分维度从头维改为序列维，消除高 TP 下 KV 重复存储 | dynamic-context-parallelism.md；本文 P5-8 |
| DCP 上限公式 | DCP 上限 = TP 并行度 ÷ KV 头数（且需整除） | glm52-dcp-config-draft.md |
| DCP（sglang 实现：虚拟 id 换算） | loc 为全局虚拟 slot 编号；写入前换算：归属 rank = v % dcp，物理行号 = v // dcp | dsa-fused-kv-store-dcp-regression.html（P6 详解；排查见 P12）；prefill CP 与 decode CP 对比见 why-prefill-cp-breaks-standalone-decode.md（§4.3） |
| 656B fp8 DSA KV 行布局 | [k_nope fp8(512B) \| scales(16B) \| k_rope bf16(128B)]，indexer 与 attention 共读 | glm52-tpot-decode-optimization.html（§背景） |
| fp8 e4m3fn 量化 | 对称格式，max = -min = 448；per-128 块 fp32 scale | glm52-tpot-decode-optimization.html（§正确性细节） |
| MTP（多 token 预测） | 每步起草多个 token 的投机解码；draft 槽位分走 KV pool | mtp-kv-cache-viz.html |
| MoE / fused_moe_triton / TMA | 专家层 gate/up/down 投影的 triton kernel 与 tile 调参 | sglang-moe-kernel-tuning.md |
| CP（上下文并行） | prefill 按序列长度切块分到多卡（与 DCP 的 decode 侧对应） | why-prefill-cp-breaks-standalone-decode.md |
| AllReduce 融合 | TP 通信的 allreduce + residual + RMSNorm 融合为一次 kernel | glm52-tpot-decode-optimization.html |
| flashinfer 三件套 | flashinfer-python / cubin / jit-cache 必须同版本 | sglang-flashinfer-fusion-error-investigation.md |
| **Kubernetes（K8s）** | 容器编排系统：把「哪些程序跑几份、各用多少资源」声明成 YAML，集群持续拉起维持 | rbg-rolebasedgroup-explained.html（P3 内联术语卡） |
| **Operator / CRD** | 用自定义控制器扩展 K8s 的模式 / 往 K8s 注册新对象类型的机制 | rbg-rolebasedgroup-explained.html（P3 内联术语卡） |
| **TP（张量并行）** | 一个模型多卡协同计算；1 leader + N worker 组成一个推理实例 | rbg-rolebasedgroup-explained.html（P4 内联术语卡；leader-worker 模式 P7） |
| **headless Service** | 不分配虚拟 IP 的 Service，DNS 直达每个 pod，pod 名稳定可寻址 | rbg-rolebasedgroup-explained.html（P8 服务发现） |
| **gang 调度** | 一组 pod 要么全部调度成功要么全不调度，避免多卡只上一半 | rbg-rolebasedgroup-explained.html（P12） |
| **原地更新（in-place update）** | 改镜像只重启容器不重建 pod，名字/IP/节点/GPU 绑定不变 | rbg-rolebasedgroup-explained.html（P11） |
| **RBG 对象层级（角色/实例/组件）** | RoleBasedGroup→Role→RoleInstanceSet→RoleInstance→Pod 四层；角色=服务成员、实例=角色一副本（可能一组 pod）、组件=实例内单个 pod | rbg-rolebasedgroup-explained.html（P14） |
| **CoordinatedPolicy / maxSkew** | RBG 跨角色协同 CRD：滚动更新/扩缩的进度差钳制 | rbg-rolebasedgroup-explained.html（P10） |

> 维护约定：出处列到文章 + 章节/页；同一个术语被多篇文章解释时，登记「首次详细解释」的那篇；粗体行 = 近期新登记。
