# Sticky Until Saturated：LLM 路由为什么要“粘住直到饱和”

这篇文章讲的是一个很实用的问题：**LLM 推理请求该怎么路由到不同的服务端 endpoint**。

作者想解决的，不是“怎么把请求发出去”这么简单，而是更具体的：

- 如何让请求尽量命中已有的 KV cache
- 如何避免某个热 endpoint 被一直粘住，最后变成热点
- 如何让路由策略既能解释、又能调得动、还能跨负载复用

一句话概括：

> 不是单纯追求“离缓存最近”，而是要做到 **先亲和，后释放；先粘住，饱和后再分流**。

---

## 1. 这篇文章到底在反对什么？

作者先批评了一个旧思路：**把多个信号加权混合起来做路由**。

旧默认策略大概是把这些东西一起算分：

- prefix-cache 匹配程度
- 队列深度
- KV cache 利用率
- LRU / 最近未命中情况

问题在于：

1. **很难直觉理解** —— 分数是怎么来的，不容易说清楚
2. **很难调参** —— 一个权重改了，整体行为可能就变了
3. **热点问题会被放大** —— 缓存越热的 endpoint 越容易继续吸流量，最后过载

所以作者提出一个更“工程化”的方法：

> 先判断 workload 的瓶颈是什么，再用和瓶颈匹配的单一负载信号去路由。

这就是他们说的 **token-aware routing**。

---

## 2. 核心思想：按瓶颈选路由信号

LLM 推理通常可以粗分成两个阶段：

### Prefill
处理 prompt，把输入 tokens 一段段灌进去。

### Decode
逐 token 生成输出，依赖 KV cache 和并发解码槽位。

不同 workload，先满的瓶颈不一样：

- **prefill-bound**：prompt 很长，输入处理先成为瓶颈
- **decode-bound**：输出很长，解码槽位先成为瓶颈

于是路由策略也该不同：

### 对 prefill-bound 负载
用：

- `prefix-cache-affinity-filter`
- `token-load-scorer`

意思是：

- 先尽量把请求留给已经有前缀缓存的 endpoint
- 但衡量负载时，看的是**未缓存的 prefill tokens in flight**

也就是：谁当前还要处理的“新输入工作”更少，就优先发给谁。

### 对 decode-bound 负载
用：

- `prefix-cache-affinity-filter`
- `active-request-scorer`

意思是：

- 仍然考虑缓存亲和性
- 但负载信号改成 **active requests 数量**

因为 decode 阶段更关心的是：

> 谁的并发生成流已经太多了？

---

## 3. “Sticky until saturated” 是什么意思？

这是文章标题里的关键。

可以把它理解成一句策略口号：

> **只要某个 endpoint 还没饱和，就让请求尽量粘在它上面；一旦超过阈值，就不再死守亲和性，改按负载分流。**

这和传统“永远优先缓存命中”的策略不同。

传统策略的问题是：

- 热 endpoint 会越来越热
- 冷 endpoint 一直吃不到活
- 最后 warm endpoint 可能比 cold endpoint 更慢

所以这里需要一个“释放阈值”——也就是 `τ`。

---

## 4. τ 是什么？为什么这么重要？

`τ` 可以理解成：

> **一个 warm endpoint 最多可以比别的 endpoint 多背多少“未缓存的 in-flight token 工作量”**，超过了就必须释放粘性。

它不是拍脑袋出来的，而是从硬件和 SLO 推导的。

作者给出的思路可以简化为：

- `B` = 每次 prefill 的 batch 大小，比如 `max-num-batched-tokens`
- `R_peak` = 峰值 prefill 吞吐
- `T_max` = 你能接受的 TTFT 退化上限

那么阈值可以写成：

> `τ_sat = R_peak × T_max`

直观理解：

- 如果 warm endpoint 已经背了太多未完成的输入工作
- 继续把请求粘过去，会让 TTFT 超出你能接受的范围
- 所以到点就“解粘”

文章里给的例子是：

- `B = 8192`
- `R_peak = 20480 tok/s`
- `T_max = 14s`
- 得到 `τ = 286720`

也就是：

> 允许 warm endpoint 领先大约 35 个 8192-token chunk 的未缓存 prefill 工作量。

这就是“sticky until saturated”的工程化落点。

---

## 5. 三个 workload 分别说明了什么？

文章拿了 3 类 workload 来验证这个思路。

### 5.1 code-generation：典型 prefill-bound

特征：

- prompt 很长
- 输出相对短
- 没有太多跨请求共享前缀

这类场景下，瓶颈是 **prefill compute**。

结论是：

- `prefix-cache-affinity + token-load` 表现最好
- TTFT 更低
- 输入吞吐更高
- 传统 k8 round-robin 容易更早撞到性能拐点

你可以把它理解成：

> 长 prompt 场景里，最值钱的是“谁的未完成输入更少”，而不是单纯谁最近命中过缓存。

---

### 5.2 reasoning：典型 decode-bound

特征：

- prompt 较短
- 输出很长
- 并发生成流很多

这类场景下，瓶颈是 **decode slots**。

结论是：

- `prefix-cache-affinity + active-request` 更合适
- 因为它直接按活跃流数均衡 decode 压力
- 在 decode 侧更容易维持均衡

这个例子说明：

> 对 decode-bound 负载，token 不是最合适的负载单位，活跃请求数更贴近真实瓶颈。

---

### 5.3 b2b-saas：病态 prefill-bound 压力测试

这是最有意思的一个。

它不是普通业务流量，而是一个**重复小语料的压力测试**：

- 很多 system prompt 会重复出现
- 缓存命中会非常关键
- 如果分流不对，容易出现 cache churn（缓存抖动/抖出抖进）

这个 workload 说明了两件事：

1. **缓存亲和策略确实有用**，能把 repeated corpus 稳定分到各个 endpoint 上
2. **τ 太小或太大都不好**
   - 太小：很快释放粘性，缓存分区不稳定
   - 太大：粘得太久，热点会累积过载

它像是一个“验证饱和阀门”的实验。

---

## 6. 为什么作者说旧的加权混合不如这个方案？

不是说加权混合一定错，而是说它有几个问题：

### 1）不可解释
你很难一眼说出：

- 当前是因为缓存亲和高所以被选中？
- 还是因为队列短？
- 还是因为 KV 利用率低？

### 2）不可迁移
权重调得好，可能只对某个 workload 好。
换个负载，原来的权重就不一定合适。

### 3）不容易做“饱和释放”
如果没有明确阈值，就容易一路偏向 warm endpoint，导致热点。

作者的方案把复杂度收敛成了：

- 一个 affinity filter
- 一个和瓶颈匹配的负载信号
- 一个可计算的 saturation threshold

这就更像一个可部署的系统，而不是一堆经验权重。

---

## 7. 这篇文章真正的工程启发是什么？

我觉得有 4 个关键词：

### 1）先识别瓶颈
不要先问“用什么算法”，先问：

- 是 prefill 卡住了？
- 还是 decode 卡住了？
- 还是 workload 本身高度混合？

### 2）让信号贴近瓶颈

- prefill 看 token load
- decode 看 active requests
- 混合/未知场景再考虑 predictor

### 3）亲和性不是越强越好
缓存亲和有价值，但必须有释放机制。
否则“粘住”会变成“粘死”。

### 4）阈值最好能推导
不要靠拍脑袋调一个“差不多”的数字。
如果能从硬件吞吐和 SLO 反推出来，策略就更稳、更可解释，也更容易在不同集群复制。

---

## 8. 如果把这篇文章翻译成一句人话

可以这么说：

> LLM 路由的关键，不是永远追着缓存跑，而是让请求先跟着缓存走，直到某个 endpoint 背不动了，再按真实瓶颈分流。这样既保留缓存收益，又避免热点过载。

---

## 9. 适合直接记住的结论

- **prefill-bound**：用 `prefix-cache-affinity + token-load`
- **decode-bound**：用 `prefix-cache-affinity + active-request`
- **sticky until saturated**：先粘住，超过阈值再释放
- **τ**：一个可由硬件和 SLO 推导的饱和阈值
- **核心目标**：既吃到 KV cache 的收益，又避免热点 endpoint 被粘爆

---

如果你愿意，我可以继续帮你把这篇内容整理成一篇**可直接发布的中文技术 blog**，我可以给你两种版本：

1. **偏科普讲解版**：适合大众读者，通俗但不失技术细节
2. **偏工程分析版**：适合 LLM infra / 推理系统读者，结构更硬核

你也可以直接让我下一步帮你输出成 `Markdown` 正文。