# RBG（RoleBasedGroup）：在 K8s 上把多角色 LLM 推理服务当"一个有机体"编排

> 一句话：**RBG 是 [SGLang 项目](https://github.com/sgl-project/rbg)开源的 Kubernetes Operator（Go，Apache-2.0），用一个 CRD 把 PD 分离这类多角色推理服务——router / prefill / decode——的部署拓扑、启动顺序、跨角色协同更新与扩缩、服务发现、gang 调度全部声明式接管**。服务不再是一堆各自为政的 Deployment，而是一个"角色化的有机体"（role-based group）。
>
> 📄 本文是详细证据版；幻灯片版见 [rbg-rolebasedgroup-explained.html](rbg-rolebasedgroup-explained.html)。
>
> 📌 讲解对象：[sgl-project/rbg](https://github.com/sgl-project/rbg) @ commit [`4acd5a7`](https://github.com/sgl-project/rbg/commit/4acd5a791e9709f34c4bd5da569e6427896d21a5)（2026-08-29，v0.7.0 已发布）。文中所有源码引用都锚定该 SHA。官方文档站：[rolebasedgroup.github.io](https://rolebasedgroup.github.io)。

---

## 1. 要解决的问题：原生 K8s 原语编排不了"一个推理服务"

PD 分离（prefill/decode disaggregation）推理服务在集群里长这样：**gateway → router → prefill 池 → decode 池**，外加 KV cache 传输引擎（Mooncake 之类）。用原生 Deployment / StatefulSet 部署，每个组件一个对象、各管各的，会遇到三件烦事（[README「Why RBG?」](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/README.md)）：

| 挑战 | 原生原语的处境 |
|---|---|
| **多角色拓扑** | router 要等 prefill/decode 就绪才有意义；prefill 与 decode 容量要配对。Deployment/StatefulSet 之间没有"谁先谁后、谁跟谁配对"的语义，只能人肉写 init container 或 Helm hook |
| **性能敏感** | GPU 亲和（NVLink/PCIe/RDMA/VPC）、跨角色共置这些拓扑约束，原生调度器一无所知 |
| **跨角色原子操作** | 部署、升级、扩缩、故障恢复都涉及多个角色同时变化。滚动升级时 prefill 全换完、decode 还是旧版，中间态就是版本漂移；扩容时 prefill 扩了 decode 没跟上，中间态就是容量失衡 |

RBG 的答案：把整个推理服务声明成**一个对象**——`RoleBasedGroup`。角色间的启动依赖、协同升级、配对扩缩、服务发现，全是这个对象的控制器语义，不再靠人肉胶水。

## 2. 核心抽象：一个 CRD，四层对象

```
RoleBasedGroup（一个推理服务，例如 pd-disagg-lws）
 │
 ├─ Role: router        （standalone 模式，1 副本）
 ├─ Role: prefill       （leader-worker 模式，2 实例 × size 4 = 8 pod）
 └─ Role: decode        （leader-worker 模式，4 实例 × size 2 = 8 pod）
       │  每个角色由一个专属 workload 管理 ↓
       RoleInstanceSet（per-role 有状态负载，v1alpha2 原生 workload）
         │  每个副本一个 ↓
       RoleInstance（pod 组，生命周期强绑定；原地更新的作用单元）
         │  组内若干组件 ↓
       Pods（component：leader-0 / worker-0 / worker-1 …）
```

| 概念 | 是什么 | 关键点 |
|---|---|---|
| **Role（角色）** | 基本调度与 rollout 单位 | 每个 role 有独立的副本数、部署模式、rollout 策略、重启策略、依赖列表（[multiroles.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/multiroles.md)） |
| **RoleBasedGroup** | 一个逻辑服务 = 一组角色 | 一个 YAML 声明整条推理链路 |
| **RoleInstanceSet** | 每角色一个的有状态负载 | v1alpha2 原生 workload，取代 v1alpha1 时代挂接的 Deployment/STS/LWS |
| **RoleInstance（实例）** | 一组生命周期强绑定的 pod | 实例整体升级、整体就绪判定（`AllPodReady`）；支持原地更新（[instance.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/instance.md)） |
| **CoordinatedPolicy** | 独立 CRD，管跨角色协同 | 滚动更新/扩缩时角色间 `maxSkew`、`maxUnavailable`、进度推进策略 |

围绕这条主干还有几个配角对象：每角色一个 headless Service（服务发现）、ControllerRevision（版本快照）、PodGroup（gang 调度）、RoleBasedGroupScalingAdapter（给 HPA 挂的伸缩适配器）、ClusterEngineRuntimeProfile（集群级 sidecar/init 注入，如 GPU driver、监控 agent）。

## 3. 三种部署模式：一个 role 怎么组织自己的 pod

（[patterns.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/patterns.md)）

| 模式 | 每实例 pod 组织 | 适用场景 |
|---|---|---|
| `standalonePattern` | 单 pod | 单卡服务、无并行度的角色（router 就是典型） |
| `leaderWorkerPattern` | 1 leader + N−1 worker | 张量并行（TP）多卡推理：leader 起主进程，worker 挂 GPU 协同 |
| `customComponentsPattern` | 任意组件组合 | 一个实例里塞异构 pod 组（比如引擎 + 专属 sidecar 分开声明） |

`leaderWorkerPattern` 专为大模型多卡推理设计：`size: 4` 即 1 leader + 3 worker，配合注入的 `RBG_LWP_*` 环境变量（见下节），torchrun 式的 `--dist-init-addr / --nnodes / --node-rank` 参数不用人算。

## 4. 内建服务发现：三件套，免手写 DNS

这是 RBG 最省心的部分。传统多角色部署里，"router 怎么知道 prefill 实例的地址" 要靠 StatefulSet 命名约定 + 手写 headless service + 环境变量模板。RBG 把它变成控制器义务：

**① 每角色自动创建 headless Service。** 服务名 `s-<rbg名>-<角色名>`——`s-` 前缀是为了满足 DNS-1035「服务名不能以数字开头」的约束（[helper.go:106-119](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/helper.go#L106-L119)）。于是 pod 的稳定 DNS 是：

```
<rbg名>-<角色名>-<序号>.s-<rbg名>-<角色名>.<namespace>.svc.cluster.local
```

**② 环境变量自动注入。** 控制器往每个 pod 注入一整套 `RBG_*` 变量（[constants/env.go](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/constants/env.go)）：

| 变量 | 含义 |
|---|---|
| `RBG_GROUP_NAME` / `RBG_ROLE_NAME` | 所在组和角色 |
| `RBG_ROLE_INDEX` | 在角色内的序号 |
| `RBG_ROLE_INSTANCE_NAME` / `RBG_COMPONENT_NAME` / `RBG_COMPONENT_INDEX` | 实例与组件身份（v1alpha2） |
| `RBG_LWP_LEADER_ADDRESS` | leader-worker 模式下 leader 的地址 |
| `RBG_LWP_GROUP_SIZE` / `RBG_LWP_WORKER_INDEX` | 实例内组件总数 / 自己的序号 |

leader-worker 模式下，worker 拿着 `RBG_LWP_LEADER_ADDRESS` + `RBG_LWP_GROUP_SIZE` + `RBG_LWP_WORKER_INDEX` 三个变量就能自动组网（[constants/env.go:60-67](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/constants/env.go#L60-L67)）——这正是 SGLang 多机启动需要的三个参数。

**③ 组件发现注解 + 端口分配器。** `customComponentsPattern` 下还可以在组件模板上打 `rolebasedgroup.workloads.x-k8s.io/component-discovery` 注解，声明"把同实例某组件的 FQDN / 分配的端口注入为环境变量"（[component_discovery.go:34-40](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/component-discovery/component_discovery.go#L34-L40)）；配套的 `rolebasedgroup.workloads.x-k8s.io/port-allocator` 注解负责给组件分配端口（pod 级或 role 级共享，[port-allocator](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/port-allocator/manager.go)）。Pod 创建时注入、注入完删注解——纯控制器指令，不污染运行时对象。

> 官方 README 把这组能力概括为"topology self-aware service discovery"：容器里不需要硬编码任何对端地址。

## 5. 启动顺序：角色依赖拓扑排序

角色声明 `dependencies: [...]`，控制器按依赖做拓扑排序、分层推进（[dependency.go:43](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/dependency/dependency.go#L43) 的 `SortRoles`、L129 的 `dependencyOrder`）：

```yaml
roles:
  - name: frontend
    replicas: 1
  - name: backend
    replicas: 3
    dependencies: ["frontend"]   # frontend Ready 之后才创建 backend
```

reconcile 主循环里的处理（[rolebasedgroup_controller.go:475-522](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/internal/controller/workloads/rolebasedgroup_controller.go#L475-L522)）：每层角色先 `CheckDependencyReady`，不 ready 直接返回等下一轮——**后续角色根本不会被创建**，而不是创建出来挂着 CrashLoop。

## 6. 跨角色协同：CoordinatedPolicy

独立 CRD，管两件事（[coordinated-policy.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/coordinated-policy.md)）：

- **协同滚动更新**：`maxSkew: "1%"`——prefill 和 decode 的更新进度差被钳制在 1% 以内，两边锁步推进；`maxUnavailable` 控制更新期间不可用量
- **协同扩缩**：角色按配对比例一起扩，进度推进策略可选 `OrderScheduled`（pod 调度成功即算推进，快但风险高）等档位

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: CoordinatedPolicy
metadata:
  name: pd-rollout-policy
spec:
  policies:
    - name: prefill-decode-rollout
      roles: [prefill, decode]
      strategy:
        rollingUpdate:
          maxSkew: "1%"
          maxUnavailable: "10%"
```

这解决了 PD 分离运维里最经典的坑：升级镜像时 prefill 和 decode 版本漂移导致 KV cache 传输协议不兼容。

## 7. 原地更新（In-place Update）：有状态负载升级不换"壳"

RoleInstance 支持三种更新策略（[rolebasedgroup_types.go:108-116](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/rolebasedgroup_types.go#L108-L116)）：

| 策略 | 行为 |
|---|---|
| `RecreatePod` | 重建 pod（传统行为，pod 名变、重新调度） |
| `InPlaceIfPossible` | 能原地就原地（改镜像 → 只重启容器，pod 不重建、不重新调度），不行再重建 |
| `InPlaceOnly` | 只允许原地 |

原地更新的价值：**pod 身份（名字、IP、节点亲和、GPU 绑定）不变，只重启容器进程**。对推理集群这很关键——重建 pod 意味着重新调度、可能换节点、拓扑全变。[instance.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/instance.md) 里有个直观演示：改镜像后 pod 的 RESTARTS +1 而 AGE 连续。

## 8. 性能两板斧：gang 调度 + 独占拓扑

**Gang 调度**（[gang-scheduling.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/gang-scheduling.md)）：注解 `rbg.workloads.x-k8s.io/group-gang-scheduling: "true"` 打在 RoleBasedGroup 上，控制器创建 `PodGroup`（scheduler-plugins）或 Volcano podgroup，`minMember` = **全组所有角色的 pod 总数**——要么全调度成功，要么全等。这对多卡 TP 推理是刚需：8 卡组只调度上 7 卡就是纯浪费 + 死锁风险。支持 scheduler-plugins（原生路线）和 Volcano（队列 + 优先级）两种实现。

**独占拓扑**（[exclusive-topology.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/exclusive-topology.md)）：注解 `rbg.workloads.x-k8s.io/group-exclusive-topology: kubernetes.io/hostname` 声明拓扑域，同组 pod 通过 `group-uid` / `group-unique-hash` 标签做 pod 亲和，全部落在同一节点 / 机架 / 可用区。PD 分离里 prefill→decode 的 KV cache 传输走 RDMA 时，共置带来的带宽收益直接决定架构是否划算。

## 9. 控制器工作流：reconcile 链条走读

主控制器 `RoleBasedGroupReconciler`（[rolebasedgroup_controller.go:150](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/internal/controller/workloads/rolebasedgroup_controller.go#L150)）每轮做的事：

```
Reconcile(RoleBasedGroup)
  ├─ preCheck（L298）                    角色名/依赖合法性
  ├─ handleRevisions（L253）              ControllerRevision 快照
  ├─ reconcilePodGroup（L465）            gang 调度 PodGroup
  └─ reconcileRoles（L475）
       ├─ dependencyManager.SortRoles     依赖拓扑排序 → 分层
       └─ 逐层：
            ├─ CheckDependencyReady（L499）   依赖不 ready → 本轮到此为止
            └─ reconcileSingleRole（L524）
                 ├─ workload reconciler        v1alpha2 → RoleInstanceSet reconciler
                 └─ ReconcileScalingAdapter    每角色伸缩适配器（HPA 挂钩）
```

往下两级自治：RoleInstanceSet reconciler 管每角色的实例集合（扩缩、分区、滚动），RoleInstance controller 管单实例的 pod 组（含原地更新的 node binding 同步）。服务发现注入发生在 pod 创建时（[discovery/injector.go](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/discovery/injector.go)）。

## 10. 实战走读：SGLang PD 分离（[pd-disagg-leader-worker.yaml](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/pd-disagg-leader-worker.yaml)）

一个 YAML、三个角色、一套拓扑：

| 角色 | 模式 | 拓扑 | 说明 |
|---|---|---|---|
| `router` | standalone × 1 | 1 pod | SGLang Router，入口路由 + PD 分发 |
| `prefill` | leader-worker × 2 实例 | 每实例 1+3 = 4 pod（TP4） | 预填充 |
| `decode` | leader-worker × 4 实例 | 每实例 1+1 = 2 pod（TP2） | 解码 |

关键细节全部"自动接线"：

- **router 寻址 prefill**：`--prefill "http://pd-disagg-lws-prefill-0.s-pd-disagg-lws-prefill:8000"`——第 0 号 prefill 实例的 leader，headless service DNS，不需要任何手工 Service
- **多机组网**：SGLang 启动参数直接用注入变量：`--dist-init-addr $(RBG_LWP_LEADER_ADDRESS):6379 --nnodes $(RBG_LWP_GROUP_SIZE) --node-rank $(RBG_LWP_WORKER_INDEX)`
- **每角色独立 rollout**：`InPlaceIfPossible` + `maxUnavailable: 1`
- **`restartPolicyConfig: type: None`**：实例内组件生命周期由 RBG 管理（leader 挂了整组联动处理），不用 K8s 默认重启策略

同一个 `examples/inference/` 目录还有聚合推理（不分离）、单卡版本（standalone），以及 [Dynamo](https://github.com/sgl-project/rbg/tree/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/ecosystem/dynamo)（NVIDIA 推理编排栈）、[Mooncake](https://github.com/sgl-project/rbg/tree/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/ecosystem/mooncake)（KV cache 传输/复用引擎，SGLang 与 vLLM 例子都有）的集成样例。

## 11. 生态位与周边

| 组件 | 关系 |
|---|---|
| **SGLang** | 同门项目；RBG 的头号场景就是 SGLang PD 分离部署 |
| **LeaderWorkerSet (LWS)** | RBG 致谢并复用其代码；v1alpha2 起用自己的 RoleInstanceSet 取代 LWS 依赖（v0.7.0 不再需要 LWS CRD） |
| **NVIDIA Dynamo** | 数据中心级推理编排栈，RBG 提供 Dynamo 运行时集成样例 |
| **Mooncake** | KV cache 传输/复用引擎，PD 分离的 KV 搬运层 |
| **[rbg-planner](https://github.com/rolebasedgroup/rbg-planner)** | SLA 驱动 autoscaler：ARIMA 负载预测 + TTFT/ITL 目标，按角色扩缩 |
| **[inference-engine-runtime](https://github.com/rolebasedgroup/inference-engine-runtime)** | Python sidecar：LoRA 管理、统一 Prometheus 指标、拓扑管理 |
| **[inference-ext-cli](https://github.com/rolebasedgroup/inference-ext-cli)** | `llmctl`：服务/模型管理、benchmark 编排、参数自动搜索（Optuna） |
| **kubectl-rbg** | 官方 CLI，`make build-cli` 构建，管理 RBG 资源与 LLM 部署 |

和原生原语的关系一句话：**Deployment/StatefulSet 是"单角色"原语，LWS 是"单角色多卡"原语，RBG 是"多角色 × 多卡 × 跨角色协同"原语**——每往上一层，多管一个维度。

## 12. 版本演进与迁移注意

| 版本 | 时间 | 亮点 |
|---|---|---|
| v0.4.0 | 2025-09 | RBGS 伸缩、Volcano podgroup |
| v0.5.0 | 2025-12 | 原生 InstanceSet、原地更新、Mooncake 集成 |
| v0.6.0 | 2026-02 | 协同扩缩、有状态 InstanceSet |
| v0.7.0 | 2026-06 | **v1alpha2 API stable**、转换 webhook、CLI 多节点 LLM serving、端口分配器、CoordinatedPolicy、gang 调度 |

> 来源：[README 发布表](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/README.md)；v0.7.0 起不再要求安装 LWS CRD，K8s ≥ v1.22。

⚠ **迁移注意**（[README「Deprecated workload types」](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/README.md)）：v1alpha1 及其 `Deployment`/`StatefulSet`/`LeaderWorkerSet` workload 类型已被 v1alpha2 + `RoleInstanceSet` 取代。chart 值 `controller.deprecatedWorkloadTypes.enabled`：

- `true`（默认）：两代控制逻辑都活着，所有 workload 类型可用——**存量部署必须保持 true**
- `false`：彻底拒绝废弃类型（不授 RBAC、不 watch、webhook 直接拒绝创建）——**只用于全新安装**。已在跑的集群没有官方支持的关闭路径，对象会变成无人 reconcile；迁移指南官方说后续版本提供

## 13. 什么时候用 / 不用

**用 RBG 的信号**：部署的是多角色推理拓扑（PD 分离、router+engine、engine+KV 传输引擎）；TP 多卡需要 leader-worker 组管理；需要跨角色协同升级/扩缩；需要 gang 调度和拓扑共置；栈在 SGLang / Dynamo / Mooncake 生态里。

**不需要 RBG 的信号**：单体推理服务（一个 Deployment + 一个 Service 完事）；没有跨角色协同诉求的多服务（普通 K8s 编排足够）；还没有 K8s 运维能力。

一句话收束：**RBG 把"部署一个 PD 分离推理服务"从"N 个 YAML + Helm hook + 人肉 DNS 约定"压缩成"一个 CRD + 一组注解"，代价是接受一个新的 workload 抽象层。** 对多角色推理这个特定领域，这笔交易目前看是划算的——尤其是 SGLang 生态用户，官方例子和周边工具（planner / runtime / CLI）都是现成的。

## 参考资料

| 主题 | 位置 |
|---|---|
| 仓库与文档站 | [github.com/sgl-project/rbg](https://github.com/sgl-project/rbg) · [rolebasedgroup.github.io](https://rolebasedgroup.github.io) |
| 对象层级 / API | [doc/reference/api.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/api.md) |
| 标签 / 注解 / 环境变量全表 | [doc/reference/variables.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/variables.md) |
| 部署模式 | [doc/features/patterns.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/patterns.md) |
| 角色依赖 | [doc/features/role-dependencies.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/role-dependencies.md) |
| 协同策略 | [doc/features/coordinated-policy.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/coordinated-policy.md) |
| 原地更新 / Instance | [doc/features/instance.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/instance.md) |
| PD 分离完整例子 | [examples/inference/pd-disagg-leader-worker.yaml](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/pd-disagg-leader-worker.yaml) |
| 服务名 `s-` 前缀规则 | [api/workloads/v1alpha2/helper.go:106-119](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/helper.go#L106-L119) |
| 依赖拓扑排序 | [pkg/dependency/dependency.go:43-79](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/dependency/dependency.go#L43-L79) |
| reconcile 主链 | [internal/controller/workloads/rolebasedgroup_controller.go:475-580](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/internal/controller/workloads/rolebasedgroup_controller.go#L475-L580) |
| 环境变量注入表 | [api/workloads/constants/env.go](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/constants/env.go) |

> 本文全部结论来自上述源码与文档（静态阅读，未上机部署）。涉及行为细节（如调度时序、性能收益）的判断以官方文档为准；标 ⚠ 处为需实测复核项。
