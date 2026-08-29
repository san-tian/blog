# RBG（RoleBasedGroup）：在 K8s 上部署大模型推理服务的"总管"

> 一句话：**RBG 是 [SGLang 项目](https://github.com/sgl-project/rbg)开源的 Kubernetes Operator（Go，Apache-2.0），把 PD 分离这类多角色推理服务——router / prefill / decode——当成一个对象来部署、扩缩、升级和寻址**。角色间的启动顺序、协同升级、配对扩缩、服务发现、gang 调度，全部从"运维技能"变成"几行 YAML"。
>
> 📄 本文是详细版；幻灯片版见 [rbg-rolebasedgroup-explained.html](rbg-rolebasedgroup-explained.html)（由浅入深五层：干嘛的 → 效果 → 功能 → 部署 → 设计）。
>
> 📌 讲解对象：[sgl-project/rbg](https://github.com/sgl-project/rbg) @ commit [`4acd5a7`](https://github.com/sgl-project/rbg/commit/4acd5a791e9709f34c4bd5da569e6427896d21a5)（2026-08-29，v0.7.0 已发布）。文中所有源码引用都锚定该 SHA。官方文档站：[rolebasedgroup.github.io](https://rolebasedgroup.github.io)。

---

## 1. 它是用来做什么的？

**管"一个服务里的多个角色"。** 先看官方 quick start 的最小例子——backend 要等 frontend 就绪再启动：

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: RoleBasedGroup
metadata:
  name: nginx-cluster
spec:
  roles:
    - name: frontend
      replicas: 1
      standalonePattern:
        template: { spec: { containers: [{ name: nginx, image: nginx:1.14.1 }] } }
    - name: backend
      replicas: 3
      dependencies: ["frontend"]   # frontend Ready 之前，backend 根本不会被创建
      standalonePattern:
        template: { spec: { containers: [{ name: nginx, image: nginx:1.14.1 }] } }
```

原生 K8s 写法是 frontend、backend 各一个 Deployment，顺序靠 init container 探测或手动分两次 apply——**角色之间的"关系"没有原生语义**。RBG 把关系本身变成 API。

> 补一句底子：K8s 即 Kubernetes，是「声明 YAML、集群持续拉起维持」的容器编排系统；Operator 是用自定义控制器扩展 K8s 的模式；CRD（自定义资源定义）是往 K8s 注册新对象类型的机制——下面这个 `RoleBasedGroup` 就是 RBG 注册的新对象类型。

**头号场景是 PD 分离的 LLM 推理。** prefill（预填充：推理第一步，把整段 prompt 一次算完、产出 KV cache，决定首字延迟）和 decode（解码：逐 token 生成，决定吐字速度）混布会互相干扰计算、耦合资源配比（[官方 quick start](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/quick_start.md)），分离后各自独立扩缩、独立配卡，代价是部署形态变成多角色 × 多卡 × 异构配比：router（单 pod）→ prefill（TP4：1 leader + 3 worker × 2 实例）→ decode（TP2 × 4 实例；TP 即张量并行——一个模型多卡协同、1 leader + N worker 组成一个推理实例），再加 KV 传输引擎（Mooncake）。这正是 RBG 设计的目标形态——**一个 YAML 声明整条链路**（[官方 PD 分离例子](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/pd-disagg-leader-worker.yaml)）。PD 分离基础详见 [pd-decode-kvcache-offload.md](pd-decode-kvcache-offload.md)。

### 1.1 定位：它和 SMG / Dynamo 是什么关系？

三者不在同一层，不是竞争关系——**RBG 是「管部署的」，SMG / Dynamo 是「干活的程序」**。用开餐厅类比：

| 餐厅角色 | 对应谁 | 干什么 |
|---|---|---|
| 厨师 | 推理引擎（SGLang / vLLM） | 把请求算成结果 |
| 领位员 | SMG（sglang-router，SGLang Model Gateway） | 把请求路由到某个 prefill/decode 实例 |
| 连锁运营标准 | Dynamo | 数据中心级推理编排栈：路由、PD 分离、KV 路由（跨 vLLM/SGLang） |
| 店长 + 施工队 | RBG | 把「几个厨师、几个领位员」编排起来：谁先谁后、扩缩、升级、寻址 |

分层结构：

```
RBG（K8s 部署编排层：怎么把程序跑起来、扩缩、升级、寻址）
  └─ 部署的对象里，可能包含：
       SMG    —— 一个路由程序（sglang-router），当「router」角色
       Dynamo —— 一个推理编排运行时（RBG 官方有「用 RBG 部署 Dynamo」样例）
       SGLang/vLLM 引擎 —— prefill / decode 角色
```

SMG 是 SGLang 自家轻量路由组件，只做「把请求分给哪个 prefill/decode」这一件事；Dynamo 是 NVIDIA 的数据中心级编排栈，范围大得多。RBG 两者都能部署：例子里 router 角色用 SMG，`examples/inference/ecosystem/dynamo/` 是 Dynamo 样例。

**直接部署一个 Gateway vs 用 RBG 部署 Gateway：**

| | 直接部署 Gateway | RBG 部署 Gateway |
|---|---|---|
| 得到什么 | 一个孤零零的 router pod | 整条链路（router + prefill + decode） |
| 后端 | 后厨自己另起一堆 Deployment/StatefulSet | 同一个 YAML 声明 |
| 寻址 | `--prefill http://…` 自己填死 | RBG 的 headless Service 自动生成地址 |
| 顺序/扩缩/升级 | 手动协调 | 依赖、协同、锁步升级都是整体动作 |

官方例子 [pd-disagg-leader-worker.yaml](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/pd-disagg-leader-worker.yaml) 的 router 就是 `lmsysorg/sglang-router`（即 SMG），命令里的 `--prefill "http://pd-disagg-lws-prefill-0.s-pd-disagg-lws-prefill:8000"` 域名由 RBG 命名规律自动生成，不用手建 Service。

一句话：**RBG 不替你做 Gateway，它替你「把 Gateway 和它背后那一大家子」一起部署好。** 只想单点跑 router 试验，直接一个 Deployment 更轻；要挂真 prefill/decode、要扩缩升级，RBG 的价值就出来了。

## 2. 它有什么效果？

同样的活，人做 vs 声明（每行机制都有源码/文档出处，见第 3 节和参考资料）：

| 要做的事 | 不用 RBG | 用 RBG |
|---|---|---|
| 启动顺序 | init container / Helm hook / 手动分次 apply | `dependencies: ["frontend"]` 一行 |
| 对端地址 | 手写 N 个 headless Service + ConfigMap 硬编码 | 自动建 Service，DNS 规律固定 + 环境变量注入 |
| 多卡组网 | 手算 node-rank / nnodes / dist-addr 塞进容器 | 容器里直接读 `RBG_LWP_*` 注入变量 |
| 升级不漂移 | 两个对象各自滚，中间版本不一致 | `maxSkew: "1%"` 两角色锁步推进 |
| 多卡原子调度 | 无现成语义，起一半卡白占资源 | 一个注解开 gang：全成或全不成 |
| 拓扑共置 | 手写 pod affinity / nodeSelector | 一个注解声明拓扑域，同组共置 |
| 整体拓扑 | router/prefill/decode + Service + ConfigMap ≥ 5 类对象 | 一个 YAML、一个对象，删除即全删 |

> ⚠ 本表是**机制对比**（源码/文档可查证），不是 benchmark——性能与稳定性收益需按场景实测。

一句话：**部署一个 PD 分离服务的门槛，从「K8s 高级运维」降到「会写一个 YAML」**。

## 3. 它能实现什么功能？

**① 多角色混搭**：一个 YAML 里 router / prefill / decode 各自独立副本数、独立 rollout 策略（[multiroles.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/multiroles.md)）。

**② 三种部署模式**（[patterns.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/patterns.md)）：`standalonePattern`（每实例 1 pod，router 类）、`leaderWorkerPattern`（1 leader + N−1 worker，TP 多卡推理；`size: 4` 即 1+3）、`customComponentsPattern`（异构 pod 组）。同一 RBG 里不同角色可用不同模式。

**③ 启动依赖**：`dependencies` 做拓扑排序分层推进，依赖不 Ready 下一层不创建（源码见第 5 节）。

**④ 服务发现三件套**（最省心的部分）：
- 每角色自动建 headless Service（不分配虚拟 IP 的 Service，DNS 直达每个 pod），名字 `s-<rbg名>-<角色名>`——`s-` 前缀是 DNS-1035「服务名不能数字开头」的要求（[helper.go:106-119](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/helper.go#L106-L119)）；pod DNS 规律固定：`<rbg>-<role>-<序号>.s-<rbg>-<role>`
- 环境变量自动注入（[constants/env.go](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/constants/env.go)）：`RBG_GROUP_NAME` / `RBG_ROLE_NAME` / `RBG_ROLE_INDEX`，leader-worker 模式再加 `RBG_LWP_LEADER_ADDRESS` / `RBG_LWP_GROUP_SIZE` / `RBG_LWP_WORKER_INDEX`——SGLang 多机组网三个参数（`--dist-init-addr/--nnodes/--node-rank`）直接用变量，零手算
- `customComponentsPattern` 配套组件发现注解（把同实例某组件的 FQDN / 端口注入为环境变量，[component_discovery.go:34-40](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/component-discovery/component_discovery.go#L34-L40)）和端口分配器注解

**⑤ 协同升级 / 扩缩**：独立 CRD `CoordinatedPolicy`（[coordinated-policy.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/coordinated-policy.md)）管跨角色：滚动更新 `maxSkew: "1%"` 钳制进度差（prefill/decode 锁步，防 KV 传输协议不兼容的中间态）+ `maxUnavailable`；协同扩缩按配对比例推进（`OrderScheduled` 等档位）。

**⑥ 原地更新**（[instance.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/instance.md)）：`RecreatePod` / `InPlaceIfPossible`（能原地就原地：改镜像只重启容器，pod 名字 / IP / 节点 / GPU 绑定不变）/ `InPlaceOnly`（[rolebasedgroup_types.go:108-116](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/rolebasedgroup_types.go#L108-L116)）。作用单元是实例（RoleInstance，pod 组整体升级）。

**⑦ gang 调度**（一组 pod 要么全部调度成功、要么全部不调度；[gang-scheduling.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/gang-scheduling.md)）：注解 `rbg.workloads.x-k8s.io/group-gang-scheduling: "true"`，控制器建 PodGroup（scheduler-plugins）或 Volcano podgroup，`minMember` = 全组所有角色 pod 总数——8 卡 TP 组只调度上 7 卡的场景被根治。

**⑧ 独占拓扑**（[exclusive-topology.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/features/exclusive-topology.md)）：注解 `rbg.workloads.x-k8s.io/group-exclusive-topology: kubernetes.io/hostname` 声明拓扑域，同组 pod 通过 `group-uid` / `group-unique-hash` 标签共置同节点 / 机架 / 可用区——KV 传输走 RDMA 时共置带宽直接决定架构划算与否。

**⑨ 每角色独立伸缩**：`scalingAdapter` 开启后控制器自动建 RoleBasedGroupScalingAdapter，HPA 直接挂它（官方 PD 例子即此用法）。

**⑩ 生态集成**：SGLang（同门，头号场景）、vLLM（Mooncake 例子有 vLLM 版）、NVIDIA Dynamo、Mooncake KV 传输/复用（[ecosystem/](https://github.com/sgl-project/rbg/tree/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/ecosystem)）；周边有 rbg-planner（SLA 驱动 autoscaler：ARIMA 负载预测 + TTFT/ITL 目标）、inference-engine-runtime（Python sidecar：LoRA 管理、统一 Prometheus、拓扑管理）、llmctl / kubectl-rbg（CLI）、rbg-agent-guide（给 AI 编码助手的部署 skill）。

## 4. 怎么部署？

**① 装 operator**（[install.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/install.md)）：

```bash
# 方式 A：helm（含 CRD 自动升级 Job）
helm upgrade --install rbgs deploy/helm/rbgs --create-namespace --namespace rbgs-system --wait
# 方式 B：纯 kubectl
kubectl apply --server-side -f ./deploy/kubectl/manifests.yaml
kubectl wait deploy/rbgs-controller-manager -n rbgs-system --for=condition=available --timeout=5m
```

前置：控制器 1 核 1G 即可跑；K8s 版本 README 版本表写 v0.7.0 需 ≥ v1.22，install.md 仍写 ≥ 1.28（⚠ 两处不一致，保守按 1.28+ 准备）。

**② 跑最小例子**：`kubectl apply` 第 1 节的 nginx YAML（或 [examples/basic/rbg/dependency/](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/basic/rbg/dependency/role-dependencies.yaml)）。

**③ 看对象**（标签 selector 见 [variables.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/variables.md)）：

```bash
kubectl get rbg nginx-cluster                                        # 本体
kubectl get roleinstanceset -l rbg.workloads.x-k8s.io/group-name=nginx-cluster   # 每角色 workload
kubectl get pods -l rbg.workloads.x-k8s.io/group-name=nginx-cluster              # 全组 pod
```

**④ 上真模型**：GPU 节点 + `lmsysorg/sglang:v0.5.9+` 镜像，`kubectl apply -f examples/inference/pd-disagg-leader-worker.yaml`——router / prefill(TP4) / decode(TP2) 全套起来，寻址与组网全自动（见第 3 节 ④）。CLI：`make build-cli` 构建 kubectl-rbg 管理资源与 LLM 部署（v0.7.0 起支持多节点 LLM serving）。

> ⚠ 本节命令摘自官方文档，未实测；RBG 官方例子索引见 [doc/quick_start.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/quick_start.md)。

## 5. 它采用了什么设计、什么架构？

**四层对象**（[api.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/api.md)）：

```
RoleBasedGroup（一个推理服务）
 ├─ Role: router / prefill / decode（各自模式、副本数、策略、依赖）
 │    ↓ 每角色一个
 │  RoleInstanceSet（有状态 workload：扩缩/分区/滚动）
 │    ↓ 每副本一个
 │  RoleInstance（pod 组，生命周期强绑定；原地更新单元）
 │    ↓
 │  Pods（组件：leader-0 / worker-0 / worker-1 …）
 + 配角：每角色 headless Service、ControllerRevision、PodGroup（gang）、
         ScalingAdapter（HPA）、CoordinatedPolicy（独立 CRD）
```

分层的意义：组管依赖/协同、角色管实例集合、实例管 pod 组——跨角色的事只在最上层做，每个 workload 不需要懂"别的角色"。

行文名词约定：**角色**= 服务成员（router / prefill / decode）；**实例**= 角色的一副本（可能是一组 pod）；**组件**= 实例内的单个 pod（leader / worker）。

**控制器 reconcile 链**（[rolebasedgroup_controller.go:475-580](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/internal/controller/workloads/rolebasedgroup_controller.go#L475-L580)）：

```
Reconcile(RoleBasedGroup)
 ├─ preCheck（L298）                角色名/依赖合法性
 ├─ handleRevisions（L253）         ControllerRevision 版本快照
 ├─ reconcilePodGroup（L465）        gang 调度 PodGroup
 └─ reconcileRoles（L475）
      ├─ SortRoles（dependency.go:43） 依赖拓扑排序 → 分层
      └─ 逐层：CheckDependencyReady（L499）→ 不 ready 本轮 return，下层不创建
           └─ reconcileSingleRole → RoleInstanceSet → RoleInstance → Pod
                （pod 创建时注入 RBG_* 变量 + 建每角色 headless Service + ScalingAdapter）
```

**技术底座**：Go + kubebuilder / controller-runtime；v1alpha2 起 RoleInstanceSet 取代 v1alpha1 挂接的 Deployment/STS/LWS——复用并致谢 [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) 代码，v0.7.0 起不再需要安装 LWS CRD。

**版本演进**：v0.4（2025-09）RBGS 伸缩、Volcano podgroup → v0.5（2025-12）原生 InstanceSet、原地更新、Mooncake → v0.6（2026-02）协同扩缩、有状态 InstanceSet → v0.7（2026-06）**v1alpha2 stable**、转换 webhook、CLI 多节点、端口分配器、CoordinatedPolicy、gang 调度（[README 发布表](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/README.md)）。

⚠ **迁移注意**：v1alpha1 的 Deployment/STS/LWS workload 类型已被取代。开关 `deprecatedWorkloadTypes.enabled`：`true`（默认，两代共存，**存量部署必须保持**）/ `false`（彻底拒绝废弃类型，**仅限全新安装**——已在跑的集群没有官方支持的关闭路径，对象会无人 reconcile）；v1alpha1 → RoleInstanceSet 迁移指南官方承诺后续版本提供。

## 6. 什么时候用 / 不用？

**用**：多角色推理拓扑（PD 分离、router+engine、engine+KV 传输引擎）；TP 多卡组管理；需要跨角色协同升级或配对扩缩；需要 gang 调度和拓扑共置；栈在 SGLang / Dynamo / Mooncake 生态（官方例子和周边工具现成）。

**不用**：单体推理服务（一个 Deployment + 一个 Service 完事）；没有跨角色协同诉求的多服务；没有 K8s 运维能力；v1alpha1 存量且等不及迁移指南。

一句话收束：**RBG 把"部署一个 PD 分离服务"从「N 个 YAML + Helm hook + 人肉 DNS 约定」压缩成「一个 CRD + 一组注解」，代价是接受一个新的 workload 抽象层**——对多角色推理这个特定领域，这笔交易目前看是划算的。

## 参考资料

| 主题 | 位置 |
|---|---|
| 仓库与文档站 | [github.com/sgl-project/rbg](https://github.com/sgl-project/rbg) · [rolebasedgroup.github.io](https://rolebasedgroup.github.io) |
| 安装 / 快速上手 | [doc/install.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/install.md) · [doc/quick_start.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/quick_start.md) |
| API / 对象 / 标签全表 | [doc/reference/api.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/api.md) · [doc/reference/variables.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/reference/variables.md) |
| 特性文档（模式/依赖/协同/gang/拓扑/原地更新/HPA） | [doc/ 目录，TOC 见 doc/TOC.md](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/doc/TOC.md) |
| PD 分离完整例子 | [examples/inference/pd-disagg-leader-worker.yaml](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/examples/inference/pd-disagg-leader-worker.yaml) |
| 服务名 `s-` 前缀规则 | [api/workloads/v1alpha2/helper.go:106-119](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/v1alpha2/helper.go#L106-L119) |
| 依赖拓扑排序 | [pkg/dependency/dependency.go:43-79](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/pkg/dependency/dependency.go#L43-L79) |
| reconcile 主链 | [internal/controller/workloads/rolebasedgroup_controller.go:475-580](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/internal/controller/workloads/rolebasedgroup_controller.go#L475-L580) |
| 环境变量注入表 | [api/workloads/constants/env.go](https://github.com/sgl-project/rbg/blob/4acd5a791e9709f34c4bd5da569e6427896d21a5/api/workloads/constants/env.go) |

> 本文全部结论来自上述源码与文档（静态阅读，未上机部署）；涉及行为细节（如调度时序、性能收益）的判断以官方文档为准；标 ⚠ 处为需实测复核项。
