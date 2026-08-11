# PD 分离架构里,decode 到底要不要开二级 KV 缓存?——一次被源码和硬件双重纠正的技术复盘

> 这是一个真实发生的技术讨论复盘。最初的一个直觉问题,在源码核查和硬件事实的双重纠正下,推翻了好几次结论。最终答案不是"该开"或"不该开",而是"为什么这套部署没开、以及什么条件下开才划算"。记录下来,因为每一步纠正都踩在一个真实的设计取舍上。

## 缘起:一个看似简单的问题

我们的 PD 分离部署(MI300X ×8 + 8× HDR InfiniBand,GLM-5.2 FP8,SGLang v0.5.14 + CjiW fork)里,prefill 实例开了 `--enable-hierarchical-cache`(HiCache),decode 实例只开了 `--disaggregation-decode-enable-radix-cache`,没开任何二级缓存。

最初的疑问是:**为什么 decode 不用二级缓存,来减少从 prefill 的 KV 传输?**

直觉上,如果 decode 本地缓存了前缀 KV,下次命中就不用从 prefill 传了,应该更快。这个直觉对吗?

**结论先行(最终版,带诚实标注)**:

- 机制层面:SGLang 确实有 decode 专用的二级缓存机制(`DecodeKVCacheOffloadManager`),支持把 decode 增量 KV offload 到 host DRAM(L2)再落 NVMe(L3),并能在下个请求命中时通过专用流 layer-wise overlap 加载回来。源码完整存在。
- 带宽层面:这套部署有 **8 张 IB 网卡**,开启 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 后 8 个 CP rank 并行传输,路径 A(跨机 RDMA)聚合带宽(~200 GB/s)远高于路径 B(本机 H2D,~50-64 GB/s)。在 8×IB 下路径 A 已经很快,decode 二级缓存的边际收益可能不足以抵消开销。
- 未实测部分:具体在什么消息大小下路径 A/B 谁快,需要实测确认,不能仅凭源码断言。

下面是完整推导过程。

---

## 一、先把 cache 结构搞清楚(纠正第一个误解)

### 误解:HiCache 的二级是 NVMe 硬盘

部署脚本里 prefill 的配置:

```bash
--enable-hierarchical-cache --hicache-ratio 1
--hicache-write-policy write_back --hicache-mem-layout page_first
--hicache-storage-backend file --file-storage-path /root/hicache  # B300 脚本
```

加上环境变量:

```bash
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="200G"
```

看起来二级介质就是 NVMe 文件。但这只是**这个部署的选择**,不是 HiCache 框架的全貌。

### 源码确认:HiCache 是三层结构,L2 默认是 host DRAM

看 `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` 的初始化:

```python
# L2:host DRAM pool(MLA 用 MLATokenToKVPoolHost)
self.decode_host_mem_pool = MLATokenToKVPoolHost(
    kv_cache,
    server_args.hicache_ratio,        # L2 大小 = ratio × L1 GPU pool
    server_args.hicache_size,
    self.page_size,
    server_args.hicache_mem_layout,
)
# L3:可选的 storage backend
self.cache_controller = HiCacheController(
    mem_pool_host=self.decode_host_mem_pool,              # ← L2 = host DRAM
    storage_backend=server_args.hicache_storage_backend,  # ← L3 = file/nixl/mooncake...
)
```

结构是**三层**,不是两层:

| 层级 | 介质 | 控制参数 | 说明 |
|---|---|---|---|
| **L1** | GPU HBM | `--mem-fraction-static` | 显存 KV pool,活跃请求的 KV 常驻这里 |
| **L2** | host DRAM(CPU 内存) | `--hicache-ratio`(默认 2.0,即 L2 = 2× L1) | **默认就有,不依赖 storage backend**;page-locked pinned memory |
| **L3**(可选) | NVMe 文件 / Mooncake / NIXL / HF3FS / ... | `--hicache-storage-backend` | 8 种可选 backend,全是外部/分布式存储 |

`backend_factory.py` 注册的 L3 backend 有 8 种:`file`、`nixl`、`mooncake`、`hf3fs`、`aibrix`、`eic`、`simm`、`mori`——**没有一种是"CPU 内存"**,因为 CPU 内存已经是 L2 了。

所以"CPU 内存做二级缓存"在 SGLang 里是**默认且独立**的 L2 层,不是某个 backend 选项。vLLM 也有类似设计(`--cpu-offload-gb`、`--kv_offloading_backend native`)。

> **第一个纠正**:之前把"二级缓存 = NVMe"当成唯一可能,是错的。L2 默认是 host DRAM,NVMe 只是 L3 选项之一。

---

## 二、两条路径的源码实现

定义清楚要对比的两条路径:

- **路径 A**:prefill GPU → Mooncake RDMA → decode GPU(跨机取 KV)
- **路径 B**:decode 本机 host DRAM(L2)→ H2D → decode GPU(本机取已缓存的 KV)

### 路径 A:Mooncake RDMA 的三段串行

源码:`mooncake/conn.py` + `common/staging_buffer.py` + `mooncake_transfer_engine.py`

**Step 1:Prefill 侧 gather(GPU kernel)**

`_gather_all_layers_*` 在 prefill 的 `_gather_stream` 上,用 Triton fused kernel 或 `gather_kv_head_slices`,把分散在 prefill KV pool 不同 page 里的目标 KV,**按层 gather 到一段连续的 staging buffer**。这是 GPU 上的 scatter→contiguous 重排。

**Step 2:RDMA 传输(GPU→GPU)**

```python
# mooncake/conn.py
ret = self.engine.batch_transfer_sync(
    mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
)
```

`register_memory` 注册的是 **GPU 显存指针**(`kv_data_ptrs` 来自 `get_contiguous_buf_infos()`,内存种类标记 `VRAM`),所以是 **GPU-to-GPU 的 RDMA 直传**(GPUDirect-RDMA),数据不进 host。

**Step 3:Decode 侧 scatter(staging→pool)**

RDMA 写到 decode 机器上预注册的 staging buffer(GPU 显存),`staging_buffer.py` 里 `_gather_stream.wait_stream` 之后,把 staging buffer 里的连续数据 scatter 写回 decode 自己 KV pool 的对应 page 位置。

**路径 A 延迟 = gather(GPU kernel) + RDMA(网络) + scatter(GPU kernel)**,三段串行,且 gather/scatter 会抢 prefill/decode 的 SM。

### 路径 B:L2 restore 的单段 + layer-wise overlap

源码:`managers/cache_controller.py:743` 的 `load()` + `start_loading()` + `pool_host/mla.py:224` 的 `load_to_device_per_layer`

**Step 1:命中查询 + 专用流加载**

`start_loading()` 在**专门的 `load_stream`** 上对每一层调 `load_to_device_per_layer`:

```python
# cache_controller.py
with device_module.stream(self.load_stream):
    for i in range(self.layer_num):
        self.mem_pool_host.load_to_device_per_layer(...)   # L2 → L1
        producer_event.complete(i)   # 每层完成记录 event
```

**Step 2:传输 kernel(自定义,nontemporal)**

`sgl-kernel/csrc/kvcacheio/transfer.cu` 里的 kernel 不是 `cudaMemcpy`,而是自定义 kernel:

```cpp
// CUDA 路径
asm volatile("ld.global.nc.b64 %0,[%1];" ...);   // non-temporal load
asm volatile("st.global.cg.b64 [%0],%1;" ...);   // write-combining store

// ROCm 路径
uint64_t tmp = __builtin_nontemporal_load(src + j);
__builtin_nontemporal_store(tmp, dst + j);
```

用 nontemporal 传输绕过 L2 cache,不污染 cache,直达显存。

**Step 3:layer-wise overlap(关键)**

`LayerDoneCounter` + `LayerLoadingEvent`(`cache_controller.py:64-115`):

```python
class LayerLoadingEvent:
    def complete(self, layer_index):
        self.load_events[layer_index].record()   # 每层加载完记录
    def wait(self, layer_index):
        device_module.current_stream().wait_event(self.load_events[layer_index])
        # 计算侧等对应层就绪,加载 layer N+1 和计算 layer N 并行
```

**路径 B 延迟 = Σ 每层 H2D,但 layer-wise overlap 后 ≈ max(单层 H2D, 单层 attention) × 层数**。专用 `load_stream` 不抢 attention 计算流,无 gather/scatter。

### 路径 B 的结构优势(源码可确认)

| 维度 | 路径 A | 路径 B |
|---|---|---|
| 串行阶段 | 3 段(gather + RDMA + scatter) | 1 段(H2D)+ overlap |
| GPU kernel 抢 SM | gather/scatter 抢 prefill/decode SM | 无,专用 `load_stream` |
| 与计算重叠 | RDMA 可与下段 prefill 重叠 | layer-wise 与 attention 重叠 |
| cache 污染 | 正常访问 | nontemporal,不污染 L2 |

> **源码层面的结构性事实(可靠)**:路径 B 有三个机制优势——无 gather/scatter、专用流不抢 SM、layer-wise overlap 隐藏延迟。

---

## 三、带宽对比(被硬件事实纠正的关键部分)

### 错误版本:按单条 IB 算

最初几轮我按"单条 HDR IB 200 Gbps = 25 GB/s"算,得出"路径 B 的 H2D PCIe 带宽(~50-64 GB/s)是路径 A 的 2 倍"。**这是错的**。

### 纠正:这套部署有 8 张 IB,而且 SGLang 在用多卡并行传输

**部署确认**:`deployment-guide.md` 写明"InfiniBand RDMA 网络(8 个 IB 设备 mlx5_ib0~ib7)"。

**源码确认多卡并行**:脚本开了 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1`:

```python
# common/conn.py:166
self.enable_all_cp_ranks_for_transfer = (
    envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()  # ← 你们开了
    or cp_sharded_prefill
    or hybrid_decode_pulls_all_ranks
)
# conn.py:186 注释:
# When SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER is True,
# all CP ranks participate in KV transfer
```

prefill 用了 CP8(Context Parallel 8 卡)+ `--enable-dsa-prefill-cp-layersplit`,8 个 CP rank 各起一个 transfer engine。Mooncake 侧 `get_ib_devices_for_gpu(ib_device, gpu_id)` 按 GPU 取对应 IB 设备(`mooncake_transfer_engine.py:130`)。8 条 IB 口并行传输。

### 修正后的带宽对比

| | 路径 A(RDMA,8×IB 并行) | 路径 B(H2D,单机 PCIe) |
|---|---|---|
| 单链路峰值 | HDR IB 200 Gbps ≈ 25 GB/s | PCIe Gen5 x16 ~50-64 GB/s |
| 并行链路数 | **8**(每 CP rank 一个 engine + IB 口) | 8 GPU 各有 PCIe,但**到同一 host DRAM 汇聚** |
| 聚合峰值 | **~200 GB/s**(8×25,理论) | ~50-64 GB/s(host DRAM 带宽是瓶颈) |
| 实际可持续 | 受拓扑/top-of-rack 限制 | 受 host DRAM 带宽和 PCIe 根复合体限制 |

> **第二个纠正(重要)**:8×IB 聚合后,路径 A 带宽(~200 GB/s)远高于路径 B(~50-64 GB/s)。之前说"路径 B 带宽是 A 的 2 倍"完全反了。带宽维度路径 A 占优。

### 净较量:取决于负载特征

- **小消息**(短前缀,几 MB):路径 A 的 gather/scatter kernel launch + RDMA per-WR 开销占比大,有效带宽远低于 200 GB/s 峰值;路径 B 的 layer-wise overlap 优势凸显 → **路径 B 可能赢**
- **大消息**(长前缀,几十 MB+):路径 A 的 8×IB 聚合带宽优势放大,gather/scatter 占比下降 → **路径 A 可能赢**
- **跨机 vs 本机**:路径 A 跨机器,路径 B 本机——跨机器的 RTT 和 bootstrap 是路径 A 独有开销

> **未实测部分(诚实标注)**:具体在什么消息大小下路径 A/B 谁快,无法仅凭源码断言,需要实测。

---

## 四、DecodeKVCacheOffloadManager 的完整机制

这是 decode 侧独有的组件,只在 `--disaggregation-decode-enable-offload-kvcache` 开启时创建。源码:`disaggregation/decode_kvcache_offload_manager.py`(352 行)。

### 开启条件(源码 `server_args.py:6759`)

```python
if self.disaggregation_decode_enable_offload_kvcache:
    if self.disaggregation_mode != "decode":   # 必须 decode 模式
        raise ValueError(...)
    if self.hicache_storage_backend is None:   # 必须配 L3 backend
        raise ValueError(...)
```

而且需要**两个开关配合**才能形成闭环:

| 开关 | 作用 | 阶段 |
|---|---|---|
| `--disaggregation-decode-enable-offload-kvcache` + `--hicache-storage-backend file` | 开 offload manager(写回侧:GPU→L2→L3) | decode 过程中 |
| `--disaggregation-decode-enable-radix-cache` + `--enable-hierarchical-cache` | 开 `enable_decode_hicache`(接收侧:L3→L2→GPU) | 下个请求进 decode 时 |

只开写回没接收侧,存了用不上;只开接收没写回,L2/L3 里没东西可取。

### 内部数据结构

```python
class DecodeKVCacheOffloadManager:
    decode_host_mem_pool       # L2:host DRAM pool
    cache_controller           # 复用 HiCacheController,管 L2<->L3
    offload_stride             # 每隔多少 token offload 一次
    offloaded_state[rid]       # 每个请求的 offload 进度(OffloadedState)
    ongoing_offload[ack_id]    # 正在 GPU→host 的 op
    ongoing_backup[ack_id]    # 正在 host→L3 的 op
    offload_inflight[rid]      # 引用计数
```

`OffloadedState` 追踪:
- `prefill_len`:prefill 段长度(page 对齐)
- `inc_len`:已 offload 的增量长度
- `last_hash`:链式 hash,用于 L3 索引

### 触发时机:每步 decode 之后

入口在 `batch_result_processor.py:945`:

```python
# 每个 req decode 完一步后调用
if self.server_args.disaggregation_decode_enable_offload_kvcache:
    if not self.decode_offload_manager.offload_kv_cache(req):
        self.decode_offload_manager.finalize_release_on_finish(req)
```

### `offload_kv_cache` 核心流程

**Step 1:计算增量边界**

```python
all_tokens = req.origin_input_ids + req.output_ids[:-1]
prefill_offloaded_len = len(req.origin_input_ids) // page_size * page_size

incremental_new = incremental_total - state.inc_len       # 减去已 offload 的
incremental_aligned_len = incremental_new // stride * stride  # 按 stride 对齐
```

关键设计:**decode offload 的是"增量部分",prefill 段不动**。prefill 段的 KV 是从 prefill 实例传来的,请求结束时才释放(见后文)。

**Step 2:异步 write(GPU→host)**

```python
host_indices = self.cache_controller.write(
    device_indices=incremental_indices.long(),
    node_id=ack_id,
)   # 异步 D2H,在专用 write_stream 上
```

**Step 3:登记 inflight,推进进度**

```python
self._mark_offload_started(req.rid)
self.ongoing_offload[ack_id] = (req, host_indices, incremental_tokens, ...)
state.inc_len += incremental_aligned_len   # 立即推进,下次只算新的
```

### 两级异步流水线

每个 decode 调度 tick 调用 `check_offload_progress`(`decode.py:2189`),驱动两级流水线:

**Stage 1:GPU→host(L2)完成检查**

```python
ack = self.cache_controller.ack_write_queue.pop(0)
ack.finish_event.synchronize()   # 等 write_stream 的 CUDA event(专用流,不阻塞 forward)
# → 触发下一级
last_hash = self._trigger_backup(req, host_indices, incremental_tokens, ...)
```

**Stage 2:host→L3(NVMe)备份,然后释放 host**

```python
def _trigger_backup(self, req, host_indices, incremental_tokens, ...):
    page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
    ack_id = self.cache_controller.write_storage(
        host_indices, incremental_tokens, hash_value=page_hashes,
    )   # 异步写 NVMe
```

L3 写完后释放 host 内存:
```python
def _check_backup_progress(self, finish_count):
    ...
    self.decode_host_mem_pool.free(host_indices)   # L3 写完,host 还回去
```

**两级流水线**:GPU→host(L2)→L3(NVMe)→释放 host。L3 是兜底,L2 是临时区。host 内存用完就还,所以 L2 不需要很大(默认 ratio 2×)。

### 链式 hash:让 L3 可被未来请求命中

```python
def _compute_prefix_hash(self, tokens, prior_hash=""):
    for offset in range(0, len(tokens), page_size):
        last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
        page_hashes.append(last_hash)
```

每个 page 的 hash = `hash(prev_hash, page_tokens)`,是**链式前缀 hash**。未来请求前缀相同则 hash 一致,L3 能按 hash 查到。这就是为什么 offload 到 L3 后还能被复用——按前缀 hash 存,不是按 rid 存。

### 请求结束:延迟释放 + 尾部处理

`_release_finished_req` 里,源码注释说得很清楚:

```python
# Prefill-aligned GPU slots are freed at request finish in
# _release_finished_req, NOT here. The decoding request
# continues to attend to those slots via req_to_token; freeing
# them mid-decode races with concurrent admission, which can
# reuse the slots and produce cross-pollinated KV reads.
```

**GPU 显存**:请求结束时才释放(因为 decode 每步 attention 还在用,提前释放会 race)。
**host/L3 里的 KV**:**不释放**,成为前缀缓存留给未来请求命中——这正是 offload 的目的。

### 与 decode forward 的并发关系

```
decode forward stream:   ──[attention N]──[attention N+1]──[attention N+2]──
                              │                │
                              ▼                ▼
offload write_stream:     ──[D2H page A]──[D2H page B]──[D2H page C]──  (专用流,不抢 SM)
                              │                │
                              ▼                ▼
backup thread:            ──[host→NVMe A]──[host→NVMe B]──[host→NVMe C]──  (后台线程,纯 IO)
```

- `write_stream` 独立 CUDA stream,nontemporal 不污染 cache
- backup 是后台线程,纯 IO
- forward 流只在 `check_offload_progress` 里 `synchronize` 等 ack 的 CUDA event,而 event 在 write_stream 上,和 forward 并行

所以 offload 的 DMA 延迟被藏在 decode forward 计算里。

---

## 五、接收侧:路径 B 的 restore 状态机

下个请求进 decode 时,如何走路径 B 取回缓存的前缀 KV。源码:`disaggregation/decode_hicache_mixin.py`。

### 前缀匹配:三层命中

```python
@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor     # L1 GPU 显存已有
    l2_host_hit_length: int          # L2 host DRAM 已有
    l3_storage_hit_length: int      # L3 NVMe 已有

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len
```

`_build_decode_prefix_match` 依次查 L1 → L2 → L3,得到三层命中长度。

### L3→L2 预取

```python
def _start_hicache_prefetch(self, req, prefix_match):
    self.tree_cache.prefetch_from_storage(
        req.rid, node, suffix, last_hash, prefix_keys
    )
```

L3 命中则预取到 L2(NVMe → host DRAM)。

### L2→L1 restore 状态机

`_process_hicache_local_restores` 三阶段推进:

```python
# Phase A:推进在途 DMA 到 READY
for dr in active:
    if dr.hicache_restored_node is not None and \
       self.tree_cache.is_load_back_event_done(dr.hicache_load_consumer_index):
        dr.hicache_restore_status = HiCacheRestoreResult.READY

# Phase B:排新 load_back(如果有空槽)
queued = [dr for dr in active if ... and self._try_hicache_queue_load_back(dr)]

# Phase C:启动合并 DMA,绑定 consumer_index
consumer_index = self.tree_cache.ready_to_load_host_cache()
```

### `HiCacheRestoreGatedKVReceiver`:闸住 poll

```python
class HiCacheRestoreGatedKVReceiver:
    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()   # 先看路径 A 的 RDMA 完没完成
        if (poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING):
            return KVPoll.Transferring   # 路径 A 完了但路径 B 没完,继续等
        return poll
```

**关键**:路径 A 和路径 B 是并行的——RDMA 传差量的同时,L2/L3 restore 取命中部分。两者都 READY 才算完成(`pop_transferred` 里的逻辑)。

### 提交:`_commit_hicache_local_restore_to_req`

```python
self.tree_cache.req_to_token_pool.write(
    (decode_req.req.req_pool_idx,
     slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len)),
    decode_req.hicache_restored_kv_indices,
)
decode_req.req.prefix_indices = torch.cat(
    [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
)
```

把 restore 回来的 KV indices 写进请求的 req_to_token 映射,attention 就能用了。

---

## 六、整个流程的闭环

把写回侧和接收侧合起来看,就是"减少从 prefill 传输"的完整闭环:

```
请求 1 进入 decode
  ├─ 路径 B(L2/L3 restore):命中前缀部分
  └─ 路径 A(Mooncake RDMA):未命中部分,从 prefill 传

请求 1 decode 过程中
  └─ DecodeKVCacheOffloadManager:每步增量 KV → L2 → L3(NVMe)

请求 1 结束
  ├─ GPU 显存:释放
  └─ L2/L3 里的 KV:留着(按前缀 hash 索引)

请求 2 进入 decode(共享前缀)
  └─ 路径 B 命中 → 直接从 L2/L3 取,不走路径 A  ← 这就是"减少传输"
```

源码确认:**最初的直觉"decode 用二级缓存减少从 prefill 的传输"在机制层面是对的**。`DecodeKVCacheOffloadManager` + `decode_hicache_mixin` 就是为此设计的——请求 1 把增量 KV 存下来,请求 2 命中前缀时走路径 B 而非路径 A。

---

## 七、那为什么这套部署 decode 没开?(最终结论)

把所有事实摆出来:

### 可能的原因(按可信度排)

1. **8×IB 下路径 A 已经够快(最可能)**
   路径 A 聚合带宽 ~200 GB/s(8×IB),路径 B 受 host DRAM 带宽限制(~50-64 GB/s)。当路径 A 已经很快时,decode offload 的收益——即"省下的路径 A 传输"换成"路径 B 加载"——带宽上不占优,且要额外付 offload 写回开销(D2H + host→NVMe)。**边际收益可能不足以抵消开销**。

2. **镜像版本的特性成熟度**
   镜像 `v0.5.14-pr47` + CjiW fork 补丁。`DecodeKVCacheOffloadManager` 在 main 分支是较新特性,测试文件还在 `test/registered/` 下。这个 fork 基于 v0.5.14,可能没有这套代码,或未在 ROCm/MI300X 上验证。
   **验证方法**:
   ```bash
   docker run --rm b200routeraca.azurecr.io/mindverse/sglang:v0.5.14-pr47 \
     python3 -m sglang.launch_server --help 2>&1 | grep -i "offload-kvcache"
   ```

3. **配置复杂度**
   要调 `num-reserved-decode-tokens`、`hicache-ratio`、stride 等,调不好影响 decode 吞吐,部署时可能先没启用。

### 怎么开(如果镜像支持)

在 decode 启动命令加:

```bash
# 接收侧 restore(路径 B)
--enable-hierarchical-cache \
--hicache-ratio 2 \                          # L2 host DRAM = 2× GPU pool
--hicache-mem-layout page_first \

# 写回侧(offload manager)
--hicache-storage-backend file \
--file-storage-path /nvme/hicache \
--disaggregation-decode-enable-offload-kvcache \
--num-reserved-decode-tokens 128 \
```

环境变量:
```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="/nvme/hicache"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="200G"
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE="10G"
# 可选:调 offload 粒度
# export SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE=128
```

`mem-fraction-static` 可能要从 0.80 降一点给 L2 元数据留空间。

### 实测验证方案

路径 A 有现成监控指标:
```python
"kv_transfer_gbps":   'sum(sglang:kv_transfer_speed_gb_s_sum{...}) / ...'
"kv_latency_s":       '...kv_transfer_latency_ms...'
"prefill_transfer_s": '...per_stage_req_latency_seconds_sum{stage="prefill_transfer_kv_cache"}'
```

对照实验:
1. 当前配置(decode 不开 offload)跑一批共享前缀请求,记 TTFT 和 `kv_transfer_speed_gb_s`(全走路径 A)
2. 打开 offload + restore,跑同样请求(第二批起命中 L2/L3,走路径 B),记 TTFT
3. 对比 TTFT 差值 = 路径 B 相对 A 的实际收益

> **注意**:路径 B 的 restore 指标(`ack_load_queue` 的 timing event)在监控脚本里没有对应 Prometheus metric,需要在 decode 机器上看 SGLang 日志里的 hicache load timing,或加 metric。

---

## 八、复盘:每一步纠正踩在什么取舍上

这次讨论最有价值的不是最终结论,而是每一步纠正都踩在一个真实的设计取舍上:

| 轮次 | 我的错误论断 | 被什么纠正 | 暴露的真实取舍 |
|---|---|---|---|
| 1 | "decode 每步读 KV,NVMe 会拖延迟" | 你指出二级缓存可以是 CPU 内存 | NVMe(L3)和 host DRAM(L2)是不同介质,延迟差几十倍 |
| 2 | "RDMA 比 NVMe load 快" | 你指出有 8 张 IB | 单条 IB vs 8×IB 聚合,带宽差 8 倍 |
| 3 | "路径 B 带宽是 A 的 2 倍" | 8×IB 事实(同上) | 按单链路算 vs 按聚合算,结论反转 |
| 4 | "decode 没开是因为框架不支持" | 源码确认 `DecodeKVCacheOffloadManager` 存在 | 框架支持 ≠ 部署开启,配置选择背后是带宽权衡 |

**核心教训**:技术分析不能只看源码机制,还得看实际硬件配置。源码告诉你"机制上能不能做",硬件告诉你"做了划不划算"。8×IB 这个事实,把"机制上路径 B 更优"的结论直接反转成"带宽上路径 A 更优"。

另外一个教训:**诚实标注不确定处比强行给结论更值钱**。最初几轮我为了支持"你的直觉对",不断往"路径 B 更快"方向找证据,但带宽事实不支持。承认"取决于负载特征、需实测"才是诚实的。

---

## 附录:关键源码文件索引

| 文件 | 作用 | 关键行 |
|---|---|---|
| `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | decode offload manager 主体 | 111 `offload_kv_cache` |
| `python/sglang/srt/managers/cache_controller.py` | HiCacheController,管 L2<->L3 读写 | 743 `load`、`start_loading` |
| `python/sglang/srt/mem_cache/pool_host/mla.py` | host DRAM pool,`load_to_device_per_layer` | 224 |
| `python/sglang/srt/mem_cache/storage/backend_factory.py` | 8 种 L3 backend 注册 | 195-235 |
| `python/sglang/srt/disaggregation/decode_hicache_mixin.py` | 接收侧 restore 状态机 | 99 `_start_hicache_prefetch`、240 `_process_hicache_local_restores` |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 路径 A 的 RDMA 传输 | 579 `batch_transfer_sync` |
| `python/sglang/srt/disaggregation/common/staging_buffer.py` | 路径 A 的 gather/scatter | 340 |
| `sgl-kernel/csrc/kvcacheio/transfer.cu` | L2→L1 的 nontemporal 传输 kernel | 33 |
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | Mooncake engine,8×IB 按 GPU 取设备 | 130 |
| `python/sglang/srt/disaggregation/common/conn.py` | `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 多卡并行 | 166、186 |

---

## 附录:部署参数对照

| 参数 | prefill(deploy_pd_new.sh) | decode(deploy_pd_new.sh) | decode(若开 offload) |
|---|---|---|---|
| `--mem-fraction-static` | 0.70 | 0.80 | 需降低(给 L2 元数据) |
| `--enable-hierarchical-cache` | ✅ | ❌ | ✅ |
| `--hicache-ratio` | 1 | — | 2 |
| `--hicache-storage-backend` | file | — | file |
| `--hicache-mem-layout` | page_first | — | page_first |
| `--disaggregation-decode-enable-radix-cache` | — | ✅ | ✅ |
| `--disaggregation-decode-enable-offload-kvcache` | — | ❌ | ✅ |
| `--num-reserved-decode-tokens` | — | — | 128 |
| `--disaggregation-ib-device` | mlx5_ib0 | mlx5_ib0 | mlx5_ib0 |
| `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` | 1 | 1 | 1 |

---

*本文基于 SGLang main 分支源码 + 实际部署脚本分析。镜像 `v0.5.14-pr47` 的具体支持情况需上机验证。所有带宽数字为硬件规格理论值,实际可持续带宽需实测。*
