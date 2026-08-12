# 为什么 PD 分离中，decode 不需要开二级缓存

PD 分离部署里，prefill 节点普遍使用 L2 cache（把 KV cache offload 到 host DRAM）来扩大可用缓存空间、提升命中率。一个自然的问题是：**decode 节点是不是也应该有类似的机制？**

答案是：**机制完整支持，但带宽维度上不需要**。SGLang 的 `DecodeKVCacheOffloadManager` 提供了完整的 decode 侧 L2 + L3 支持，但 8×IB 聚合下从 prefill 跨机重传（路径 A，~200 GB/s）比从 decode 本地 L2 缓存加载（路径 B，~50-64 GB/s）更快，因此 decode offload 缺乏边际收益。

---

## PD 分离基础

LLM 推理分两阶段：**prefill**（处理整段输入 prompt，算出 KV cache，计算重）和 **decode**（逐个生成 token，每步读全部历史 KV，计算轻）。**PD 分离**把两阶段拆到不同实例：prefill 实例算完 prompt 后通过 RDMA 把 KV 传给 decode 实例；decode 实例只负责逐 token 生成。

---

## HiCache 三层架构

SGLang 的 HiCache 是三层结构：

| 层级 | 介质 | 控制参数 | 说明 |
|---|---|---|---|
| **L1** | GPU HBM | `--mem-fraction-static` | 显存 KV pool，活跃请求常驻 |
| **L2** | host DRAM | `--hicache-ratio`（默认 2.0） | 默认就有，不依赖 storage backend |
| **L3**（可选） | NVMe / Mooncake / NIXL / ... | `--hicache-storage-backend` | 8 种可选 backend，全是外部/分布式存储 |

L2 默认是 host DRAM，NVMe 只是 L3 的一个选项。源码确认（`decode_kvcache_offload_manager.py`）：

```python
# L2：host DRAM pool
self.decode_host_mem_pool = MLATokenToKVPoolHost(
    kv_cache, server_args.hicache_ratio, ...
)
# L3：可选的 storage backend
self.cache_controller = HiCacheController(
    mem_pool_host=self.decode_host_mem_pool,                  # ← L2 = host DRAM
    storage_backend=server_args.hicache_storage_backend,      # ← L3 = file/nixl/...
)
```

---

## Decode 侧二级缓存：机制完整

`DecodeKVCacheOffloadManager` 提供完整的 decode offload 机制：

- **写回**：每步 decode 之后把增量 KV 异步 offload 到 host DRAM（L2）→ L3（可选）
- **加载**：下次请求命中时，通过专用 `load_stream`（不抢 SM）做 layer-wise overlap 恢复
- **链式 hash**：让 L3 可被未来请求命中

机制完整，源码都在（见文末索引）。

---

## 核心论证：路径 A 比路径 B 更快

PD 分离里 decode 拿到 KV 有两条路径：

- **路径 A**：从 prefill 跨机 RDMA 重新传输（每次新请求都走）
- **路径 B**：从 decode 本地 L2 cache 加载（命中时走）

**关键事实**：开启 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` 后，8 个 CP rank 各起一个 transfer engine，8 条 IB 口并行传输。源码：

```python
# common/conn.py:166
self.enable_all_cp_ranks_for_transfer = (
    envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()  # ← 8 卡并行
    or cp_sharded_prefill
    or hybrid_decode_pulls_all_ranks
)
```

### 带宽

| | 路径 A（8×IB RDMA） | 路径 B（单机 H2D） |
|---|---|---|
| 单链路峰值 | HDR IB 200 Gbps ≈ 25 GB/s | PCIe Gen5 x16 ~50-64 GB/s |
| 并行链路数 | **8**（每 CP rank 一个 engine + IB 口） | 8 GPU 各有 PCIe，但到同一 host DRAM 汇聚 |
| 聚合峰值 | **~200 GB/s** | ~50-64 GB/s（host DRAM 带宽是瓶颈） |

**带宽维度路径 A 占优**：8×IB 聚合后 ~200 GB/s 远高于路径 B 的 ~50-64 GB/s。

### 路径 B 的优势

路径 B 在结构上有几个优势：无 gather/scatter、专用流不抢 SM、layer-wise overlap 隐藏延迟。但这些结构优势无法弥补带宽差距——在带宽主导的大消息场景下不占优，在小消息场景下可能反过来（见下节例外）。

---

## 例外：小消息场景

- **小消息**（短前缀，几 MB）：路径 A 的 gather/scatter kernel launch + RDMA per-WR 开销占比大，有效带宽远低于峰值；路径 B 的 layer-wise overlap 优势凸显 → 路径 B 可能赢
- **大消息**（长前缀，几十 MB+）：路径 A 的 8×IB 聚合带宽优势放大 → 路径 A 可能赢
- **跨机 vs 本机**：路径 A 跨机器，路径 B 本机——跨机器的 RTT 和 bootstrap 是路径 A 独有开销

具体在什么消息大小下路径 A/B 谁快，无法仅凭源码断言，需要实测。

---

## 实测验证方案

路径 A 有现成监控指标：

```promql
sum(sglang:kv_transfer_speed_gb_s_sum{...})   # 路径 A 实测带宽
...kv_transfer_latency_ms...                   # KV 传输延迟
...per_stage_req_latency_seconds_sum{stage="prefill_transfer_kv_cache"}
```

对照实验：

1. 当前配置（decode 不开 offload）跑一批共享前缀请求，记 TTFT 和 `kv_transfer_speed_gb_s`（全走路径 A）
2. 打开 offload + restore，跑同样请求（第二批起命中 L2/L3，走路径 B），记 TTFT
3. 对比 TTFT 差值 = 路径 B 相对 A 的实际收益

> 路径 B 的 restore 指标（`ack_load_queue` 的 timing event）在监控脚本里没有对应 Prometheus metric，需要在 decode 机器上看 SGLang 日志里的 hicache load timing，或加 metric。

---

## 附录：关键源码文件索引

| 文件 | 作用 | 关键行 |
|---|---|---|
| `disaggregation/decode_kvcache_offload_manager.py` | decode offload manager 主体 | 111 `offload_kv_cache` |
| `managers/cache_controller.py` | HiCacheController，管 L2↔L3 读写 | 743 `load`、`start_loading` |
| `mem_cache/pool_host/mla.py` | host DRAM pool，`load_to_device_per_layer` | 224 |
| `mem_cache/storage/backend_factory.py` | 8 种 L3 backend 注册 | 195-235 |
| `disaggregation/decode_hicache_mixin.py` | 接收侧 restore 状态机 | 99、240 |
| `disaggregation/mooncake/conn.py` | 路径 A 的 RDMA 传输 | 579 `batch_transfer_sync` |
| `disaggregation/common/conn.py` | `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 多卡并行 | 166、186 |

---

*本文基于 SGLang v0.5.14 源码与实际部署配置分析。所有带宽数字为硬件规格理论值，实际可持续带宽需实测。*
