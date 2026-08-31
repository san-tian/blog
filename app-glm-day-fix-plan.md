# app_macaron_relay_stack_glm_day 修复方案（评审稿 v2）

> **v2 变更**：按「先格式修复、再质量筛选」两层重构。每个问题明确定归属：**格式 / 筛选 / 格式层丢弃**。新增 tools 空数组归因实测（99.4% 无工具流量）、空 messages 6 条明细、两层流水线与分层验收。
>
> **一句话结论**：这张表目前是"未闭环、未对齐、未筛选"的中间产物。修复分两层：**格式修复层**（每条产出记录 = 结构合法的完整对话，造不出的不产出）→ **质量筛选层**（从合法记录里选有训练价值的，18 条款）。共 6 个问题 + 若干边缘 case，全部定归属。

---

## 0. 基线与口径说明

**分区语义**：`statis_day` 是**全量快照**（非日增量），数字随快照累积增长：

| statis_day | 总行数 |
|---|---|
| 20260827（已过期） | 4,486 |
| 20260828 | 49,876 |
| 20260829 | 66,016 |
| **20260830（最新基线）** | **69,330** |

**LIFECYCLE 3 天**：验证请用最新分区（0827 已被回收，本文实测数字基于 0830 快照）。

**0830 快照按 source 实测**（69,330 条 = chat 65,691 + messages 3,093 + responses 546）：

| source | 行数 | 空 messages | tools=[] | **assistant 结尾** | effort ∈{high,max} |
|---|---|---|---|---|---|
| chat_completions | 65,691 | 2 | 43,557 (66%) | **19 (0.03%)** | **0 (0%)** |
| messages | 3,093 | 0 | 710 (23%) | 2,995 (96.8%) | 2,908 (94%) |
| responses | 546 | 4 | 128 (23%) | **0 (0%)** | 239 (44%) |
| **合计** | 69,330 | 6 | 44,395 (64%) | **3,014 (4.3%)** | 3,147 (4.5%) |

**下游契约**（训练侧 L0-L6 校验器，实测依据）：
- 顶层只认 `messages/tools/chat_template_kwargs/meta` 四键，`extra="forbid"`（多一个字段直接 FATAL，实测 `Extra inputs are not permitted`）
- `reasoning_effort: Literal["high","max"]`（实测 medium 报 `Input should be 'high' or 'max'`）
- `tools` 为空数组直接 FATAL（`_reject_empty_tools`），省略键则 OK
- `messages` 至少 1 条（`min_length=1`）
- 本地转换器 `to_glm52_sft.py` 的既有约定：**强制 `reasoning_effort="max"`、空 tools 省略键**——格式层向它对齐

---

## 1. 两层架构：先格式修复，再质量筛选

### 1.1 分类判据（一句话）

> **数据在、但 ETL 表达错了 = 格式问题（修）**
> **数据在、表达也对、但内容没训练价值 = 筛选问题（滤）**
> **数据不在、构造不出合法记录 = 格式层丢弃（不产出，不硬造）**

格式层的契约：**产出的每条记录都是结构合法的完整对话**（有问有答、契约对齐、字段语义正确）。它不做价值判断——无工具纯对话照样产出（tools 省略键）。质量层的契约：**从合法记录里选有训练价值的**（多轮、工具、闭环、非闲聊、非 cron、去重）。两层职责不交叉。

### 1.2 问题归属总表

| # | 问题 | 归属 | 判据 | 动作 |
|---|---|---|---|---|
| P0-1 | chat 源缺 assistant 回复 | **格式** | 数据在（resp_choices），ETL 没拼 | 拼 `choices[0].message` |
| P0-2 | responses 源缺 assistant 回复 | **格式** | 数据在（resp_output），ETL 没用 | UDF 转+拼 |
| — | 续话客户端（previous_response_id） | **格式层丢弃** | 增量 input 构造不出完整对话 | UDF 检测碎片 → 返回 NULL → 不产出 |
| P1-3 | reasoning_effort 值域非法 | **格式** | 契约不对齐，可无损映射 | 归一 high/max |
| P1-4a | tools 表达成 `[]` | **格式** | "无工具"的合法表达是省略键，不是空数组 | `ELSE NULL` + MAX_BY 条件化 |
| P1-4b | 无工具纯对话流量（99.4%） | **筛选** | 记录合法，但无 agentic 训练价值 | 18 条款 2.x/3.x 淘汰 |
| P1-5 | 空 messages（6 条） | **格式层丢弃** | 输入缺失（续话 4 条 + 畸形 body 2 条） | WHERE 不产出 |
| — | session 结束在 tool_call 无结果 | **筛选** | 结构完整（有 assistant）但未闭环 | 18 条款 4.x |
| — | 最后请求失败 | **格式**（取最后有效交换）；全失败 → 格式层丢弃 | 数据部分在 | MAX_BY 有效 ts / WHERE |
| P2-6 | 无 18 条款筛选 | **筛选** | 表的本意 | 质量层 UDAF |
| 附-7 | 顶层 9 列 vs 训练 4 键 | **交付层** | 表设计合理（查询用辅助列） | 导出时剥离，不改表 |

### 1.3 两层流水线

```
stg → dwd（已有，不动）
        │
        ▼
【格式修复 ETL】app_macaron_relay_stack_glm_day（修好后 = 结构合法的全量表）
  职责：完整对话（拼 resp）、契约对齐（effort/tools 语义）、造不出 → 不产出
  产出：含无工具纯对话（它们结构合法，只是没训练价值）
        │
        ▼
【质量筛选 ETL】app_macaron_relay_stack_glm_day_qualified（新表）
  职责：18 条款（glm 口径，豁免 7.1/7.3 signature）+ 无工具/未闭环/单轮/闲聊/cron/去重淘汰
  产出：可直接进 L0-L6 的训练候选
        │
        ▼
【交付层】导出时剥 4 键（messages/tools/chat_template_kwargs/meta）
```

两张表分离的好处：**筛选前/后都能看**（对照排查，同 opus 的 dwd vs app_opus_day 关系），且格式修复和质量筛选可以独立迭代、独立验收。

---

## 2. 问题清单（总览）

| # | 问题 | 归属 | 根因（ETL 现状） | 影响（0830 实测） | 优先级 |
|---|---|---|---|---|---|
| P0-1 | chat_completions 源缺 assistant 回复 | 格式 | CTE 没取 `resp_choices`，messages 直接用 `req_messages` | 65,691 条只有 19 条闭环（0.03%） | **P0** |
| P0-2 | responses 源缺 assistant 回复 | 格式 | `final_resp_output` 取了但 INSERT 没用（注释写了没做） | 546 条 0 条闭环 | **P0** |
| P1-3 | reasoning_effort 值域非法 | 格式 | `COALESCE(reasoning_effort, 'medium')`；契约 `Literal["high","max"]` | chat 源 100% 非法 | P1 |
| P1-4a | tools 空数组 | 格式 | `ELSE JSON_ARRAY()` 兜底；契约要求空则省略键 | 44,395 条 (64%) | P1 |
| P1-5 | 空 messages 记录 | 格式层丢弃 | `ELSE JSON_ARRAY()` 兜底产出废记录 | 6 条 | P1 |
| P2-6 | 无 18 条款达标筛选 | 筛选 | 无 qualified 筛选（opus_day 有，glm_day 没有） | 全量含单轮/无工具/不闭环/闲聊 | P2 |
| 附-7 | 顶层 9 列 vs 训练 4 键 | 交付层 | 表设计（查询用辅助列） | 导出层剥离，不改表 | 备注 |

---

## 3. P0-1：chat_completions 源缺 assistant 回复【格式】

### 现状（etl/app/app_macaron_relay_stack_glm_day.sql）

```sql
-- CTE：根本没取 resp_choices
chat_sessions AS (
  SELECT
    r.session_id,
    ...
    MAX_BY(r.req_messages, r.ts) AS final_messages,   -- 只有请求侧
    MAX_BY(r.req_tools, r.ts) AS final_tools,
    -- ✗ 没有 MAX_BY(r.resp_choices, r.ts)
  FROM dwd_macaron_relay_stack_chat_completions_total_day r
  WHERE r.statis_day = '${statis_day}'
  GROUP BY r.session_id
)

-- INSERT：messages = 客户端重发的请求历史，最后一条是 user/tool
CASE
  WHEN final_messages IS NOT NULL AND ... THEN JSON_PARSE(final_messages)
  ELSE JSON_ARRAY()
END AS messages,
```

**为什么缺**：chat completions 协议客户端每次重发完整历史（`req_messages` 含之前所有轮次），但**本轮模型的回答在 `resp_choices` 里**。ETL 只取了请求侧 → 训练样本"有问无答"（24,417 条 user 结尾 + 18,440 条 tool 结尾）。

**关键证据（stg 原始 response_body 实测）**：`choices[0].message` 就是标准 GLM assistant 格式，可直接拼接：

```json
{"id":"...","object":"chat.completion","model":"glm-5.2","created":...,
 "choices":[{"index":0,"message":{
   "role":"assistant",
   "content":"Let me check how many total custom class names...",
   "reasoning_content":"Found it! The exclusion regex is:..."}}]}
```

### 修法（SQL-only，无需新 UDF）

**① CTE 加取 resp_choices + 行级过滤只保留有效应答的请求**（同时实现「最后请求失败→取最后有效交换」「全失败→消失」）：

```sql
chat_sessions AS (
  SELECT
    r.session_id,
    MAX_BY(r.request_id, r.ts) AS request_id,
    MAX(r.request_date) AS biz_date,
    MAX_BY(r.model, r.ts) AS model,
    MAX_BY(r.upstream_model, r.ts) AS actual_model,
    MAX_BY(r.req_messages, r.ts) AS final_messages,
    MAX_BY(r.resp_choices, r.ts) AS final_resp_choices,   -- [新增] 最后一次有效应答
    MAX_BY(r.req_tools, CASE WHEN r.req_tools IS NOT NULL AND r.req_tools != '' AND r.req_tools != '\\N'
                             THEN r.ts END) AS final_tools,   -- [新增] 条件化：会话中途带工具、最后一次没带 → 取最后一次带 tools 的请求（否则声明丢失，18 条款 3.x 配对会挂）
    MAX_BY(r.req_reasoning_effort, r.ts) AS reasoning_effort,
    COUNT(*) AS session_n_turns,
    CAST(MAX_BY(r.n_messages, r.ts) AS BIGINT) + SUM(r.n_tool_calls) + 1 AS session_n_messages,
    SUM(CASE WHEN r.n_choices > 0 THEN 1 ELSE 0 END) AS total_n_thinking,
    MAX_BY(r.n_tool_calls, r.ts) AS total_n_tool_use
  FROM dwd_macaron_relay_stack_chat_completions_total_day r
  WHERE r.statis_day = '${statis_day}'
    AND r.resp_choices IS NOT NULL
    AND r.resp_choices != '' AND r.resp_choices != '\\N'    -- [新增] 无应答的请求不参与聚合
  GROUP BY r.session_id
)
```

> 行级过滤的语义：`MAX_BY(req_messages, ts)` 变成"最后一次**成功应答**的请求"。session 尾部失败请求被跳过（历史+应答仍完整闭环）；整个 session 全失败则自然消失（构造不出，格式层丢弃）。副作用（正向）：`COUNT(*)` 变成有效轮数。

**② INSERT 的 messages 拼回应答**（同时修 P1-3/P1-4a/P1-5 的 chat 部分）：

```sql
SELECT
  session_id, request_id, biz_date, model, actual_model,
  -- [修复] 历史去掉尾 ']'，拼上 choices[0].message，再闭合
  JSON_PARSE(CONCAT(
    SUBSTR(final_messages, 1, LENGTH(final_messages) - 1), ',',
    GET_JSON_OBJECT(final_resp_choices, '$[0].message'), ']'
  )) AS messages,
  CASE
    WHEN final_tools IS NOT NULL AND final_tools != '' AND final_tools != '\\N' AND JSON_VALID(final_tools)
    THEN JSON_PARSE(final_tools)
    ELSE NULL                                              -- [修复 P1-4a] 空数组 → NULL（省略键语义）
  END AS tools,
  JSON_OBJECT('reasoning_effort',
    CASE WHEN reasoning_effort IN ('high', 'max') THEN reasoning_effort ELSE 'max' END
  ) AS chat_template_kwargs,                                -- [修复 P1-3]
  JSON_OBJECT( ... 原样不变 ... ) AS meta
FROM chat_sessions
WHERE final_messages IS NOT NULL AND final_messages != '' AND final_messages != '\\N'
  AND final_messages != '[]'                               -- [修复 P1-5] 空历史丢弃
  AND JSON_VALID(final_messages)
  AND SUBSTR(final_messages, 1, 1) = '['                   -- 防御 'null'/对象等非法输入
  AND GET_JSON_OBJECT(final_resp_choices, '$[0].message') IS NOT NULL  -- 防御空 choices
```

**边界说明**：
- `message.content` 为 null（纯 tool_calls 应答）合法，`tool_calls` 字段原样保留
- session 结束在 tool_call（客户端没续）→ 拼后以 assistant+tool_calls 结尾、无 tool result → **结构完整但未闭环，归筛选层**（18 条款 4.x 淘汰）
- **NUL 字节风险**：dwd ETL 注释提过 reasoning 字段需在 parse UDF 里清洗 NUL（ODPS-0121095）。若 `resp_choices.message.reasoning_content` 含 NUL，需在 chat_completions parse UDF 里同样清洗
- 多 choice（n>1）取 `[0]`

---

## 4. P0-2：responses 源缺 assistant 回复【格式 + 格式层丢弃】

### 现状

```sql
-- CTE 取了 final_resp_output，但 INSERT 里完全没用：
-- 注释写着 "[system_msg] + input_items + [assistant_response]"
-- 实际代码只拼了 instructions(可选) + final_input，assistant_response 从未出现
```

**证据**：0830 快照 546 条 responses 记录，assistant 结尾 **0 条**。dwd 的 `req_input` 已被 parse UDF 归一成 chat messages 格式（实测 app 表 responses 记录的 messages 是 `[{"role":"system","content":"..."}]` 标准格式），input 侧不用动，**只需补 output → assistant 的转换**。

### 续话客户端（previous_response_id）→ 格式层丢弃

空 messages 的 4 条 responses 记录全是它（见 §6 P1-5 明细）：n_turns 75/288/212/33 的大 session，`req_input` 却为空或只有增量——**靠服务端状态续话，input 拼不出全量历史**。这类不是筛选问题（数据质量没问题），是构造不出（格式层丢弃）。

### 修法（需要一个小 UDF）

output items 是变长异构数组（`type: reasoning / message / function_call` 混排），SQL 字符串手术不可靠，参照现有 `macaron_anthropic_to_glm_sft` 的模式新增：

```
macaron_responses_to_glm_sft(final_input STRING, final_instructions STRING, final_resp_output STRING)
  -> STRING（JSON 数组，完整 messages；构造不出返回 NULL）
```

UDF 规则：
1. `final_input` 已是 chat messages 数组（dwd 归一过）→ 原样作为基底；纯文本 input → 包成 `[{"role":"user","content":...}]`
2. `final_instructions` 非空 → 头部插 system 消息
3. `final_resp_output` 逐项转**一条** assistant 消息：`reasoning`→`reasoning_content`；`message`（output_text）→`content`；`function_call`→`tool_calls`
4. **碎片检测（格式层丢弃）**：input 消息数与 session 轮数严重失配（如 input 只有 1 条 user 但 n_turns>10）→ 返回 NULL（续话客户端，构造不出完整对话）
5. 输入全空/非法 → 返回 NULL

ETL 改动：

```sql
responses_sessions AS (
  SELECT
    ...,
    MAX_BY(r.req_input, r.ts) AS final_input,
    MAX_BY(r.req_instructions, r.ts) AS final_instructions,
    MAX_BY(r.req_tools, CASE WHEN <tools 有效> THEN r.ts END) AS final_tools,   -- 同 P0-1 条件化
    MAX_BY(r.req_reasoning_effort, r.ts) AS reasoning_effort,
    MAX_BY(r.resp_output, r.ts) AS final_resp_output,
    COUNT(*) AS session_n_turns,   -- [新增] 供 UDF 碎片检测
    ...
  FROM dwd_macaron_relay_stack_responses_total_day r
  WHERE r.statis_day = '${statis_day}'
    AND r.resp_output IS NOT NULL
    AND r.resp_output != '' AND r.resp_output != '\\N'    -- [新增]
  GROUP BY r.session_id
)

-- INSERT：
  JSON_PARSE(
    macaron_responses_to_glm_sft(final_input, final_instructions, final_resp_output)
  ) AS messages,
  -- tools / chat_template_kwargs / WHERE 同 chat 分支修法
  -- WHERE 加：UDF 返回非 NULL（碎片/空 input 丢弃）
```

---

## 5. P1-3 / P1-4a / P1-5：格式层小改

### P1-3 reasoning_effort 值域【格式】

| 位置 | 现状 | 修法 |
|---|---|---|
| chat/responses | `COALESCE(reasoning_effort,'medium')` | `CASE WHEN effort IN ('high','max') THEN effort ELSE 'max' END` |
| messages | `CASE WHEN thinking_type='adaptive' THEN 'max' ELSE 'medium' END` | 同上（统一） |

与本地 `to_glm52_sft.py` 强制 max 对齐；保留合法原值（high/max），非法归 max。

### P1-4a tools 空数组【格式】+ 归因实测【筛选】

**格式侧**：`ELSE JSON_ARRAY()` → `ELSE NULL`（messages 分支：`JSON_PARSE(NULLIF(macaron_tools_to_openai_format(...), '[]'))`）。

**归因实测（0830 快照 44,395 条空 tools）**：

| source | n_tool_use | 条数 | 含义 |
|---|---|---|---|
| chat_completions | 0 | 43,313 | 整个 session 无工具调用（纯对话） |
| messages | 0 | 708 | 同上 |
| chat/responses | null | 369 | 响应解析无工具信息 |
| messages | 209 / 18 | **2** | 唯二例外：用过工具但最后一次请求没带 tools |

**99.4% 是本来就没带工具的纯文本对话**——记录本身结构合法（修完 P1-4a 后 tools 省略键），只是**没有 agentic 训练价值，归筛选层**（18 条款 2.x 任务类型 / 3.x 执行真实性淘汰）。chat_completions 源 66% 是这种流量（messages 源仅 23%，因为 Anthropic 协议客户端多为 agentic）。

那 2 条例外（n_tool_use 209/18 但 tools 空）是 `MAX_BY(req_tools, ts)` 取最后一次请求的边缘 case——P0-1 修法里已把 tools 的 MAX_BY 条件化（取最后一次**带 tools** 的请求），归格式层修复。

### P1-5 空 messages 记录【格式层丢弃】

**修法**：分支 WHERE 丢弃（见 §3/§4），不产出废记录。

**6 条明细与归因（0830 实测）**：

| source | session_id | biz_date | n_turns | 归因 |
|---|---|---|---|---|
| responses | 01a00759-…95a3 | 20260830 | **75** | req_input 空 → 续话客户端 |
| responses | 01a01e6d-…690a | 20260829 | **288** | 同上 |
| responses | 01a04036-…d963 | 20260827 | 33 | 同上（0827 就发现的那条） |
| responses | 01a04487-…9698 | 20260828 | **212** | 同上 |
| chat | 1b0aa417378c8b93 | 20260828 | 12 | req_messages 空/非法（session_id 是 16 位 hex 非 UUID，另一类接入端；疑 body 截断/畸形，待确认） |
| chat | 25ff2bfb720781b7 | 20260829 | 4 | 同上 |

responses 4 条全是大轮数 session + 空 input → **previous_response_id 续话**（input 只带增量，MAX_BY 拼不出全量历史）→ UDF 碎片检测丢弃（§4 规则 4）。

---

## 6. 质量筛选层（Layer 2）规格

**输入**：格式修复后的 `app_macaron_relay_stack_glm_day`（结构合法全量）。
**输出**：新表 `app_macaron_relay_stack_glm_day_qualified`（18 条款 pass）。

**淘汰清单（全部归筛选层）**：

| 淘汰项 | 条款 | 备注 |
|---|---|---|
| 无工具纯对话（99.4% 的 tools 空记录） | 2.x 任务类型 / 3.x 执行真实性 | 格式层修完是 tools 省略键的合法记录 |
| 单轮/闲聊 | 2.x | |
| 未闭环（assistant tool_call 结尾无结果） | 4.x 会话闭环 | P0-1 修完后仍存在的尾部 case |
| cron/batch eval | 6.x | |
| 跨 session 重复 | 5.x | |
| signature 检查 7.1/7.3 | **豁免**（glm 口径） | glm 没签名不是缺陷 |

**实现：UDAF + 三源字段适配**（不能直接抄 opus 那段，两个原因：① UDAF 入参是 messages 源字段（`req_messages/resp_content/resp_stop_reason`），chat 源是 OpenAI 格式（`resp_choices/resp_finish_reason`）对不上；② opus 口径含 7.1/7.3 signature，glm 豁免）：

```sql
macaron_session_quality_check(
  req_messages, resp_content, resp_stop_reason, req_tools,
  reasoning_signature, session_id, ts,
  'glm'   -- [新增] 'opus'=全18条(默认) / 'glm'=豁免7.1/7.3
) = 'pass'
```

chat 源适配：`resp_content ← GET_JSON_OBJECT(resp_choices,'$[0].message.content')`、`resp_stop_reason ← resp_finish_reason`、`reasoning_signature`（dwd 有此列，glm 模式不检查）。

**两层串联（session_id join）**：

```sql
-- Layer 2 ETL 骨架
INSERT OVERWRITE TABLE app_macaron_relay_stack_glm_day_qualified PARTITION (statis_day='${statis_day}')
SELECT g.* FROM app_macaron_relay_stack_glm_day g
INNER JOIN qualified_sessions q ON g.session_id = q.session_id   -- qualified_sessions：三源 UDAF pass 的 session_id
WHERE g.statis_day = '${statis_day}'
```

**注意**：UDAF 逐 session 聚合，跨 session 去重（5.x）能否覆盖需与 UDF 实现确认（opus 线同款在用，对齐它即可）。**方案 B（务实）**：质量筛选留本地流水线（`msh_supplier_selftest.py`）跑——短期不增加数仓工作量，两层职责不变。

---

## 7. 交付层：导出剥离（不改表）

表的 9 列（session_id/request_id/biz_date/model/actual_model + 四键）**保留不动**（查询/排重有用），导出训练文件时剥成四键。已验证可用的导出 SQL：

```sql
SELECT CONCAT(
  '{"messages":', JSON_FORMAT(messages),
  CASE WHEN tools IS NOT NULL THEN CONCAT(',"tools":', JSON_FORMAT(tools)) ELSE '' END,  -- NULL 省键
  ',"chat_template_kwargs":', JSON_FORMAT(chat_template_kwargs),
  ',"meta":', JSON_FORMAT(meta), '}'
) AS line
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day_qualified
WHERE statis_day = '${statis_day}'
```

**meta 无敏感字段**（已核对：session_id/request_id/model/actual_model/n_*/source，无 signature/trace_id/user_id）。

---

## 8. 验收标准（分两层）

**格式层验收**（修完 Layer 1 后，跑在 `app_glm_day` 上）：

```sql
-- V1 闭环率：三源 last_role=assistant 都应接近 100%（messages 源基线 96.8%）
SELECT GET_JSON_OBJECT(JSON_FORMAT(meta),'$.source') AS src,
       GET_JSON_OBJECT(JSON_FORMAT(messages),
         CONCAT('$[', CAST(JSON_LENGTH(messages)-1 AS STRING), '].role')) AS last_role,
       COUNT(*) AS n
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day
WHERE statis_day = '<最新分区>' GROUP BY 1, 2 ORDER BY n DESC;

-- V2 effort 合法率：eff ∉ {high,max} 应为 0
SELECT GET_JSON_OBJECT(JSON_FORMAT(chat_template_kwargs),'$.reasoning_effort') AS eff, COUNT(*) AS n
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day WHERE statis_day='<最新分区>' GROUP BY 1;

-- V3 空 tools 数组 / 空 messages 应为 0（NULL 的 tools 不算空数组）
SELECT SUM(CASE WHEN JSON_FORMAT(tools)='[]' THEN 1 ELSE 0 END) AS empty_tools,
       SUM(CASE WHEN JSON_LENGTH(messages)=0 OR JSON_LENGTH(messages) IS NULL THEN 1 ELSE 0 END) AS empty_msgs
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day WHERE statis_day='<最新分区>';

-- V4 行数对账：chat 源下降量 = 无有效应答的 session 数，需可解释
```

**V5 本地 L0-L6 抽检**（格式层最终验收）：导出 50 条过 `validate_glm52_sft.py --max-seq-len 98304`，预期 reasoning_effort/tools 空数组/extra 字段/空 messages 四类 FATAL 归零。

**筛选层验收**（Layer 2 后）：`qualified` 表行数 / 全量表行数 = 通过率（glm 口径）；无 tools 记录为 0；未闭环为 0；对照本地 `msh_supplier_selftest.py` 抽样一致。

---

## 9. 不修项（已确认）

| 项 | 结论 |
|---|---|
| 顶层 9 列 | 保留，交付层剥离（见第 7 节） |
| meta 内容 | 无敏感字段，L0-L6 不校验 meta 内容，不动 |
| messages 源 messages 构造 | 96.8% 闭环，UDF 工作正常，不动（只改 effort/tools 两行） |
| opus_day 的脱敏问题 | 另一张表，另行处理 |
| session_n_messages 等统计口径 | meta 信息性字段，不影响训练，不动 |
| 无工具纯对话是否留在格式层全量表 | **留**（tools 省略键的合法记录），筛选层淘汰 |

## 10. 建议实施顺序（对应两层）

1. **Phase 1 = 格式修复层**（一次 ETL 改动）：P0-1 chat 分支（CTE+INSERT）+ P0-2 responses UDF + P1-3/4a/5 三处小改 → 产出结构合法全量表，跑格式层验收 V1-V5
2. **Phase 2 = 质量筛选层**：UDAF 加 glm 模式 + 三源适配 + `_qualified` 表；或方案 B 留本地跑
3. 每层独立验收、独立迭代；两层产出都可查（对照排查同 opus 的 dwd vs app_opus_day 关系）

---

*证据存档：source×last_role 矩阵、effort 分布、空数组统计、tools 空值归因（n_tool_use 交叉）、空 messages 6 条明细、stg response_body 结构采样、dwd/app ETL 源文件（etl/app/app_macaron_relay_stack_glm_day.sql、etl/dwd/dwd_macaron_relay_stack_chat_completions_total_day.sql）。基线分区 20260830，v2 生成于 2026-08-31。*
