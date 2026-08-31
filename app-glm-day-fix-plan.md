# app_macaron_relay_stack_glm_day 修复方案（评审稿 v1）

> **一句话结论**：这张表目前是"**未闭环、未对齐、未筛选**"的中间产物——96% 的记录缺模型回复、格式契约与训练侧 L0-L6 不对齐、没有达标筛选。共 6 个问题（2 个 P0、3 个 P1、1 个 P2），本文档给出每个问题的证据、根因和具体修法。

---

## 0. 基线与口径说明

**分区语义**：`statis_day` 是**全量快照**（非日增量），数字随快照累积增长：

| statis_day | 总行数 |
|---|---|
| 20260827（已过期） | 4,486 |
| 20260828 | 49,876 |
| 20260829 | 66,016 |
| **20260830（最新基线）** | **69,330** |

**LIFECYCLE 3 天**：验证请用最新分区（今天的 0827 已被回收，本文所有实测数字基于 0830 快照）。

**0830 快照按 source 实测**（69,330 条 = chat 65,691 + messages 3,093 + responses 546）：

| source | 行数 | 空 messages | tools=[] | **assistant 结尾** | effort ∈{high,max} |
|---|---|---|---|---|---|
| chat_completions | 65,691 | 2 | 43,557 (66%) | **19 (0.03%)** | **0 (0%)** |
| messages | 3,093 | 0 | 710 (23%) | 2,995 (96.8%) | 2,908 (94%) |
| responses | 546 | 4 | 128 (23%) | **0 (0%)** | 239 (44%) |
| **合计** | 69,330 | 6 | 44,395 (64%) | **3,014 (4.3%)** | 3,147 (4.5%) |

---

## 1. 问题清单（总览）

| # | 问题 | 根因（ETL 现状） | 影响（0830 实测） | 优先级 |
|---|---|---|---|---|
| P0-1 | chat_completions 源缺 assistant 回复 | CTE 没取 `resp_choices`，messages 直接用 `req_messages` | 65,691 条只有 19 条闭环（0.03%） | **P0** |
| P0-2 | responses 源缺 assistant 回复 | `final_resp_output` 取了但 INSERT 没用（注释写了没做） | 546 条 0 条闭环 | **P0** |
| P1-3 | reasoning_effort 值域非法 | `COALESCE(reasoning_effort, 'medium')`；训练契约 `Literal["high","max"]` | chat 源 100% 非法（全是 medium 等） | P1 |
| P1-4 | tools 空数组 | `ELSE JSON_ARRAY()` 兜底；校验器要求"空则省略键" | 44,395 条 (64%) | P1 |
| P1-5 | 空 messages 记录 | `ELSE JSON_ARRAY()` 兜底产出空记录 | 6 条 | P1 |
| P2-6 | 无 18 条款达标筛选 | 无 qualified_sessions CTE（opus_day 有，glm_day 没有） | 全量含单轮/无工具/不闭环/闲聊 | P2 |
| 附-7 | 顶层 9 列 vs 训练 4 键 | 表设计（查询用辅助列） | 导出层剥离，**不改表** | 备注 |

**下游契约**（训练侧 L0-L6 校验器，实测依据）：
- 顶层只认 `messages/tools/chat_template_kwargs/meta` 四键，`extra="forbid"`（多一个字段直接 FATAL，实测 `Extra inputs are not permitted`）
- `reasoning_effort: Literal["high","max"]`（实测 medium 报 `Input should be 'high' or 'max'`）
- `tools` 为空数组直接 FATAL（`_reject_empty_tools`），省略键则 OK
- `messages` 至少 1 条（`min_length=1`）
- 本地转换器 `to_glm52_sft.py` 的既有约定：**强制 `reasoning_effort="max"`、空 tools 省略键**——修复向它对齐

---

## 2. P0-1：chat_completions 源缺 assistant 回复

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

**为什么缺**：chat completions 协议客户端每次重发完整历史（`req_messages` 含之前所有轮次），但**本轮模型的回答在 `resp_choices` 里**。ETL 只取了请求侧，响应侧取都没取 → 训练样本"有问无答"（24,417 条 user 结尾 + 18,440 条 tool 结尾，到工具结果就断）。

**关键证据（stg 原始 response_body 实测）**：`choices[0].message` 就是标准 GLM assistant 格式，可直接拼接：

```json
{"id":"...","object":"chat.completion","model":"glm-5.2","created":...,
 "choices":[{"index":0,"message":{
   "role":"assistant",
   "content":"Let me check how many total custom class names...",
   "reasoning_content":"Found it! The exclusion regex is:..."}}]}
```

### 修法（SQL-only，无需新 UDF）

**① CTE 加取 resp_choices + 行级过滤只保留有效应答的请求**（这同时修复 P1-5 的 chat 部分）：

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
    MAX_BY(r.req_tools, r.ts) AS final_tools,
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

> 行级过滤的语义：`MAX_BY(req_messages, ts)` 变成"最后一次**成功应答**的请求"。若 session 最后一次请求失败，取到的是最后一次成功交换（历史+应答仍完整闭环）；整个 session 全失败则自然消失（本来就不可训练）。副作用（正向）：`COUNT(*)` 变成有效轮数。

**② INSERT 的 messages 拼回应答**（同时修 P1-4/P1-5）：

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
    ELSE NULL                                              -- [修复 P1-4] 空数组 → NULL
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
- `message.content` 为 null（纯 tool_calls 应答）合法，`tool_calls` 字段原样保留，GLM 格式兼容
- session 结束在 tool_call（客户端没续）→ 拼后以 assistant+tool_calls 结尾、无 tool result，这类由 P2 的达标筛选兜底淘汰
- **NUL 字节风险**：dwd ETL 注释提过 reasoning 字段需在 parse UDF 里清洗 NUL（ODPS-0121095）。若 `resp_choices.message.reasoning_content` 含 NUL，需在 chat_completions parse UDF 里同样清洗（校验器 L2 有 `\x00` 保留标记检查，本地 L0-L6 会抓）
- 多 choice（n>1）取 `[0]`

---

## 3. P0-2：responses 源缺 assistant 回复

### 现状

```sql
-- CTE 取了 final_resp_output，但 INSERT 里完全没用：
-- 注释写着 "[system_msg] + input_items + [assistant_response]"
-- 实际代码只拼了 instructions(可选) + final_input，assistant_response 从未出现
CASE
  WHEN final_input IS NOT NULL AND ... THEN
    CASE WHEN final_instructions IS NOT NULL ...
    THEN JSON_PARSE(CONCAT('[{"role":"system","content":', TO_JSON(final_instructions), '},', SUBSTR(final_input, 2)))
    ELSE JSON_PARSE(final_input)
    END
  ELSE JSON_ARRAY()
END AS messages,
```

**证据**：0830 快照 546 条 responses 记录，assistant 结尾 **0 条**；4 条 messages 空（input 非法时 `JSON_ARRAY()` 兜底）。

**另一证据**：dwd 的 `req_input` 已被 parse UDF 归一成 chat messages 格式（实测 app 表里 responses 记录的 messages 是 `[{"role":"system","content":"...Codex CLI..."}]` 这种标准格式，不是 raw input items），所以 input 侧不用动，**只需补 output → assistant 的转换**。

### 修法（需要一个小 UDF）

output items 是变长异构数组（`type: reasoning / message / function_call` 混排），SQL 字符串手术不可靠，建议参照现有 `macaron_anthropic_to_glm_sft` 的模式新增一个 UDF：

```
macaron_responses_to_glm_sft(final_input STRING, final_instructions STRING, final_resp_output STRING)
  -> STRING（JSON 数组，完整 messages）
```

UDF 规则：
1. `final_input` 已是 chat messages 数组（dwd 归一过）→ 原样作为基底；若是纯文本（input 传字符串的客户端）→ 包成 `[{"role":"user","content":...}]`
2. `final_instructions` 非空 → 头部插 system 消息
3. `final_resp_output` 逐项转**一条** assistant 消息：
   - `type=reasoning` → `reasoning_content`
   - `type=message`（content 里 `output_text`）→ `content`
   - `type=function_call` → `tool_calls[{id: call_id, type: function, function: {name, arguments}}]`
4. 输入全空/非法 → 返回 `NULL`（ETL WHERE 丢弃）

ETL 改动：

```sql
responses_sessions AS (
  SELECT
    ...,
    MAX_BY(r.req_input, r.ts) AS final_input,
    MAX_BY(r.req_instructions, r.ts) AS final_instructions,
    MAX_BY(r.req_tools, r.ts) AS final_tools,
    MAX_BY(r.req_reasoning_effort, r.ts) AS reasoning_effort,
    MAX_BY(r.resp_output, r.ts) AS final_resp_output,
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
```

> 注：responses 源若客户端用 `previous_response_id`（服务端续话），`req_input` 只有增量，MAX_BY 拼不出全量历史——这类 session 建议在 UDF 里检测（如 input 只有 1 条 user 但 n_input_items 与 session 轮数不符）返回 NULL 丢弃，或接受现状由 P2 筛选兜底。

---

## 4. P1-3 / P1-4 / P1-5：三处小改（前面分支修法已含）

| 问题 | 现状 | 修法 | 位置 |
|---|---|---|---|
| P1-3 effort | chat/responses: `COALESCE(reasoning_effort,'medium')`；messages: `CASE WHEN thinking_type='adaptive' THEN 'max' ELSE 'medium' END` | 统一：`CASE WHEN effort IN ('high','max') THEN effort ELSE 'max' END`（保留合法原值，非法归 max，与本地 `to_glm52_sft.py` 强制 max 对齐） | 三处 chat_template_kwargs |
| P1-4 tools | `ELSE JSON_ARRAY()` | `ELSE NULL`（messages 分支：`JSON_PARSE(NULLIF(macaron_tools_to_openai_format(...), '[]'))`） | 三处 tools |
| P1-5 空 messages | `ELSE JSON_ARRAY()` 兜底产出空记录 | 分支 WHERE 丢弃（见上两节） | chat/responses |

messages 分支（UDF 转换的那个）只需改两行：

```sql
  JSON_PARSE(NULLIF(macaron_tools_to_openai_format(final_tools, 'anthropic'), '[]')) AS tools,  -- '[]' → NULL
  JSON_OBJECT('reasoning_effort', 'max') AS chat_template_kwargs,  -- 或 CASE 保留 high/max
```

> messages 分支本身是健康的（96.8% 闭环），不用动 messages 构造。

---

## 5. P2-6：18 条款达标筛选（建议二期）

**现状**：`app_opus_day` 有 `qualified_sessions` CTE（UDAF `macaron_session_quality_check`，opus 口径含 signature），`app_glm_day` 没有。

**为什么不能直接抄 opus 那段**：
1. UDAF 入参是 messages(Anthropic) 源的字段（`req_messages/resp_content/resp_stop_reason/reasoning_signature`），chat 源是 OpenAI 格式（`resp_choices/resp_finish_reason`），字段和格式都对不上
2. opus 口径含 7.1/7.3 signature 检查，**glm 口径豁免**（glm 没签名不是缺陷），照抄会把大量合格 glm session 误杀

**方案 A（推荐，二期做）**：UDAF 加 `mode` 参数 + 三源字段适配

```sql
macaron_session_quality_check(
  req_messages, resp_content, resp_stop_reason, req_tools,
  reasoning_signature, session_id, ts,
  'glm'   -- [新增] 'opus'=全18条(默认) / 'glm'=豁免7.1/7.3
) = 'pass'
```

三源适配（chat 源）：

| UDAF 入参 | chat_completions 适配 |
|---|---|
| req_messages | `req_messages`（OpenAI 格式，UDF 需兼容双格式） |
| resp_content | `GET_JSON_OBJECT(resp_choices, '$[0].message.content')` |
| resp_stop_reason | `resp_finish_reason` |
| req_tools | `req_tools` |
| reasoning_signature | `reasoning_signature`（dwd 有此列；glm 模式下不检查） |

**方案 B（务实，先跑通）**：MC 只修 P0/P1（数据正确性），18 条款留在本地流水线（`msh_supplier_selftest.py`）跑——你们本地本来就有这步，短期不增加数仓工作量。

**注意**：UDAF 是逐 session 聚合，跨 session 去重（5.x）能否覆盖需与 UDF 实现确认（opus 线同款在用，对齐它即可）。

---

## 6. 附-7：导出层剥离（不改表）

表的 9 列（session_id/request_id/biz_date/model/actual_model + 四键）**保留不动**（查询/排重有用），导出训练文件时剥成四键。已验证可用的导出 SQL：

```sql
SELECT CONCAT(
  '{"messages":', JSON_FORMAT(messages),
  CASE WHEN tools IS NOT NULL THEN CONCAT(',"tools":', JSON_FORMAT(tools)) ELSE '' END,  -- NULL 省键
  ',"chat_template_kwargs":', JSON_FORMAT(chat_template_kwargs),
  ',"meta":', JSON_FORMAT(meta), '}'
) AS line
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day
WHERE statis_day = '${statis_day}'
  AND ... 筛选条件 ...
```

**meta 无敏感字段**（已核对：session_id/request_id/model/actual_model/n_*/source，无 signature/trace_id/user_id），脱敏层面 glm_day 无问题（opus_day 的 meta 保留 signature 是另一张表的事，不在本文档范围）。

---

## 7. 验收标准（修复后跑）

```sql
-- V1 闭环率：三个源的 last_role=assistant 都应接近 100%（messages 源基线 96.8%）
SELECT GET_JSON_OBJECT(JSON_FORMAT(meta),'$.source') AS src,
       GET_JSON_OBJECT(JSON_FORMAT(messages),
         CONCAT('$[', CAST(JSON_LENGTH(messages)-1 AS STRING), '].role')) AS last_role,
       COUNT(*) AS n
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day
WHERE statis_day = '<最新分区>'
GROUP BY 1, 2 ORDER BY n DESC;

-- V2 effort 合法率：eff ∉ {high,max} 应为 0
SELECT GET_JSON_OBJECT(JSON_FORMAT(chat_template_kwargs),'$.reasoning_effort') AS eff, COUNT(*) AS n
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day
WHERE statis_day = '<最新分区>' GROUP BY 1;

-- V3 空 tools / 空 messages 应为 0（NULL 的 tools 不算空数组）
SELECT SUM(CASE WHEN JSON_FORMAT(tools) = '[]' THEN 1 ELSE 0 END) AS empty_tools,
       SUM(CASE WHEN JSON_LENGTH(messages) = 0 OR JSON_LENGTH(messages) IS NULL THEN 1 ELSE 0 END) AS empty_msgs
FROM macaron_relay_stack.app_macaron_relay_stack_glm_day WHERE statis_day = '<最新分区>';

-- V4 行数对账：修复后 chat 源行数会下降（丢弃全失败/空历史 session），下降量 = 无有效应答的 session 数，需可解释
```

**V5 本地 L0-L6 抽检**（训练侧最终验收）：

```bash
# 用第 6 节导出 SQL 拿 N 条 → line 去掉外层引号存 jsonl →
python3 scripts/validate_glm52_sft.py sample.jsonl --max-seq-len 98304
# 预期：reasoning_effort/tools 空数组/extra 字段三类 FATAL 归零
```

---

## 8. 不修项（已确认）

| 项 | 结论 |
|---|---|
| 顶层 9 列 | 保留，导出层剥离（见第 6 节） |
| meta 内容 | 无敏感字段，L0-L6 不校验 meta 内容，不动 |
| messages 源 messages 构造 | 96.8% 闭环，UDF 工作正常，不动 |
| opus_day 的脱敏问题 | 另一张表，另行处理 |
| session_n_messages 等统计口径 | meta 信息性字段，不影响训练，不动 |

## 9. 建议实施顺序

1. **第一批（P0-1 + P1-3/4/5，一次 ETL 改动）**：chat 分支 CTE+INSERT、responses 分支行过滤+WHERE、三处 effort/tools——除 responses 的 UDF 外全是 SQL 改动
2. **第二批（P0-2）**：`macaron_responses_to_glm_sft` UDF 开发 + responses 分支 messages 构造替换
3. **第三批（P2-6，可选）**：UDAF 加 glm 模式 + 三源适配；不做则维持本地自检
4. 每批跑第 7 节验收，V5 用最新分区抽 50 条过 L0-L6

---

*证据存档（本文所有实测 SQL 可在新分区复跑）：source×last_role 矩阵、effort 分布、空数组统计、stg response_body 结构采样、dwd/app ETL 源文件（etl/app/app_macaron_relay_stack_glm_day.sql、etl/dwd/dwd_macaron_relay_stack_chat_completions_total_day.sql）。基线分区 20260830，生成于 2026-08-31。*
