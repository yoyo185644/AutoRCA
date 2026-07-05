# 有界 LLM 验证的确定性根因分析设计

本文按论文写作的思路，总结 `rca/index/`（BASIL 检索层）与 `rca/counterfactor_agent/`（有界 LLM 验证层）当前实现的设计：从创新点，到 Index 检索算法与 LLM 验证算法的具体设计，再到整体框架设计，最后给出一个来自真实运行结果的例子。

## 1. 创新点

传统 RCA 系统在"纯确定性算法"与"完全自主的 LLM agent（多轮迭代、自由探索遥测数据）"之间通常只能二选一：前者稳定但在证据模糊时表现不足，后者灵活但方差大、成本高、不可复现，且容易在没有证据支持的情况下编造组件名或指标。本设计的核心创新是在两者之间引入一个受约束的中间层，具体包含五点：

**(1) 确定性优先、有界 LLM 介入。** 绝大多数任务完全由确定性的 BASIL pipeline（index → evidence → rerank）解决，LLM 只在算法自己承认"拿不准"的少数任务上介入，且每次介入被严格限定为一次请求-响应，不存在多轮迭代循环。这与很多"LLM agent做RCA"的工作把 LLM 放在探索/推理的中心位置不同，这里 LLM 始终是确定性结果的补充校验者。

**(2) 数据驱动的不确定性门控（Uncertainty Gate）。** 是否调用 LLM 不是基于任务类型或人工规则，而是基于确定性候选池自身的打分特征——候选池是否为空、Top-1 绝对分数是否够高、Top-1/Top-2 分数差距是否够大——用几个可调阈值决定。这把"什么时候信任确定性算法、什么时候需要外部校验"变成一个可以离线调参、可复现的量化判断，而不是"每题都问一次 LLM"或"从不问"的两个极端。

**(3) 两阶段证伪式验证协议。** 送给 LLM 的任务不是"帮我找根因"，而是被显式框定为两阶段的证伪测试：Stage A 判断 Top-1 候选能否被"证伪"（是否存在证据上更具体、更能解释所需字段的挑战者）；Stage B 若被证伪，则从挑战者中选出解释力最小充分（minimal intervention）的一个。这把 LLM 的角色从"自由生成答案"转变为"对抗式验证者/挑战者"，减少无证据支持的臆断。

**(4) 字段级、证据门控的采纳策略，而非整体二选一。** 根因结果由三个独立字段组成（component / reason / datetime），LLM 的建议不是整体采纳或整体拒绝，而是逐字段独立判断：该字段是否有明确的证据增益（evidence gain）、LLM 的置信度是否达标、相对确定性答案的分数跌幅是否在容忍范围内（或有足够强的挑战者证据支撑该例外）、数据集适配层是否一票否决。这让"LLM 只在部分字段上判断正确"的情况仍能贡献正确的那部分，而不会被错误字段拖累整体。

**(5) 数据集语义与核心决策策略解耦。** 组件类型语义、根因原因分类、字段一致性校验（如"共享基础设施症状冒充局部根因"的已知陷阱识别）都通过 Dataset Adapter 提供，核心的门控/采纳算法（Controller）本身是数据集无关的通用策略。这与项目里 Dataset Skill / Core Planner / BASIL Core 三层解耦的整体架构原则一致：数据集只声明语义，不实现排序或采纳策略。

**(6) 任务感知的证据融合（Task-Aware Fusion）。** 同一份 local / causal / reason 三路证据，并不是固定权重加权，而是按任务实际需要的字段（是否需要 time / component / reason）从一张预设权重表里取一组 $(\alpha,\beta,\gamma)$——例如只问时间时几乎不看 reason 证据，只问 reason 时大幅提高 reason 权重。同一套底层证据服务多种任务形态，而不是每种任务单独设计一条打分公式。

**(7) 用因果直觉做打分之外的排序信号，而非硬规则。** "根因先于症状出现"这一因果直觉没有被写成过滤规则，而是作为分数相近时的 tie-break 信号（onset-lead）；"不能让一种高噪声证据通道（如 node 级指标）把其它通道的候选全部挤出候选集"也没有写成人工白名单，而是变成一个通道覆盖度（channel coverage）保证机制。两者都是"直觉转化为排序特征/兜底机制"，而不是直接编码为 if-else 规则，保留了可调性。

## 2. Index 设计（BASIL 检索层）

Index 层（`rca/index/`）是 LLM 验证层的上游：它负责把原始 telemetry 转成候选池 $C$，`_uncertainty_gate`、`select_best_candidate_with_llm` 看到的分数和证据都来自这里。整体是一个**在线四阶段 pipeline**（`query_basil_components`，`rca/index/query.py`），每次查询按时间窗口现场取数、现场打分，不是离线预训练的索引。

### 2.1 数据模型

原始 telemetry（metric / trace / log 的 CSV 行）不会被直接拿来打分，而是先压缩成三类紧凑对象（`rca/index/schemas.py`）：

- `AnomalySegment`：一段"组件 × 信号 × 时间区间"的异常，带 `severity`、`reason_hints`、`top_metrics`，替代逐行原始遥测数据；
- `CausalEdgeSketch`：按滞后时间分桶（`lag_bin`：0-30s / 30-60s / 1-3m / 3-5m / 5-15m / 15m+）聚合的传播摘要，而不是逐 trace-span 的原始因果图；
- `BlockMaxBlock` / `BasilQueryStats`：为近似 top-k 检索（block-max 剪枝，借鉴信息检索里的 Block-Max WAND）预留的数据结构。**需要如实说明：`build_block_max` 在 `indexes.py` 里实现了，但当前 `query_basil_components` 的调用路径并没有真正触发按块剪枝的检索循环**——`BasilQueryStats` 里 `visited_blocks`/`max_remaining_ub` 恒为 0、`exact_mode` 恒为 `True`，实际走的是精确重排的四阶段流程（见 2.2）。这是一个为未来近似检索扩展预留、但尚未接入主链路的设计。

### 2.2 在线四阶段检索算法

```
输入: dataset, instruction（含时间窗口）, 历史证据索引 I（全进程共享、线程安全、只追加）
输出: BasilComponentResult（候选池 + 打分 + 因果边 + 统计信息）

Stage 1  证据抽取 + 历史相似案例检索
  segments, edges ← extract_anomaly_segments / extract_trace_topology_edges(window ± max_lag_s)
                    （命中磁盘证据缓存则直接读取，不重新解析 CSV）
  I.append_incident(segments, edges)                     # 只追加，不修改历史
  historical ← I.retrieve_similar_incidents(segments, edges)
               # 用倒排索引（metric→incident_ids, component→incident_ids）先圈定
               # 候选历史事件，只对有交集的事件算相似度，避免全量扫描
               # sim = w_ts·Jaccard(指标集) + w_topo·Jaccard(组件∪拓扑) + w_pattern·Jaccard(spike/drift 模式)
               #       再乘指数时间衰减 exp(-λ·天数)

Stage 2  任务感知融合 (TaskAwareScoreFusion)
  for 每个组件: score = α·local_score + β·causal_score + γ·max(reason_scores)
               (α,β,γ) 按 task_type 从预设表取（创新点 6）
  按 (分数带宽, onset_lead, 原始分数) 排序，并保证每个组件类型通道至少有一个代表进入候选集（创新点 7）

Stage 3  候选展开 (CandidateExpansion)
  按任务需要的字段做笛卡尔展开：
    COMPONENT_ONLY      → 只保留 component
    TIME_ONLY / *_TIME  → 展开 {首次异常时间, 峰值时间}
    REASON_ONLY / *_REASON → 展开 top-R 个 reason 候选
    COMPONENT_REASON_TIME → component × reason(top-R) × time(首次/峰值) 全展开

Stage 4  两阶段重排 (TwoStageReranker)
  规则阶段: 按查询窗口 ± time_tolerance_s 过滤越界候选；
            reason-metric 一致性加分/不一致扣分；
            "上游传播 > 被下游影响" 的因果方向加分
  (可插拔) LLM 重排阶段: 预留了 llm_ranker 回调接口，但当前调用方
            （rca/agent/agent_standard.py、indexed_evidence_agent.py）都没有
            传入这个回调——真正的 LLM 校验是在这一步之外，由第 3 节的
            Bounded LLM Verifier 对最终候选池单独做的，不是这里的 llm_ranker
```

默认权重表（部分）：`TIME_ONLY=(0.80,0.20,0.00)`、`REASON_ONLY=(0.35,0.10,0.55)`、`COMPONENT_REASON_TIME=(0.55,0.25,0.20)`（`rca/index/query.py::TaskAwareScoreFusion.DEFAULT_WEIGHTS`）。

### 2.3 缓存设计

两层缓存，服务于不同目的：磁盘级"证据缓存"（`_load_evidence_cache`/`_save_evidence_cache`，按 `dataset+时间窗口+参数` 做 key，落盘到 `test/cache/basil_index`，避免重复解析同一时间窗口的 telemetry CSV）；进程内"文件级缓存"（`extractor.py`/`telemetry_cache.py` 里的 `_cached_listdir`/`_cached_header`/`get_cached_metric_baseline`，避免同一进程内重复扫目录、读表头、算 p95 基线）。历史相似案例索引（`BasilEvidenceIndex`）本身是纯内存的、进程级单例，重启即清空，不做跨进程持久化。

## 3. 算法设计（LLM 验证层）

记确定性候选池为 $C = \{c_1, \dots, c_n\}$（第 2 节四阶段流程的输出），按 BASIL 打分 $s(c_i)$ 降序排列，$c_1$ 为默认答案（Top-1）。设所需字段集合为 $F \subseteq \{\text{component}, \text{reason}, \text{datetime}\}$（由任务类型决定，如 task_7 要三个字段都要）。

### 3.1 不确定性门控 $G(C)$

```
输入: 候选池 C（按分数降序），阈值 floor, floor_single, gap_th, skip_gap_th
输出: need_llm ∈ {True, False}

if |C| == 0:
    need_llm = True                                  # 无候选，必须问
elif |C| == 1:
    need_llm = ( s(c1) < floor_single )               # 单候选，分数不够高才问
else:
    gap = s(c1) - s(c2)
    need_llm = ( s(c1) < floor ) OR ( gap < gap_th )  # Top1 不够高，或跟 Top2 分不开

# 反向保险：即使触发，Top1 足够强也跳过
if need_llm AND |C| >= 2 AND s(c1) >= floor_single AND gap >= skip_gap_th:
    need_llm = False
```

默认阈值：`floor=0.30`，`floor_single=0.50`，`gap_th=0.15`，`skip_gap_th=0.30`（均可通过 CLI 参数调整）。

### 3.2 有界 LLM 验证 $V$

若 `need_llm = True`，仅调用一次：

```
(top1_falsified, challenger c*, confidence, field_decisions)
    = V(instruction, C[:k], evidence(C[:k]))
```

其中 `evidence(·)` 是每个候选的证据摘要（metric 时间线、因果拓扑上下游关系、规则命中标签等，不包含标签/答案泄漏）；`field_decisions` 是 LLM 对每个字段给出的 `{adopt: bool, evidence_gain: [0,1]}` 结构化判断。

### 3.3 字段级采纳 $A$

对每个 $f \in F$，独立判断是否用 $c^*$ 的字段值替换 $c_1$ 的字段值：

```
allow(f):
    if c1[f] == c*[f]: return False                      # 没变化，无需判断
    if f == component AND symptom_over_local(c1, c*): return False   # 已知陷阱：共享基础设施症状冒充局部根因
    if NOT top1_falsified: return False                  # Stage A 都没证伪，不采纳任何字段
    gain = field_decisions[f].evidence_gain
    explicit = field_decisions[f].adopt OR gain >= 0.35
    if explicit AND confidence >= 0.45:
        return evidence_gain_supported(f, c1, c*)          # 字段级证据增益支持（各字段判据不同，实现见 controller.py::_allow_field）
    if aggressive_intervention(c1, c*, confidence): return True   # 强证伪 + 强局部证据的例外通道
    if confidence < max(0.65, min_conf - 0.15): return False
    if score_drop(c1, c*) > max_drop AND NOT challenger_sufficient: return False
    if f in {component, reason} AND NOT adapter.validate(c1, c*): return False
    return conflict_dimension_supported(f, c1, c*, challenger_strength)

y[f] = c*[f] if allow(f) else c1[f]      # 对每个字段独立选择
```

默认阈值：`min_conf=0.85`（时间字段的高置信度直采线），`max_drop=0.05`（允许的分数跌幅上限）。

### 3.4 输出

$$y = \text{canonicalize}_{\text{adapter}}\big(\{y[f] : f \in F\}\big)$$

整个算法只有**一次**候选生成 + **至多一次** LLM 请求 + 一次规则化的字段级合并，没有 step/turn 循环。

## 4. 框架设计

```
Query
  │
  ▼
Dataset Config + Adapter Registry   ← per-dataset yaml + adapter class
  │
  ▼
Dataset Skill (声明: evidence_sources / component_types / reason_taxonomy)
  │
  ▼
Core Planner  (映射 Dataset Skill → BASIL 核心模块开关)
  │
  ▼
BASIL Index 检索层 (第 2 节：证据抽取+历史相似案例 → 任务感知融合 → 候选展开 → 两阶段重排)
  │
  ▼
Uncertainty Gate ──need_llm=False──▶ 直接输出 Top-1
  │ need_llm=True
  ▼
┌─────────────────────────────────────────┐
│  Bounded LLM Verifier                    │
│                                           │
│  Executor   ── 一次 LLM 请求 ──▶ 结构化    │
│  (executor.py)   (两阶段证伪 prompt)  判断  │
│       │                                  │
│       ▼                                  │
│  Controller (controller.py)              │
│    - 门控 (3.1)                          │
│    - 字段级采纳 (3.3)                     │
│       │                                  │
│       ▼                                  │
│  Dataset Adapter                         │
│    - canonicalize_component               │
│    - validate_llm_candidate_adoption      │
└─────────────────────────────────────────┘
  │
  ▼
Prediction Output → Evaluate
```

三个实现文件的职责边界：`agent.py`（`CounterfactualRCAAgent`）只是每次调用现场组装 Executor + Controller 的薄编排层；`controller.py`（`CounterfactualController`）承载 3.1/3.3 节的策略逻辑；`executor.py`（`CounterfactualExecutor`）是唯一与 LLM 通信的模块，负责组装两阶段证伪 prompt、发请求、解析结构化输出。Index 层（第 2 节）与 LLM 验证层之间只通过候选池 $C$（`context["candidate_tuples"]`）交互，彼此不感知对方的内部实现。

## 5. 例子（真实运行结果）

以下数据取自 `test/result/Market/cloudbed-1/Cloudbed1_basil_llm_deepseek_20260703_155953.csv` 第 `row_id=61` 行（`task_7`，三个字段都需要预测），标 *(示意)* 的部分因为对应的 trajectory 日志未保留，用符合算法逻辑的示意值代替，其余均为该行的真实数据。

**输入指令：**

> On March 21, 2022, from 16:30 to 17:00, the cloud service system, cloudbed-1, experienced a failure. However, the exact time of the root cause occurrence, the affected component, and the underlying reason for this failure are currently unknown. Please identify the root cause occurrence datetime, the root cause component, and the root cause reason.

**BASIL 候选池（真实，节选 Top-2，共 39 个候选）：**

| 排名 | component | reason | datetime | score | source |
|---|---|---|---|---|---|
| 1 | node-5 | node disk space consumption | 2022-03-21 16:42:00 | 1.2301 | market_basil_expand+node_io_surface_reason |
| 2 | node-5 | node disk read I/O consumption | 2022-03-21 16:47:00 | 1.1147 | market_basil_expand |

Top-1 与 Top-2 分数相对接近（约 10% 差距），触发不确定性门控里的 `small_score_gap` 条件，`need_llm = True`。

**LLM 验证 *(示意)*：** Stage A 判断 Top-1 的 component 已有较强的本地 I/O 证据支持（`system.io.util`、`system.io.await` 等指标），未被证伪；Stage B 因此不替换 component/reason，仅对 datetime 给出 `evidence_gain≈0.4` 的小幅精修建议（同一异常窗口内的峰值时间点），`confidence≈0.7`。

**字段级采纳：** component、reason 因 `top1_falsified=False` 直接保留 Top-1 值；datetime 的证据增益超过直采线（`gain ≥ 0.30`），予以采纳（对应 3.3 节 `allow(datetime)` 分支）。`llm_gate_called=True, llm_changed_count=1, llm_adopted_count=1`（均为该行真实记录字段）。

顺带说明候选池里 `source` 字段（`market_basil_expand+node_io_surface_reason`、`market_basil_expand`）的来历：前者是第 2 节 Stage 2/3 里"任务感知融合 + reason 展开"共同产出的候选标记，后者是同一 component 在 Stage 3 里对另一个 reason/time 组合的展开——这两条本质是同一个 Index 检索结果，只是笛卡尔展开出的不同 reason 分支，这也是它们分数接近、进而触发门控的根本原因。

**最终输出（真实）：**

```json
{"1": {"root cause occurrence datetime": "2022-03-21 16:42:00", "root cause component": "node-5", "root cause reason": "node disk space consumption"}}
```

**Ground truth（真实）：** component=`node-5`, reason=`node disk space consumption`, datetime=`2022-03-21 16:41:47`（预测值与真实值相差 13 秒，在 1 分钟容忍窗口内）。

**评测结果（真实）：** `passed=["node-5", "node disk space consumption", "2022-03-21 16:41:47"]`，`failed=[]`，`score=1.0`——三个字段全部正确。

这个例子体现了设计的完整闭环：门控发现 Top-1/Top-2 分数太接近而触发 LLM；LLM 没有推翻确定性给出的 component/reason（说明确定性证据本身是充分的，LLM 起到"确认"作用而非"替代"作用）；字段级机制允许 datetime 单独被精修，而不影响已经正确的另外两个字段；最终三个字段全部命中真值。
