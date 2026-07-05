# OpenRCA

OpenRCA 是一个用于评测大模型（LLMs）在软件运维场景下根因分析（Root Cause Analysis, RCA）能力的基准数据集与代码仓库。输入是一条自然语言故障查询（query），模型/Agent 需要结合大规模遥测数据（metric / trace / log）输出根因要素：

- root cause occurrence datetime
- root cause component
- root cause reason

本仓库也包含一个可复现的 Agent baseline（RCA-agent + Counterfactual agent），通过 Python 工具化数据检索与分析，降低超长上下文带来的负担，提升可扩展性。

## 项目目标

- 提供可复现、可扩展的 RCA benchmark 与 baseline 实现
- 统一执行入口与策略路由：根据 difficulty 自动选择执行路径（默认 auto）
  - easy → fast-solver
  - middle / hard → counterfactual agent
- 保持现有逻辑、输入输出语义与实验脚本习惯不变

## 整体架构

核心是一个“统一执行框架”（Runner + Router + Strategy/Adapter）：

- Runner：执行一次任务求解，屏蔽内部多策略细节
- Router：根据 difficulty / CLI 参数决定策略顺序
- Strategy/Adapter：对已有 fast-solver 与 counterfactual agent 做薄封装，复用旧实现而非重写算法

关键实现文件：

- `rca/execution/framework.py`：通用执行框架（TaskContext / Strategy / Router / UnifiedTaskRunner）
- `rca/agent/agent_standard.py`：面向当前项目的策略实现与默认路由（fast-solver + counterfactual agent）
- `main/evaluate.py`：唯一的评测脚本入口（跑数据集 + 逐行打分），内部调用统一框架

## Difficulty 与执行策略

当前 difficulty 默认按 `task_id` 映射（与原逻辑保持一致）：

- task_1 ~ task_3 → easy
- task_4 ~ task_6 → middle
- task_7 → hard

默认 `--route_mode auto` 下：

- easy：优先 fast-solver，失败则自动回退到 counterfactual agent（与旧行为一致）
- middle/hard：直接走 counterfactual agent（与旧行为一致）

可通过 `--route_mode` 显式控制（不影响旧命令）：

- `auto`：默认（推荐）
- `fast`：easy 优先 fast-solver（仍保留回退）
- `counterfactual`：强制走 counterfactual agent

## 目录结构（关键部分）

```
.
├── main/
│   └── evaluate.py
├── rca/
│   ├── execution/
│   │   └── framework.py
│   ├── agent/
│   │   └── agent_standard.py
│   ├── baseline/
│   │   ├── rca_agent/
│   │   └── counterfactor_agent/
└── README.md
```

## 安装

需要 Python >= 3.10：

```bash
pip install -r requirements.txt
```

数据集下载后放入 `dataset/`（仓库默认为空）。数据目录结构如下：

```text
dataset/
├── {SYSTEM}
│   ├── query.csv
│   ├── record.csv
│   └── telemetry/
│       ├── {DATE}/
│       │   ├── log/
│       │   ├── metric/
│       │   └── trace/
│       └── ...
└── ...
```

其中 `{SYSTEM}` 取值：`Telecom`、`Bank`、`Market`（Market 包含 `Market/cloudbed-1` 与 `Market/cloudbed-2` 两套数据）。

## 常用命令

### 运行 + 评测（唯一入口：main.evaluate）

`main/evaluate.py` 现在是唯一的测试入口：跑数据集里的每条 query、用 `evaluate()` 逐行打分、写出结果 CSV，一步到位。

```bash
python3 -m main.evaluate --dataset Bank
python3 -m main.evaluate --dataset Market/cloudbed-1
python3 -m main.evaluate --dataset Market/cloudbed-2
python3 -m main.evaluate --dataset Telecom
```

### 路由参数示例

```bash
python3 -m main.evaluate --dataset Telecom --route_mode auto
python3 -m main.evaluate --dataset Telecom --route_mode counterfactual
```

## 参数说明（节选）

`python3 -m main.evaluate --help` 查看完整参数。以下是与执行路由相关的重点参数：

- `--route_mode {auto|fast|counterfactual}`：策略路由模式（新增，默认 auto，不破坏旧命令）
- `--fast_easy_solver / --no_fast_easy_solver`：是否启用 fast-solver（原有参数）
- `--force_counterfactual`：强制使用 counterfactual agent（原有参数）
- `--no_llm_fallback`：禁止 LLM 回退（原有参数）

## 输出结果说明

运行 `main.evaluate` 后，会生成 `test/` 目录（与旧行为一致）：

- `test/result/{DATASET}/agent-{tag}-{model}-{uid}.csv`：每条 query 的 prediction 与评测得分
- `test/monitor/{DATASET}/agent-{tag}-{model}/{uid}/history/`：日志
- `test/monitor/{DATASET}/agent-{tag}-{model}/{uid}/trajectory/`：trajectory（ipynb）
- `test/monitor/{DATASET}/agent-{tag}-{model}/{uid}/prompt/`：prompt（json）

## 如何扩展新的 solver / agent

推荐通过新增 Strategy 的方式接入，而不是侵入式改写旧流程：

1. 在 `rca/agent/agent_standard.py` 中新增一个 `Strategy`（实现 `can_handle()` 与 `run()`，返回 `ExecutionResult`）
2. 在 `build_agent_standard_runner()` 中把新 Strategy 注册进路由（控制顺序即可实现优先级与回退）
3. 如果要新增一个全新的评测脚本入口，参考 `main/evaluate.py` 新增一个 `main/*.py`

## 其他

时间相关注意事项：OpenRCA 的故障时间均记录在 UTC+8（Asia/Shanghai）时区，若用本地时区转换可能导致时间偏移而影响评测。
