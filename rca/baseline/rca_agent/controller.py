import json
import re
import time
from IPython.terminal.embed import InteractiveShellEmbed

try:
    from rca.baseline.rca_agent.executor import execute_act
except Exception:
    execute_act = None

from rca.api_router import get_chat_completion
try:
    from rca.multi_agent import MultiAgentCoordinator
except Exception:
    MultiAgentCoordinator = None
try:
    from rca.skill_module import load_skill_knowledge
except Exception:
    load_skill_knowledge = None

ENABLE_EXTERNAL_KNOWLEDGE = False

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Administrator of a DevOps Assistant system for failure diagnosis. To solve each given issue, you should iteratively instruct an Executor to write and execute Python code for data analysis on telemetry files of target system. By analyzing the execution results, you should approximate the answer step-by-step.

There is some domain knowledge for you:

{background}

There is some prior knowledge retrieved by Skill Module. Use it only as priors and never fabricate missing details:

{skill_knowledge}

When the prior knowledge contains ranked candidates or relevant historical cases, use them only as soft priors to prioritize your next verification step. Always let the current metrics, traces, and logs override the priors when they disagree.

For reverse reasoning, do not only search for supporting evidence. When uncertainty remains, ask what contradiction would falsify the current leading candidate and choose a small next check that can separate it from the next-best alternative.

{agent}

The issue you are going to solve is:

{objective}

Task-specific evaluation focus:

{task_focus}

Prioritize only the required root-cause fields for this task and avoid spending extra steps optimizing fields that are not evaluated.

Solve the issue step-by-step. In each step, your response should follow the JSON format below:

{format}

Let's begin."""

STEP_FORMAT = """{
    "analysis": (Your analysis of the code execution result from Executor in the last step, with detailed reasoning of 'what have been done' and 'what can be derived'. Respond 'None' if it is the first step.),
    "completed": ("True" if you believe the issue is resolved, and an answer can be derived in the 'instruction' field. Otherwise "False"),
    "instruction": (Your instruction for the Executor to perform via code execution in the next step. Do not involve complex multi-step instruction. Keep your instruction atomic, with clear request of 'what to do' and 'how to do'. Respond a summary by yourself if you believe the issue is resolved.)
}
(DO NOT contain "```json" and "```" tags. DO contain the JSON object with the brackets "{}" only. Use '\\n' instead of an actual newline character to ensure JSON compatibility when you want to insert a line break within a string.)"""

CONTINUE_PROMPT = """Continue your reasoning process for the target issue:

{objective}

Follow the rules during issue solving:

{rules}

If there are multiple plausible candidates, identify the current leading candidate, note one contradiction that would weaken it, and choose the next atomic check that most helps distinguish it from the next-best alternative. If local evidence is already sufficient, you may complete without forcing extra verification steps.

Response format:

{format}"""

SUMMARY_PROMPT = """Now, you have decided to finish your reasoning process. You should now provide the final answer to the issue. The candidates of possible root cause components and reasons are provided to you. The root cause components and reasons must be selected from the provided candidates.

{cand}

Recall the issue is: {objective}

Task-specific evaluation focus:

{task_focus}

Before the final JSON, clearly separate:
1) Retrieved historical cases & related knowledge (from Skill Module),
2) Your inferred causal path for this issue,
3) Your predicted root cause for this issue.

Please first review your previous reasoning process to infer an exact answer of the issue. Then, summarize your final answer of the root causes using the following JSON format at the end of your response:

```json
{{
    "1": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise omitted),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise omitted),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise omitted),
    }}, (mandatory)
    "2": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise omitted),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise omitted),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise omitted),
    }}, (only if the failure number is "unknown" or "more than one" in the issue)
    ... (only if the failure number is "unknown" or "more than one" in the issue)
}}
```
(Please use "```json" and "```" tags to wrap the JSON object. You only need to provide the elements asked by the issue, and omit the other fields in the JSON.)
Note that all the root cause components and reasons must be selected from the provided candidates. Do not reply 'unknown' or 'null' or 'not found' in the JSON. Do not be too conservative in selecting the root cause components and reasons. Be decisive to infer a possible answer based on your current observation."""

MAX_STEPS_PREAMBLE = "Now, the maximum steps of your reasoning have been reached. "

# ── Kernel initialisation code ────────────────────────────────────────────────

KERNEL_INIT_CODE = """
import pandas as pd
import os
try:
    from rca.trace_fast import infer_downstream_component_per_trace
except Exception:
    infer_downstream_component_per_trace = None

_CSV_CACHE = {}

def load_csv_cached(path, **kwargs):
    key = (str(path), tuple(sorted(kwargs.items())))
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(path, **kwargs)
    return _CSV_CACHE[key]

def load_all_metrics(metric_dir):
    dfs = [
        load_csv_cached(os.path.join(metric_dir, f))
        for f in os.listdir(metric_dir)
        if f.endswith('.csv')
    ]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

pd.set_option('display.width', 427)
pd.set_option('display.max_columns', 10)
""".strip()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json_loads(raw: str) -> dict:
    """Best-effort JSON parser that tolerates common LLM output quirks."""
    if not isinstance(raw, str):
        return {}
    s = raw.strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract outermost {...}
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    s = s[start: end + 1]

    # Fix common LLM JSON mistakes
    s = re.sub(r",\s*([}\]])", r"\1", s)           # trailing commas
    s = re.sub(r":\s*None(\s*[,}])", r": null\1", s)
    s = re.sub(r":\s*True(\s*[,}])", r": true\1", s)
    s = re.sub(r":\s*False(\s*[,}])", r": false\1", s)

    try:
        return json.loads(s)
    except Exception:
        return {}


def _is_completed(value) -> bool:
    """Normalise the 'completed' field regardless of LLM casing/type."""
    return str(value).strip().lower() == "true"


def _strip_json_fence(text: str) -> str:
    """Remove ```json ... ``` fences if present."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    return match.group(1).strip() if match else text.strip()


def _build_summary_prompt(objective: str, task_focus: str, cand: str, *, max_steps_reached: bool = False) -> dict:
    content = SUMMARY_PROMPT.format(objective=objective, task_focus=task_focus, cand=cand)
    if max_steps_reached:
        content = MAX_STEPS_PREAMBLE + content
    return {"role": "user", "content": content}


def _call_llm(messages: list, logger) -> str:
    """Thin wrapper around get_chat_completion with basic validation."""
    response = get_chat_completion(messages=messages)
    if not isinstance(response, str) or not response.strip():
        logger.warning("LLM returned an empty response.")
        return ""
    return response


def _make_kernel() -> InteractiveShellEmbed:
    kernel = InteractiveShellEmbed()
    kernel.run_cell(KERNEL_INIT_CODE)
    return kernel

# ── Main control loop ─────────────────────────────────────────────────────────

def control_loop(
    objective: str,
    plan: str,
    ap,
    bp,
    logger,
    dataset_name: str = "",
    max_step: int = 15,
    max_turn: int = 3,
    task_focus: str = "",
) -> tuple[str, list, list]:
    """
    Iterative Controller-Executor loop for root-cause analysis.

    Returns
    -------
    answer     : JSON string with the final root-cause answer
    trajectory : list of {code, result} dicts, one per executed step
    prompt     : full conversation history (for debugging / logging)
    """
    if execute_act is None:
        raise ModuleNotFoundError(
            "rca.baseline.rca_agent.executor is missing; "
            "controller-based LLM execution is unavailable."
        )

    # ── Skill / coordinator setup ─────────────────────────────────────────────
    coordinator = None
    skill_knowledge = ""
    if ENABLE_EXTERNAL_KNOWLEDGE:
        if load_skill_knowledge is not None and MultiAgentCoordinator is not None:
            skill = load_skill_knowledge(dataset_name=dataset_name) if dataset_name else load_skill_knowledge(dataset_name="")
            coordinator = MultiAgentCoordinator(dataset_name=dataset_name, objective=objective, skill_blob=skill)
            skill_knowledge = coordinator.build_skill_context()

    _task_focus = task_focus or "Follow the issue instruction and optimise the fields explicitly required by the task."

    # ── Initialise conversation ───────────────────────────────────────────────
    prompt: list[dict] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                objective=objective,
                format=STEP_FORMAT,
                agent=ap.rules,
                background=bp.schema,
                skill_knowledge=skill_knowledge,
                task_focus=_task_focus,
            ),
        },
        {"role": "user", "content": "Let's begin."},
    ]

    # continue_note is appended per-call but NOT stored in prompt to avoid bloat
    continue_note = CONTINUE_PROMPT.format(
        objective=objective,
        rules=ap.rules,
        format=STEP_FORMAT,
    )

    history: list = []
    trajectory: list[dict] = []
    kernel = _make_kernel()

    # ── Step loop ─────────────────────────────────────────────────────────────
    try:
        for step in range(max_step):
            # Build the per-call message list without permanently appending the note
            messages_for_call = prompt + [{"role": "user", "content": continue_note}]

            # -- Controller inference --
            t0 = time.perf_counter()
            response_raw = _call_llm(messages_for_call, logger)
            logger.info(f"Controller time: {time.perf_counter() - t0:.3f}s")

            if not response_raw:
                prompt.append({
                    "role": "user",
                    "content": "Your previous response was empty. Please output one valid JSON object in the requested format.",
                })
                continue

            response_raw = _strip_json_fence(response_raw)
            logger.debug(f"Raw Response:\n{response_raw}")

            # Validate required fields before parsing
            required_fields = ('"analysis":', '"instruction":', '"completed":')
            if not all(f in response_raw for f in required_fields):
                logger.warning("Response missing required fields; requesting correction.")
                prompt.append({"role": "assistant", "content": response_raw})
                prompt.append({"role": "user", "content": "Please provide your analysis in the requested JSON format."})
                continue

            response = _safe_json_loads(response_raw)
            if not response:
                logger.warning("JSON parsing failed; requesting correction.")
                prompt.append({"role": "assistant", "content": response_raw})
                prompt.append({
                    "role": "user",
                    "content": "Your JSON was invalid. Please output only one JSON object using double quotes and no trailing commas.",
                })
                continue

            analysis    = response["analysis"]
            instruction = response["instruction"]
            completed   = response["completed"]

            logger.info(
                "-" * 80 + "\n"
                f"### Step [{step + 1}]\n"
                f"Analysis: {analysis}\n"
                f"Instruction: {instruction}\n"
                + "-" * 80
            )

            # -- Early termination: controller believes it's done --
            if _is_completed(completed):
                prompt.append({"role": "assistant", "content": response_raw})
                prompt.append(_build_summary_prompt(objective, _task_focus, bp.cand))

                answer = _call_llm(prompt, logger)
                if not answer:
                    answer = "{}"
                logger.debug(f"Raw Final Answer:\n{answer}")
                prompt.append({"role": "assistant", "content": answer})
                answer = _strip_json_fence(answer)

                if coordinator is not None:
                    coordinator.on_final_answer(answer=answer, trajectory=trajectory)
                return answer, trajectory, prompt

            # -- Executor: run the instructed code --
            t0 = time.perf_counter()
            code, result, status, history = execute_act(
                instruction, bp.schema, history, [], kernel, logger, max_retry=max_turn
            )
            logger.info(f"Executor time: {time.perf_counter() - t0:.3f}s")

            if not status:
                logger.warning("Self-correction failed.")
                observation = "The Executor failed to execute the instruction. Please provide a new instruction."
            else:
                observation = str(result)

            trajectory.append({
                "code":   f"# In[{step + 1}]:\n\n{code}",
                "result": f"Out[{step + 1}]:\n```\n{result}```",
            })
            logger.info("-" * 80 + "\n" f"Step [{step + 1}]\n### Observation:\n{result}\n" + "-" * 80)

            # Commit this step to the permanent conversation
            prompt.append({"role": "assistant", "content": response_raw})
            prompt.append({"role": "user", "content": observation})

    except Exception as e:
        logger.error(f"Unexpected error in control loop: {e}")
        prompt.append({"role": "assistant", "content": ""})
        prompt.append({"role": "user", "content": f"{e}\nPlease provide your analysis in the requested JSON format."})
        if "context_length_exceeded" in str(e):
            logger.warning("Token length exceeds the limit.")
            return "Token length exceeded. No root cause found.", trajectory, prompt

    # ── Fallback: max steps reached ───────────────────────────────────────────
    logger.warning("Max steps reached; generating best-effort final answer.")

    final_note = _build_summary_prompt(objective, _task_focus, bp.cand, max_steps_reached=True)
    # Replace trailing user turn if present to avoid double user messages
    if prompt and prompt[-1]["role"] == "user":
        prompt[-1] = final_note
    else:
        prompt.append(final_note)

    answer = _call_llm(prompt, logger)
    if not answer:
        answer = "{}"
    logger.debug(f"Raw Final Answer:\n{answer}")
    prompt.append({"role": "assistant", "content": answer})
    answer = _strip_json_fence(answer)

    if coordinator is not None:
        coordinator.on_final_answer(answer=answer, trajectory=trajectory)
    return answer, trajectory, prompt
