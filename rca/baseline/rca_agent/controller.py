import json
import re
import time
from IPython.terminal.embed import InteractiveShellEmbed

try:
    from rca.baseline.rca_agent.executor import execute_act
except Exception:
    execute_act = None

from rca.api_router import get_chat_completion
from rca.multi_agent import MultiAgentCoordinator
from rca.skill_module import load_skill_knowledge

ENABLE_EXTERNAL_KNOWLEDGE = False

def _safe_json_loads(raw: str) -> dict:
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r":\s*None(\s*[,}])", r": null\1", s)
    s = re.sub(r":\s*True(\s*[,}])", r": true\1", s)
    s = re.sub(r":\s*False(\s*[,}])", r": false\1", s)

    try:
        return json.loads(s)
    except Exception:
        return {}

system = """You are the Administrator of a DevOps Assistant system for failure diagnosis. To solve each given issue, you should iteratively instruct an Executor to write and execute Python code for data analysis on telemetry files of target system. By analyzing the execution results, you should approximate the answer step-by-step.

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

format = """{
    "analysis": (Your analysis of the code execution result from Executor in the last step, with detailed reasoning of 'what have been done' and 'what can be derived'. Respond 'None' if it is the first step.),
    "completed": ("True" if you believe the issue is resolved, and an answer can be derived in the 'instruction' field. Otherwise "False"),
    "instruction": (Your instruction for the Executor to perform via code execution in the next step. Do not involve complex multi-step instruction. Keep your instruction atomic, with clear request of 'what to do' and 'how to do'. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved.)
}
(DO NOT contain "```json" and "```" tags. DO contain the JSON object with the brackets "{}" only. Use '\\n' instead of an actual newline character to ensure JSON compatibility when you want to insert a line break within a string.)"""

summary = """Now, you have decided to finish your reasoning process. You should now provide the final answer to the issue. The candidates of possible root cause components and reasons are provided to you. The root cause components and reasons must be selected from the provided candidates.

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
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (mandatory)
    "2": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (only if the failure number is "unknown" or "more than one" in the issue)
    ... (only if the failure number is "unknown" or "more than one" in the issue)
}}
```
(Please use "```json" and "```" tags to wrap the JSON object. You only need to provide the elements asked by the issue, and ommited the other fields in the JSON.)
Note that all the root cause components and reasons must be selected from the provided candidates. Do not reply 'unknown' or 'null' or 'not found' in the JSON. Do not be too conservative in selecting the root cause components and reasons. Be decisive to infer a possible answer based on your current observation."""

def control_loop(objective:str, plan:str, ap, bp, logger, dataset_name:str = "", max_step = 15, max_turn = 3, task_focus: str = "") -> str:
    if execute_act is None:
        raise ModuleNotFoundError("rca.baseline.rca_agent.executor is missing; controller-based LLM execution is unavailable.")
   
    skill = {}
    coordinator = None
    skill_knowledge = ""
    if ENABLE_EXTERNAL_KNOWLEDGE:
        skill = load_skill_knowledge(dataset_name=dataset_name) if dataset_name else load_skill_knowledge(dataset_name="")
        coordinator = MultiAgentCoordinator(dataset_name=dataset_name, objective=objective, skill_blob=skill)
        skill_knowledge = coordinator.build_skill_context()
    prompt = [
            {'role': 'system', 'content': system.format(objective=objective,
                                                        format=format,
                                                        agent=ap.rules, 
                                                        background=bp.schema,
                                                        skill_knowledge=skill_knowledge,
                                                        task_focus=task_focus or "Follow the issue instruction and optimize the fields explicitly required by the task.")},
            {'role': 'user', 'content': "Let's begin."}
        ]

    history = []
    trajectory = []
    observation = "Let's begin."
    status = False
    kernel = InteractiveShellEmbed()
    init_code = (
            "import pandas as pd\n"
            "import os\n"
            "from rca.trace_fast import infer_downstream_component_per_trace\n"
            "_CSV_CACHE = {}\n"
            "def load_csv_cached(path, **kwargs):\n"
            "    key = (str(path), tuple(sorted(kwargs.items())))\n"
            "    if key in _CSV_CACHE:\n"
            "        return _CSV_CACHE[key]\n"
            "    df = pd.read_csv(path, **kwargs)\n"
            "    _CSV_CACHE[key] = df\n"
            "    return df\n"
            "def load_all_metrics(metric_dir):\n"
            "    dfs = []\n"
            "    for f in os.listdir(metric_dir):\n"
            "        if f.endswith('.csv'):\n"
            "            df = load_csv_cached(os.path.join(metric_dir, f))\n"
            "            dfs.append(df)\n"
            "    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()\n"
            "pd.set_option('display.width', 427)\n"
            "pd.set_option('display.max_columns', 10)\n"
    )
    kernel.run_cell(init_code)

    for step in range(max_step):
        
        note = [{'role': 'user', 'content': f"Continue your reasoning process for the target issue:\n\n{objective}\n\nFollow the rules during issue solving:\n\n{ap.rules}.\n\nIf there are multiple plausible candidates, identify the current leading candidate, note one contradiction that would weaken it, and choose the next atomic check that most helps distinguish it from the next-best alternative. If local evidence is already sufficient, you may complete without forcing extra verification steps.\n\nResponse format:\n\n{format}"}]
        attempt_actor = []
        response_raw = ""
        try:
            t_controller = time.perf_counter()
            response_raw = get_chat_completion(
                messages=prompt + note,
            )
            t_controller = time.perf_counter() - t_controller
            if not isinstance(response_raw, str) or not response_raw.strip():
                logger.warning("Empty response from controller model. Asking for a retry.")
                prompt.append({'role': 'user', 'content': "Your previous response was empty. Please output one valid JSON object in the requested format."})
                continue
            if "```json" in response_raw:
                response_raw = re.search(r"```json\n(.*)\n```", response_raw, re.S).group(1).strip()
            logger.debug(f"Raw Response:\n{response_raw}")
            if '"analysis":' not in response_raw or '"instruction":' not in response_raw or '"completed":' not in response_raw:
                logger.warning("Invalid response format. Please provide a valid JSON response.")
                prompt.append({'role': 'assistant', 'content': response_raw})
                prompt.append({'role': 'user', 'content': "Please provide your analysis in requested JSON format."})
                continue
            response = _safe_json_loads(response_raw)
            if not response:
                logger.warning("JSON parsing failed. Please provide a valid JSON response with double quotes.")
                prompt.append({'role': 'assistant', 'content': response_raw})
                prompt.append({'role': 'user', 'content': "Your JSON was invalid. Please output only one JSON object using double quotes and no trailing commas."})
                continue
            analysis = response['analysis']
            instruction = response['instruction']
            completed = response['completed']
            if isinstance(completed, bool):
                completed = "True" if completed else "False"
            logger.info('-'*80 + '\n' + f"### Step[{step+1}]\nAnalysis: {analysis}\nInstruction: {instruction}" + '\n' + '-'*80)
            logger.info(f"Controller time: {t_controller:.3f}s")

            if completed == "True":
                kernel.reset()
                prompt.append({'role': 'assistant', 'content': response_raw})
                prompt.append({'role': 'user', 'content': summary.format(objective=objective,
                                                                                task_focus=task_focus or "Follow the issue instruction and optimize the fields explicitly required by the task.",
                                                                                cand=bp.cand)})
                answer = get_chat_completion(
                    messages=prompt,
                )
                if not isinstance(answer, str) or not answer.strip():
                    logger.warning("Empty final answer from controller model.")
                    answer = "{}"
                logger.debug(f"Raw Final Answer:\n{answer}")
                prompt.append({'role': 'assistant', 'content': answer})
                if "```json" in answer:
                    answer = re.search(r"```json\n(.*)\n```", answer, re.S).group(1).strip()
                if coordinator is not None:
                    coordinator.on_final_answer(answer=answer, trajectory=trajectory)
                return answer, trajectory, prompt

            t_exec = time.perf_counter()
            code, result, status, new_history = execute_act(instruction, bp.schema, history, attempt_actor, kernel, logger, max_retry=max_turn)
            t_exec = time.perf_counter() - t_exec
            if not status:
                logger.warn(f'Self-Correction failed.')
                observation = "The Executor failed to execute the instruction. Please provide a new instruction."
            observation = f"{result}"
            history = new_history
            trajectory.append({'code': f"# In[{step+1}]:\n\n{code}", 'result': f"Out[{step+1}]:\n```\n{result}```"})
            logger.info('-'*80 + '\n' + f"Step[{step+1}]\n### Observation:\n{result}" + '\n' + '-'*80)
            logger.info(f"Executor time: {t_exec:.3f}s")
            prompt.append({'role': 'assistant', 'content': response_raw})
            prompt.append({'role': 'user', 'content': observation})

        except Exception as e:
            logger.error(e)
            prompt.append({'role': 'assistant', 'content': response_raw})
            prompt.append({'role': 'user', 'content': f"{str(e)}\nPlease provide your analysis in requested JSON format."})
            if 'context_length_exceeded' in str(e):
                logger.warning("Token length exceeds the limit.")
                kernel.reset()
                return "Token length exceeds. No root cause found.", trajectory, prompt

    logger.warning("Max steps reached. Please check the history.")
    kernel.reset()
    final_prompt = {'role': 'user', 'content': summary.format(objective=objective,
                                                                    task_focus=task_focus or "Follow the issue instruction and optimize the fields explicitly required by the task.",
                                                                    cand=bp.cand).replace('Now, you have decided to finish your reasoning process. ', 'Now, the maximum steps of your reasoning have been reached. ')}
    if prompt[-1]['role'] == 'user':
        prompt[-1]['content'] = final_prompt['content']
    else:
        prompt.append({'role': 'user', 'content': final_prompt['content']})
    answer = get_chat_completion(
        messages=prompt,
    )
    if not isinstance(answer, str) or not answer.strip():
        logger.warning("Empty final answer from controller model after max steps.")
        answer = "{}"
    logger.debug(f"Raw Final Answer:\n{answer}")
    prompt.append({'role': 'assistant', 'content': answer})
    if "```json" in answer:
        answer = re.search(r"```json\n(.*)\n```", answer, re.S).group(1).strip()
    if coordinator is not None:
        coordinator.on_final_answer(answer=answer, trajectory=trajectory)
    return answer, trajectory, prompt
