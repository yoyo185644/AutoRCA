import io
import re
import ast
import json
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Tuple

from rca.api_router import get_chat_completion


def _extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"```python\\s*(.*?)\\s*```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\\s*(.*?)\\s*```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def _path_from_instruction(instruction: str) -> str:
    s = str(instruction or "")
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1)
    m = re.search(r"\"([^\"]+)\"", s)
    if m:
        return m.group(1)
    return ""


def _maybe_auto_code(instruction: str) -> str:
    s = str(instruction or "").strip()
    sl = s.lower()
    path = _path_from_instruction(s)
    if not path:
        return ""
    if ("list" in sl or "show" in sl) and ("directory" in sl or "subdirector" in sl or "files" in sl):
        return (
            "import os\n"
            f"p = {path!r}\n"
            "print('exists', os.path.exists(p))\n"
            "if os.path.isdir(p):\n"
            "    items = sorted(os.listdir(p))\n"
            "    print('count', len(items))\n"
            "    print('\\n'.join(items[:200]))\n"
            "print('OK')\n"
        )
    if ("check" in sl or "verify" in sl) and ("exists" in sl or "exist" in sl) and ("directory" in sl or "path" in sl):
        return (
            "import os\n"
            f"p = {path!r}\n"
            "print('exists', os.path.exists(p))\n"
            "print('is_dir', os.path.isdir(p))\n"
            "print('OK')\n"
        )
    return ""


def _build_messages(instruction: str, schema: str, history: List[Dict[str, Any]], last_error: str) -> List[Dict[str, str]]:
    hist = history[-6:] if isinstance(history, list) else []
    hist_blob = []
    for h in hist:
        if not isinstance(h, dict):
            continue
        c = str(h.get("code", ""))
        r = str(h.get("result", ""))
        if c or r:
            hist_blob.append({"code": c, "result": r})

    sys = (
        "You are a Python executor for RCA tasks.\n"
        "Write a single Python code cell to satisfy the given instruction.\n"
        "Constraints:\n"
        "- use Python + pandas + os only\n"
        "- do not write files\n"
        "- do not use shell commands\n"
        "- do not plot\n"
        "Never output commands like ls/cd/cat/pwd/grep; they are invalid here.\n"
        "If asked to list files, use os.listdir(path) (optionally with os.path.exists).\n"
        "If asked to read a CSV, use pandas.read_csv(path) and print a small summary.\n"
        "Always end by printing a short conclusion for the controller.\n"
        "Output format: return only valid Python code (no markdown, no JSON, no explanations).\n"
        "Time handling: be careful with timezone-aware timestamps; prefer using epoch(ms) -> Asia/Shanghai.\n"
    )
    user = {
        "instruction": str(instruction),
        "dataset_schema": str(schema),
        "recent_history": hist_blob,
        "previous_error": str(last_error),
    }
    return [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _validate_python(code: str) -> Tuple[bool, str]:
    if not isinstance(code, str) or not code.strip():
        return False, "empty_code"
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    except Exception as e:
        return False, f"parse_error: {e}"


def execute_act(
    instruction: str,
    schema: str,
    history: List[Dict[str, Any]],
    attempt_actor: List[Dict[str, str]],
    kernel: Any,
    logger: Any,
    max_retry: int = 3,
) -> Tuple[str, str, bool, List[Dict[str, Any]]]:
    last_error = ""
    new_history = list(history) if isinstance(history, list) else []
    for _ in range(max(1, int(max_retry))):
        auto_code = _maybe_auto_code(instruction)
        if auto_code:
            code = auto_code
        else:
            messages = _build_messages(instruction, schema, new_history, last_error)
            resp = get_chat_completion(messages=messages)
            code = _extract_code(resp)
            attempt_actor.append({"role": "assistant", "content": resp if isinstance(resp, str) else str(resp)})

        ok_syntax, syntax_err = _validate_python(code)
        if not ok_syntax:
            last_error = syntax_err
            new_history.append({"code": code, "result": syntax_err, "ok": False})
            logger.warning(f"Executor generated invalid python: {syntax_err}")
            continue

        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            res = kernel.run_cell(code)
        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()

        ok = True
        err_text = ""
        if getattr(res, "error_before_exec", None) is not None:
            ok = False
            err_text = str(res.error_before_exec)
        if getattr(res, "error_in_exec", None) is not None:
            ok = False
            err_text = str(res.error_in_exec)

        result = (stdout + ("\n" + stderr if stderr else "")).strip()
        if not result and getattr(res, "result", None) is not None:
            result = str(res.result)
        if not result:
            result = "OK"

        new_history.append({"code": code, "result": result, "ok": ok})
        if ok:
            return code, result, True, new_history

        last_error = err_text or result
        logger.warning(f"Executor execution failed: {last_error}")
    return code, last_error or "Executor failed.", False, new_history
