import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rca.api_router import get_chat_completion
from rca.execution.evidence_builder import build_candidate_evidence
from .schemas import CandidateHypothesis, CounterfactualCheck, DeterministicEvidence
from .utils import ensure_list, normalize_score, safe_json_loads


def deduplicate_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


class CounterfactualExecutor:
    """
    Executor 负责具体动作执行：
    1. 提取候选根因
    2. 为候选生成反事实问题
    3. 对候选进行反事实验证（LLM + deterministic evidence）
    4. 汇总最终结论
    """

    def __init__(self, agent_prompt: str, basic_prompt: str, logger=None):
        self.agent_prompt = agent_prompt
        self.basic_prompt = basic_prompt
        self.logger = logger
        self.last_llm_call: Dict[str, Any] = {}

    @staticmethod
    def estimate_input_tokens(messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("role", "")) + "\n" + str(m.get("content", "")) for m in messages)
        return max(1, int(len(text) / 4))

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        input_tokens = self.estimate_input_tokens(messages)
        self.last_llm_call = {
            "approx_input_tokens": int(input_tokens),
            "elapsed_seconds": 0.0,
            "attempted": True,
        }
        if self.logger:
            self.logger.info(f"[Executor] Sending prompt to LLM... approx_input_tokens={input_tokens}")

        t0 = time.perf_counter()
        try:
            resp = get_chat_completion(messages, temperature=0.0)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.last_llm_call = {
                "approx_input_tokens": int(input_tokens),
                "elapsed_seconds": float(elapsed),
                "attempted": True,
                "error": str(e),
            }
            if self.logger:
                self.logger.info(f"[Executor] LLM request failed. elapsed_seconds={elapsed:.6f}, error={e}")
            raise
        elapsed = time.perf_counter() - t0
        content = str(resp)
        self.last_llm_call = {
            "approx_input_tokens": int(input_tokens),
            "elapsed_seconds": float(elapsed),
            "attempted": True,
        }

        if self.logger:
            self.logger.info(f"[Executor] LLM response received. elapsed_seconds={elapsed:.6f}")
        return content

    def select_best_candidate_with_llm(
        self,
        instruction: str,
        candidates: List[CandidateHypothesis],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        context = context or {}
        top = list(candidates or [])[: max(1, int(top_k))]
        if not top:
            raise ValueError("No candidates for LLM verification.")
        enable_trace = bool(context.get("enable_trace_evidence", True))

        candidate_payload = []
        all_components = [str(c.component) for c in list(candidates or []) if str(getattr(c, "component", "") or "").strip()]
        for i, cand in enumerate(top, start=1):
            next_score = float(top[i].score) if i < len(top) else 0.0
            try:
                ev = build_candidate_evidence(
                    dataset=dataset_name,
                    instruction=instruction,
                    component=str(cand.component),
                    reason=str(cand.reason),
                    score=float(cand.score),
                    score_gap_vs_next=max(0.0, float(cand.score) - next_score),
                    all_candidate_components=all_components,
                    topology_max_rows=context.get("basil_max_trace_rows"),
                    enable_trace=bool(enable_trace),
                ).to_dict()
            except Exception:
                ev = {
                    "component": str(cand.component),
                    "reason": str(cand.reason),
                    "score": float(cand.score),
                    "score_gap_vs_next": max(0.0, float(cand.score) - next_score),
                    "metric_timeline": [],
                    "trace_timeline": [],
                    "causal_evidence": {},
                    "evidence_for": [str(x) for x in list(cand.evidence or [])[:4]],
                    "evidence_against": ["Evidence builder failed; fallback to candidate summary."],
                }
            ev["id"] = i
            candidate_payload.append(ev)

        system_prompt = """
You are an RCA evidence judge. Your job is to choose between candidates based on
evidence, not just scores.

You must choose exactly one candidate from the given top-k list. Do not invent a
new component or reason. Focus on:
- which anomaly appeared first
- which candidate has stronger metric timeline evidence
- which candidate is causally upstream rather than merely impacted downstream
- whether the reason matches the anomalous metrics

Output strictly in JSON:
{
  "selected_id": 1,
  "confidence": 0.0,
  "rationale": "one sentence: why this candidate over the others",
  "supporting_evidence": ["decisive evidence item", "..."]
}
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}

Uncertainty gate:
{json.dumps(context.get("uncertainty_gate", {}), ensure_ascii=False)}

Candidate evidence comparison:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

Choose the most likely root-cause component/reason from the candidate list only.
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "selected_id": 1,
                "confidence": float(top[0].score),
                "rationale": "Fallback to top-1 because the LLM output is invalid.",
                "supporting_evidence": [],
            },
        )
        try:
            selected_id = int(data.get("selected_id", 1))
        except Exception:
            selected_id = 1
        if selected_id < 1 or selected_id > len(top):
            selected_id = 1
        selected = top[selected_id - 1]
        details = {
            "selected_id": int(selected_id),
            "selected_candidate": selected.to_dict(),
            "confidence": normalize_score(data.get("confidence", selected.score)),
            "rationale": str(data.get("rationale", "")).strip(),
            "supporting_evidence": ensure_list(data.get("supporting_evidence")),
            "raw_response": raw,
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
        }
        return {"candidate": selected, "details": details}

    def generate_multi_root_with_llm(
        self,
        *,
        instruction: str,
        evidence_groups: List[Dict[str, Any]],
        target_k: int,
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        k = int(target_k) if int(target_k) >= 1 else 1
        groups = list(evidence_groups or [])
        component_level = str(context.get("component_level", "") or "").strip().lower()

        system_prompt = f"""
You are an RCA agent for multi-failure queries.

The query explicitly contains {k} root causes / failures.
The system has selected a candidate root-cause set using coverage-first joint selection.
Each EvidenceGroup represents one possible root cause.

Component naming note:
- The expected component level for this dataset is: "{component_level or 'unknown'}".
- Ensure your output component matches that level (pod vs service vs node) when possible.

Your job:
- Verify the evidence in each EvidenceGroup.
- Output exactly {k} root causes.
- Avoid duplicate components unless the evidence clearly indicates repeated independent failures.
- For cascading cases, prefer upstream root causes unless the query asks for all failure sources.

Output strictly in JSON with keys "1"..."{k}". Each item must include:
- "root cause occurrence datetime"
- "root cause component"
- "root cause reason"
- "supporting evidence" (list of short strings)
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}

QueryMeta:
{json.dumps(context.get("query_meta", {}), ensure_ascii=False)}

Joint selection details:
{json.dumps(context.get("multi_root_selection_details", {}), ensure_ascii=False)}

EvidenceGroups (selected set, structured):
{json.dumps(groups, ensure_ascii=False, indent=2)}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(raw, {})

        out: Dict[str, Any] = {}
        for i in range(1, k + 1):
            key = str(i)
            item = data.get(key) if isinstance(data, dict) else None
            if not isinstance(item, dict):
                item = {}
            comp = str(item.get("root cause component", "") or "").strip()
            reason = str(item.get("root cause reason", "") or "").strip()
            dt = str(item.get("root cause occurrence datetime", "") or "").strip()
            sup = ensure_list(item.get("supporting evidence", []))[:6]

            fallback_comp = ""
            fallback_reason = ""
            fallback_dt = ""
            if 0 <= (i - 1) < len(groups):
                g = groups[i - 1] if isinstance(groups[i - 1], dict) else {}
                fallback_comp = str(g.get("candidate_component", "") or "").strip()
                fallback_reason = str(g.get("candidate_reason", "") or "").strip()
                fallback_dt = str(g.get("candidate_time", "") or "").strip()

            out[key] = {
                "root cause component": comp or fallback_comp,
                "root cause reason": reason or fallback_reason,
            }
            if dt or fallback_dt:
                out[key]["root cause occurrence datetime"] = dt or fallback_dt
            if sup:
                out[key]["supporting evidence"] = sup

        details = {
            "target_k": int(k),
            "group_count": int(len(groups)),
            "raw_response": str(raw),
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
        }
        return {"prediction_json": json.dumps(out, ensure_ascii=False), "details": details}

    def select_time_with_llm(
        self,
        instruction: str,
        component: str,
        reason: str,
        time_candidates: List[Dict[str, Any]],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        context = context or {}
        comp = str(component or "")
        from rca.execution.evidence_builder import build_time_evidence
        from rca.adapter.utils.time_candidates import TimeCandidate
        enable_trace = bool(context.get("enable_trace_evidence", True))

        def _base_score(item: Dict[str, Any]) -> float:
            try:
                return float(item.get("score", 0.0) or 0.0)
            except Exception:
                return 0.0

        by_dt: Dict[str, Dict[str, Any]] = {}
        for rank, cand in enumerate(time_candidates or [], start=1):
            if not isinstance(cand, dict):
                continue
            dt = str(cand.get("datetime", "") or "").strip()
            if not dt:
                continue
            item = dict(cand)
            item["_original_rank"] = rank
            old = by_dt.get(dt)
            if old is None or _base_score(item) > _base_score(old):
                merged_sources = []
                merged_candidates = []
                if old is not None:
                    merged_sources.extend(list(old.get("_merged_sources", []) or [old.get("source", "")]))
                    merged_candidates.extend(list(old.get("_merged_candidates", []) or [old]))
                merged_sources.append(str(item.get("source", "")))
                merged_candidates.append(dict(item))
                item["_merged_sources"] = [s for s in merged_sources if s]
                item["_merged_candidates"] = merged_candidates
                by_dt[dt] = item
            else:
                old.setdefault("_merged_sources", [])
                src = str(item.get("source", ""))
                if src and src not in old["_merged_sources"]:
                    old["_merged_sources"].append(src)
                old.setdefault("_merged_candidates", [])
                old["_merged_candidates"].append(dict(item))

        deduped = list(by_dt.values())
        deterministic_order = sorted(
            deduped,
            key=lambda x: (_base_score(x), -int(x.get("_original_rank", 10**9))),
            reverse=True,
        )
        deterministic_best = dict(deterministic_order[0]) if deterministic_order else {}

        def _rerank_score(item: Dict[str, Any]) -> float:
            score = _base_score(item)
            details = {"base_score": score}
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            if comp and c:
                if c == comp:
                    score += 0.32
                    details["component_match_bonus"] = 0.32
                else:
                    score -= 0.10
                    details["component_mismatch_penalty"] = -0.10
            if reason and r:
                if r == reason:
                    score += 0.28
                    details["reason_match_bonus"] = 0.28
                else:
                    score -= 0.08
                    details["reason_mismatch_penalty"] = -0.08
            item["_rerank_score"] = float(score)
            item["_rerank_details"] = details
            return float(score)

        reranked = sorted(deduped, key=lambda x: (_rerank_score(x), _base_score(x), -int(x.get("_original_rank", 10**9))), reverse=True)
        top = reranked[: min(5, max(1, int(top_k)))]
        if not top:
            raise ValueError("No time candidates for LLM verification.")

        time_payload = []
        for i, cand in enumerate(top, start=1):
            evidence = dict(cand.get("evidence", {}) or {})
            if cand.get("_merged_sources"):
                evidence["merged_sources"] = list(cand.get("_merged_sources", []) or [])
            minute_evidence = {}
            try:
                dt = str(cand.get("datetime", "") or "")
                src = str(cand.get("source", "") or "")
                score = float(cand.get("score", 0.0) or 0.0)
                minute_evidence = build_time_evidence(
                    dataset=dataset_name,
                    instruction=instruction,
                    time_candidate=TimeCandidate(
                        datetime=dt,
                        source=src,
                        score=score,
                        component=str(cand.get("component", "") or ""),
                        reason=str(cand.get("reason", "") or ""),
                        evidence=dict(cand.get("evidence", {}) or {}),
                    ),
                    component=comp,
                    reason=str(reason or ""),
                    enable_trace=bool(enable_trace),
                ).to_dict()
            except Exception:
                minute_evidence = {}
            time_payload.append(
                {
                    "id": i,
                    "datetime": str(cand.get("datetime", "")),
                    "source": str(cand.get("source", "")),
                    "score": float(cand.get("score", 0.0) or 0.0),
                    "rerank_score": float(cand.get("_rerank_score", cand.get("score", 0.0)) or 0.0),
                    "rerank_details": dict(cand.get("_rerank_details", {}) or {}),
                    "component": str(cand.get("component", "") or ""),
                    "reason": str(cand.get("reason", "") or ""),
                    "evidence": evidence,
                    "minute_evidence": minute_evidence,
                }
            )

        first_times = context.get("component_first_times", {}) if isinstance(context.get("component_first_times"), dict) else {}
        peak_times = context.get("component_peak_times", {}) if isinstance(context.get("component_peak_times"), dict) else {}
        top_metrics = context.get("top_metrics", {}) if isinstance(context.get("top_metrics"), dict) else {}

        system_prompt = """
You are a concise RCA time verifier.

Choose exactly one root-cause occurrence datetime from the provided time candidates.
Do not invent a new datetime. Prefer the earliest stable root-cause onset.

Use minute_evidence heavily:
- Prefer candidates where component metrics show FIRST_BREACH at this minute or just started breaching,
  and trace is not yet severely impacted at this minute.
- If a later candidate is explained by trace already being impacted, it is likely too late.

You may only rerank the provided candidates.
If the deterministic top-1 candidate has trace_first_breach/trace_onset/trace_stable_onset with score >= 1.0,
treat it as a protected high-confidence trace onset. Override it only when another candidate has stronger
and earlier component-metric onset evidence for the fixed component.

Output strictly in JSON:
{
  "selected_id": 1,
  "confidence": 0.0,
  "rationale": "short reason"
}
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}

Fixed root-cause component: {comp}
Fixed root-cause reason: {reason}

Component time evidence:
- first_time: {first_times.get(comp, "")}
- peak_time: {peak_times.get(comp, "")}
- top_metrics: {json.dumps(list(top_metrics.get(comp, []) or [])[:8], ensure_ascii=False)}

Time candidates:
{json.dumps(time_payload, ensure_ascii=False, indent=2)}

Select the best root-cause occurrence datetime from the candidate list only.
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "selected_id": 1,
                "confidence": float(top[0].get("score", 0.0) or 0.0),
                "rationale": "Fallback to top time candidate because the LLM output is invalid.",
            },
        )
        try:
            selected_id = int(data.get("selected_id", 1))
        except Exception:
            selected_id = 1
        if selected_id < 1 or selected_id > len(top):
            selected_id = 1
        proposed = dict(top[selected_id - 1])
        rerank_best = dict(top[0])
        if not deterministic_best:
            deterministic_best = dict(rerank_best)

        def _score(item: Dict[str, Any]) -> float:
            try:
                return float(item.get("score", 0.0) or 0.0)
            except Exception:
                return 0.0

        def _source_parts(item: Dict[str, Any]) -> set:
            parts = set()
            source = str(item.get("source", "") or "")
            parts.update([s for s in source.split("+") if s])
            for src in item.get("_merged_sources", []) or []:
                parts.update([s for s in str(src).split("+") if s])
            evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
            for src in evidence.get("merged_sources", []) or []:
                parts.update([s for s in str(src).split("+") if s])
            return parts

        def _evidence_allows_low_score(item: Dict[str, Any]) -> bool:
            source_parts = _source_parts(item)
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            strong_sources = {
                "reason_metric_onset",
                "component_metric_first",
                "trace_stable_onset",
                "trace_onset",
            }
            if not (source_parts & strong_sources):
                return False
            if c and c == comp:
                return True
            if r and r == str(reason or ""):
                return True
            return bool(source_parts & {"reason_metric_onset", "component_metric_first"})

        def _is_protected_trace_onset(item: Dict[str, Any]) -> bool:
            return _score(item) >= 1.0 and bool(_source_parts(item) & {"trace_stable_onset", "trace_onset"})

        def _strong_override_for_protected(item: Dict[str, Any], drop: float) -> bool:
            source_parts = _source_parts(item)
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            if source_parts & {"trace_stable_onset", "trace_onset"}:
                return True
            if "reason_metric_onset" in source_parts and r and r == str(reason or ""):
                return True
            if "component_metric_first" in source_parts and c and c == comp and drop <= 0.05:
                return True
            return False

        max_drop = float(context.get("time_llm_max_score_drop", 0.20) or 0.20)
        score_drop = max(0.0, _score(deterministic_best) - _score(proposed))
        differs_from_deterministic = str(proposed.get("datetime", "")) != str(deterministic_best.get("datetime", ""))
        protected_deterministic_top1 = _is_protected_trace_onset(deterministic_best)
        accepted = True
        veto_reason = ""
        if differs_from_deterministic and score_drop > max_drop and not _evidence_allows_low_score(proposed):
            accepted = False
            veto_reason = (
                "LLM selected a lower-scored time candidate without explicit "
                "component/reason/source evidence; reverted to deterministic top-1."
            )
        if (
            accepted
            and differs_from_deterministic
            and protected_deterministic_top1
            and not _strong_override_for_protected(proposed, score_drop)
        ):
            accepted = False
            veto_reason = (
                "Deterministic top-1 is a protected high-confidence trace onset; "
                "LLM override lacked strong reason/trace evidence."
            )
        selected = proposed if accepted else deterministic_best
        try:
            def _dt_s(x: str) -> Optional[int]:
                try:
                    return int(datetime.fromisoformat(str(x)).timestamp())
                except Exception:
                    return None

            selected_dt_s = _dt_s(selected.get("datetime", ""))
            trace_items = []
            for it in deduped:
                if not isinstance(it, dict):
                    continue
                parts = _source_parts(it)
                if parts & {"trace_first_breach", "trace_onset", "trace_stable_onset"} and _score(it) >= 0.90:
                    t_s = _dt_s(it.get("datetime", ""))
                    if t_s is not None:
                        trace_items.append((t_s, dict(it)))
            if selected_dt_s is not None and trace_items:
                trace_items.sort(key=lambda x: x[0])
                earliest_trace_s, earliest_trace = trace_items[0]
                delta = int(selected_dt_s) - int(earliest_trace_s)
                if 120 <= delta <= 300 and not (_source_parts(selected) & {"trace_first_breach", "trace_onset", "trace_stable_onset"}):
                    selected = dict(earliest_trace)
                    accepted = True
                    veto_reason = "Prefer earlier high-confidence trace onset within 2-5 minutes."
        except Exception:
            pass
        accepted_selected_id = -1
        for i, item in enumerate(top, start=1):
            if str(item.get("datetime", "")) == str(selected.get("datetime", "")):
                accepted_selected_id = i
                break
        details = {
            "selected_id": int(selected_id),
            "accepted_selected_id": int(accepted_selected_id),
            "selected_time_candidate": selected,
            "deterministic_top1_candidate": deterministic_best,
            "reranked_time_candidates": time_payload,
            "datetime": str(selected.get("datetime", "")),
            "confidence": normalize_score(data.get("confidence", selected.get("score", 0.0))),
            "rationale": str(data.get("rationale", "")).strip(),
            "rerank_policy": "llm_may_rerank_top_candidates_with_protected_trace_onset_veto",
            "score_drop_from_deterministic_top1": float(score_drop),
            "protected_deterministic_top1": bool(protected_deterministic_top1),
            "max_score_drop_without_evidence": float(max_drop),
            "accepted": bool(accepted),
            "veto_reason": veto_reason,
            "raw_response": raw,
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
        }
        return {"datetime": details["datetime"], "details": details}

    def propose_candidates(
        self,
        instruction: str,
        dataset_name: str = "",
        task_focus: str = "",
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateHypothesis]:
        context = context or {}
        allowed_components = context.get("allowed_components")
        allowed_reasons = context.get("allowed_reasons")
        component_scores = context.get("component_scores", {}) if isinstance(context.get("component_scores"), dict) else {}
        reason_scores = context.get("reason_scores", {}) if isinstance(context.get("reason_scores"), dict) else {}
        det_components: List[str] = []
        det_reasons: List[str] = []
        if isinstance(allowed_components, list):
            det_components = [str(x) for x in allowed_components if str(x or "").strip()]
        if isinstance(allowed_reasons, list):
            det_reasons = [str(x) for x in allowed_reasons if str(x or "").strip()]

        if not det_components and component_scores:
            try:
                det_components = [
                    str(k)
                    for k, _v in sorted(component_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:12]
                    if str(k or "").strip()
                ]
            except Exception:
                det_components = []
        if not det_reasons and reason_scores:
            try:
                det_reasons = [
                    str(k)
                    for k, _v in sorted(reason_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:12]
                    if str(k or "").strip()
                ]
            except Exception:
                det_reasons = []

        if det_components or det_reasons:
            max_c = max([abs(float(v)) for v in component_scores.values()] + [0.0])
            max_r = max([abs(float(v)) for v in reason_scores.values()] + [0.0])
            comp_list = det_components[:12] if det_components else ["unknown"]
            reason_list = det_reasons[:12] if det_reasons else ["unknown"]
            cand: List[CandidateHypothesis] = []
            for c in comp_list:
                cs = float(component_scores.get(str(c), 0.0) or 0.0)
                cn = cs / max(max_c, 1e-9) if max_c > 0 else 0.0
                for r in reason_list:
                    rs = float(reason_scores.get(str(r), 0.0) or 0.0)
                    rn = rs / max(max_r, 1e-9) if max_r > 0 else 0.0
                    if not det_components:
                        joint = rn
                    elif not det_reasons:
                        joint = cn
                    else:
                        joint = 0.60 * cn + 0.40 * rn
                    cand.append(
                        CandidateHypothesis(
                            component=str(c),
                            reason=str(r),
                            evidence=[],
                            score=normalize_score(joint),
                            source="deterministic_proposal",
                        )
                    )
            cand.sort(key=lambda x: x.score, reverse=True)
            return cand[: int(top_k)]

        system_prompt = f"""{self.agent_prompt}

You are an RCA executor specialized in counterfactual reasoning.
Your task is to read the RCA query and propose plausible root-cause hypotheses.

Output strictly in JSON:
{{
  "candidates": [
    {{
      "component": "...",
      "reason": "...",
      "evidence": ["...", "..."],
      "score": 0.0
    }}
  ]
}}

Requirements:
- score must be in [0,1]
- return at most {top_k} candidates
- prefer precise component and reason names
- evidence should be short, factual, and grounded in the query
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(raw, {"candidates": []})

        candidates = []
        for item in data.get("candidates", [])[:top_k]:
            component = str(item.get("component", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not component or not reason:
                continue

            candidates.append(
                CandidateHypothesis(
                    component=component,
                    reason=reason,
                    evidence=ensure_list(item.get("evidence")),
                    score=normalize_score(item.get("score", 0.0)),
                    source="llm_proposal",
                )
            )

        return candidates

    def build_counterfactual_question(
        self,
        instruction: str,
        candidate: CandidateHypothesis,
        dataset_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        context = context or {}
        if bool(context.get("deterministic_only", False)):
            return {
                "counterfactual_question": (
                    f"If {candidate.component} had remained normal and {candidate.reason} had not occurred, "
                    "would the failure still happen?"
                ),
                "expected_effect": "The key anomaly signals related to this candidate should disappear or be significantly weakened.",
            }
        system_prompt = f"""{self.basic_prompt}

You are generating a counterfactual validation question for RCA.

Output strictly in JSON:
{{
  "counterfactual_question": "...",
  "expected_effect": "..."
}}

Requirements:
- The question must follow the form:
  'If component X had remained normal / if reason Y had not occurred, would the observed failure still happen?'
- expected_effect should describe what anomaly/path should disappear or weaken if the candidate is true
"""

        user_prompt = f"""
Dataset: {dataset_name}
RCA query:
{instruction}

Candidate:
- component: {candidate.component}
- reason: {candidate.reason}
- evidence: {candidate.evidence}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "counterfactual_question": (
                    f"If {candidate.component} had remained normal and "
                    f"{candidate.reason} had not occurred, would the failure still happen?"
                ),
                "expected_effect": (
                    "The downstream anomaly should disappear or be significantly weakened."
                ),
            },
        )
        return data

    def _compute_deterministic_evidence(
        self,
        candidate: CandidateHypothesis,
        context: Optional[Dict[str, Any]] = None,
    ) -> DeterministicEvidence:
        """
        根据外部上下文计算确定性证据。
        context 里可接:
        - metric_scores
        - peak_alignment
        - dependency_scores
        - trace_scores
        - historical_stats
        - graph_scores
        - matched_metrics
        - matched_traces
        - dependency_paths
        - historical_cases
        - graph_explanations
        """
        context = context or {}

        metric_scores = context.get("metric_scores", {})
        peak_alignment = context.get("peak_alignment", {})
        dependency_scores = context.get("dependency_scores", {})
        trace_scores = context.get("trace_scores", {})
        historical_stats = context.get("historical_stats", {})
        graph_scores = context.get("graph_scores", {})

        matched_metrics_map = context.get("matched_metrics", {})
        matched_traces_map = context.get("matched_traces", {})
        dependency_paths_map = context.get("dependency_paths", {})
        historical_cases_map = context.get("historical_cases", {})
        graph_explanations_map = context.get("graph_explanations", {})

        key_component_reason = f"{candidate.component}::{candidate.reason}"
        key_component = candidate.component

        metric_anomaly_score = normalize_score(
            metric_scores.get(key_component_reason, metric_scores.get(key_component, 0.0))
        )
        peak_timestamp_alignment = normalize_score(
            peak_alignment.get(key_component_reason, peak_alignment.get(key_component, 0.0))
        )

        dep_item = dependency_scores.get(key_component_reason, dependency_scores.get(key_component, {}))
        if not isinstance(dep_item, dict):
            dep_item = {}

        upstream_dependency_score = normalize_score(dep_item.get("upstream", 0.0))
        downstream_dependency_score = normalize_score(dep_item.get("downstream", 0.0))

        trace_duration_anomaly = normalize_score(
            trace_scores.get(key_component_reason, trace_scores.get(key_component, 0.0))
        )
        historical_frequency = normalize_score(
            historical_stats.get(key_component_reason, historical_stats.get(key_component, 0.0))
        )
        causal_graph_consistency = normalize_score(
            graph_scores.get(key_component_reason, graph_scores.get(key_component, 0.0))
        )

        matched_metrics = deduplicate_keep_order(
            ensure_list(matched_metrics_map.get(key_component_reason))
            + ensure_list(matched_metrics_map.get(key_component))
        )
        matched_traces = deduplicate_keep_order(
            ensure_list(matched_traces_map.get(key_component_reason))
            + ensure_list(matched_traces_map.get(key_component))
        )
        dependency_paths = deduplicate_keep_order(
            ensure_list(dependency_paths_map.get(key_component_reason))
            + ensure_list(dependency_paths_map.get(key_component))
        )
        historical_cases = deduplicate_keep_order(
            ensure_list(historical_cases_map.get(key_component_reason))
            + ensure_list(historical_cases_map.get(key_component))
        )
        graph_explanations = deduplicate_keep_order(
            ensure_list(graph_explanations_map.get(key_component_reason))
            + ensure_list(graph_explanations_map.get(key_component))
        )

        return DeterministicEvidence(
            metric_anomaly_score=metric_anomaly_score,
            peak_timestamp_alignment=peak_timestamp_alignment,
            upstream_dependency_score=upstream_dependency_score,
            downstream_dependency_score=downstream_dependency_score,
            trace_duration_anomaly=trace_duration_anomaly,
            historical_frequency=historical_frequency,
            causal_graph_consistency=causal_graph_consistency,
            matched_metrics=matched_metrics,
            matched_traces=matched_traces,
            dependency_paths=dependency_paths,
            historical_cases=historical_cases,
            graph_explanations=graph_explanations,
        )

    def _aggregate_deterministic_score(self, ev: DeterministicEvidence) -> float:
        """
        一个较稳妥的默认加权方式，后面可以按数据集再调。
        """
        score = (
            0.22 * ev.metric_anomaly_score
            + 0.18 * ev.peak_timestamp_alignment
            + 0.12 * ev.upstream_dependency_score
            + 0.08 * ev.downstream_dependency_score
            + 0.15 * ev.trace_duration_anomaly
            + 0.10 * ev.historical_frequency
            + 0.15 * ev.causal_graph_consistency
        )
        return normalize_score(score)

    def validate_candidate(
        self,
        instruction: str,
        candidate: CandidateHypothesis,
        counterfactual_question: str,
        expected_effect: str,
        dataset_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualCheck:
        deterministic_evidence = self._compute_deterministic_evidence(
            candidate=candidate,
            context=context,
        )
        deterministic_score = self._aggregate_deterministic_score(deterministic_evidence)
        context = context or {}
        if bool(context.get("deterministic_only", False)):
            conf = normalize_score(candidate.score, deterministic_score)
            judgement = "support" if conf >= 0.75 else ("weaken" if conf <= 0.25 else "uncertain")
            return CounterfactualCheck(
                candidate=candidate,
                counterfactual_question=counterfactual_question,
                expected_effect=expected_effect,
                judgement=judgement,
                confidence=conf,
                rationale="Deterministic-only mode: decision derived from deterministic evidence.",
                supporting_evidence=[],
                deterministic_evidence=deterministic_evidence,
            )

        system_prompt = f"""{self.agent_prompt}

You are performing counterfactual validation for RCA.

You must combine:
1. counterfactual reasoning
2. deterministic evidence

Output strictly in JSON:
{{
  "judgement": "support",
  "confidence": 0.0,
  "rationale": "...",
  "supporting_evidence": ["...", "..."]
}}

Rules:
- judgement must be one of: support, weaken, uncertain
- support: if removing this cause would likely eliminate or significantly reduce the observed anomaly
- weaken: if the anomaly likely remains even after removing this cause
- uncertain: insufficient evidence
- confidence must be in [0,1]
- deterministic evidence should be treated as strong grounding signals
"""

        user_prompt = f"""
Dataset: {dataset_name}
RCA query:
{instruction}

Candidate:
- component: {candidate.component}
- reason: {candidate.reason}
- evidence: {candidate.evidence}
- prior score: {candidate.score}

Counterfactual question:
{counterfactual_question}

Expected effect:
{expected_effect}

Deterministic evidence:
- metric_anomaly_score: {deterministic_evidence.metric_anomaly_score}
- peak_timestamp_alignment: {deterministic_evidence.peak_timestamp_alignment}
- upstream_dependency_score: {deterministic_evidence.upstream_dependency_score}
- downstream_dependency_score: {deterministic_evidence.downstream_dependency_score}
- trace_duration_anomaly: {deterministic_evidence.trace_duration_anomaly}
- historical_frequency: {deterministic_evidence.historical_frequency}
- causal_graph_consistency: {deterministic_evidence.causal_graph_consistency}
- aggregated_deterministic_score: {deterministic_score}

Matched details:
- matched_metrics: {deterministic_evidence.matched_metrics}
- matched_traces: {deterministic_evidence.matched_traces}
- dependency_paths: {deterministic_evidence.dependency_paths}
- historical_cases: {deterministic_evidence.historical_cases}
- graph_explanations: {deterministic_evidence.graph_explanations}

Please decide whether this candidate is supported, weakened, or uncertain under the counterfactual assumption.
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "judgement": "uncertain",
                "confidence": deterministic_score,
                "rationale": "Fallback to deterministic evidence because the LLM output is invalid.",
                "supporting_evidence": [],
            },
        )

        llm_conf = normalize_score(data.get("confidence", deterministic_score), deterministic_score)
        final_conf = normalize_score(0.6 * llm_conf + 0.4 * deterministic_score)

        judgement = str(data.get("judgement", "uncertain")).strip().lower()
        if judgement not in {"support", "weaken", "uncertain"}:
            judgement = "uncertain"

        return CounterfactualCheck(
            candidate=candidate,
            counterfactual_question=counterfactual_question,
            expected_effect=expected_effect,
            judgement=judgement,
            confidence=final_conf,
            rationale=str(data.get("rationale", "")).strip(),
            supporting_evidence=ensure_list(data.get("supporting_evidence")),
            deterministic_evidence=deterministic_evidence,
        )

    def refine_candidates(
        self,
        checks: List[CounterfactualCheck],
        top_k: int = 3,
    ) -> List[CandidateHypothesis]:
        scored = []
        for chk in checks:
            base = chk.candidate.score
            cf = chk.confidence

            det = 0.0
            if chk.deterministic_evidence is not None:
                det = self._aggregate_deterministic_score(chk.deterministic_evidence)

            if chk.judgement == "support":
                final_score = 0.25 * base + 0.45 * cf + 0.30 * det
            elif chk.judgement == "weaken":
                final_score = 0.35 * base + 0.10 * cf + 0.15 * det
            else:
                final_score = 0.30 * base + 0.30 * cf + 0.25 * det

            merged_evidence = list(chk.candidate.evidence) + list(chk.supporting_evidence)
            if chk.deterministic_evidence is not None:
                merged_evidence += chk.deterministic_evidence.matched_metrics
                merged_evidence += chk.deterministic_evidence.matched_traces
                merged_evidence += chk.deterministic_evidence.dependency_paths
                merged_evidence += chk.deterministic_evidence.graph_explanations

            scored.append(
                CandidateHypothesis(
                    component=chk.candidate.component,
                    reason=chk.candidate.reason,
                    evidence=deduplicate_keep_order(merged_evidence),
                    score=normalize_score(final_score),
                    source="counterfactual_refine",
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def finalize(
        self,
        checks: List[CounterfactualCheck],
        refined_candidates: List[CandidateHypothesis],
    ) -> Dict[str, Any]:
        """
        最终决策规则：
        1. 优先选择 judgement = support 且 confidence 最高的
        2. 若没有 support，则从 refined_candidates 里选分数最高的
        """
        support_checks = [c for c in checks if c.judgement == "support"]
        if support_checks:
            support_checks.sort(key=lambda x: x.confidence, reverse=True)
            best = support_checks[0]
            return {
                "component": best.candidate.component,
                "reason": best.candidate.reason,
                "confidence": best.confidence,
                "reasoning": best.rationale,
            }

        if refined_candidates:
            best = refined_candidates[0]
            return {
                "component": best.component,
                "reason": best.reason,
                "confidence": best.score,
                "reasoning": "Selected as the highest-ranked hypothesis after counterfactual refinement.",
            }

        return {
            "component": "unknown",
            "reason": "unknown",
            "confidence": 0.0,
            "reasoning": "No valid candidate was produced.",
        }
