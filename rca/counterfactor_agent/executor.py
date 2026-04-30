from typing import Any, Dict, List, Optional

from rca.api_router import get_chat_completion
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

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.logger:
            self.logger.info("[Executor] Sending prompt to LLM...")

        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        resp = get_chat_completion(messages, temperature=0.0)
        content = str(resp)

        if self.logger:
            self.logger.info("[Executor] LLM response received.")
        return content

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
        if (
            isinstance(allowed_components, list)
            and isinstance(allowed_reasons, list)
            and (allowed_components or allowed_reasons)
        ):
            max_c = max([abs(float(v)) for v in component_scores.values()] + [0.0])
            max_r = max([abs(float(v)) for v in reason_scores.values()] + [0.0])
            cand = []
            comp_list = allowed_components if allowed_components else ["unknown"]
            reason_list = allowed_reasons if allowed_reasons else ["unknown"]
            for c in comp_list:
                cs = float(component_scores.get(str(c), 0.0) or 0.0)
                cn = cs / max(max_c, 1e-9) if max_c > 0 else 0.0
                for r in reason_list:
                    rs = float(reason_scores.get(str(r), 0.0) or 0.0)
                    rn = rs / max(max_r, 1e-9) if max_r > 0 else 0.0
                    if not allowed_components:
                        joint = rn
                    elif not allowed_reasons:
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
