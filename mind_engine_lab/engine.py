from __future__ import annotations

from typing import Any, Dict, List

from .models import (
    EngineConfig, WorldModel, Observation, ObserverState, L2Input,
    MotivationSignal, MotivationDecision, L4Utterance, RunResult,
)
from .dsl import eval_expr, Namespace
from .aggregators import ACTION_AGGREGATORS, StimulusVec
from .arbiter import ARBITERS


def run_engine(cfg: EngineConfig, world: WorldModel, l2_in: L2Input) -> RunResult:
    obs = l2_in.observation
    observer = l2_in.observer_state
    bindings = _build_bindings(obs, observer)

    # RelatedActions: obs.evidence_profile.related_actions 优先，否则使用 cfg.intent_actions
    related_actions = obs.evidence_profile.related_actions or list(cfg.intent_actions)

    action_meanings = {am.action: am for am in cfg.action_meanings}
    dims = list(cfg.stimulus_dims)

    # ---- Step A: per-action per-dim contributions (signed * weight) ----
    action_traces: List[Dict[str, Any]] = []
    dim_to_values: Dict[str, List[float]] = {d: [] for d in dims}

    for act in related_actions:
        am = action_meanings.get(act)
        if not am:
            continue

        per_action = {"action": act, "dims": {}}
        for d in dims:
            md = am.meas.get(d)
            if not md:
                continue
            raw, tr = eval_expr(md.value, bindings)
            signed = raw if md.polarity == "+" else -raw
            contrib = signed * float(md.weight)
            dim_to_values[d].append(contrib)

            per_action["dims"][d] = {
                "expr": md.value,
                "polarity": md.polarity,
                "weight": md.weight,
                "raw": raw,
                "contrib": contrib,
                "trace": tr.__dict__,
            }
        action_traces.append(per_action)

    # ---- Step B: ActionAggregator -> subject stimulus_vec ----
    aa = ACTION_AGGREGATORS.get(cfg.plugins.action_aggregator)
    if not aa:
        raise ValueError(f"Unknown action_aggregator: {cfg.plugins.action_aggregator}")
    subject_vec = aa.aggregate(dim_to_values, cfg.plugins.params.get("action_aggregator", {}))

    # ---- Step E: event_stimulus_vec -> drive/resist (MVP rule) ----
    drive, resist, map_trace = _map_to_motivation_space(cfg, subject_vec)

    motivation_signal = MotivationSignal(
        obs_id=obs.obs_id,
        agent_name=observer.agent_name,
        stimulus_vec=subject_vec,
        drive=drive,
        resist=resist,
    )

    # ---- L3: MotivationArbiter ----
    arb = ARBITERS.get(cfg.plugins.arbiter)
    if not arb:
        raise ValueError(f"Unknown arbiter: {cfg.plugins.arbiter}")
    top, level, arb_trace = arb.decide(drive, resist, cfg.plugins.params.get("arbiter", {}))

    motivation_decision = MotivationDecision(
        obs_id=obs.obs_id,
        agent_name=observer.agent_name,
        top_motivations=top,
        level=level,
    )

    # ---- L4 ----
    l4 = L4Utterance(
        obs_id=obs.obs_id,
        agent_name=observer.agent_name,
        utterance=_render_utterance(motivation_decision),
    )

    trace = {
        "observation": obs.model_dump(),
        "bindings": _bindings_for_trace(bindings),
        "action_traces": action_traces,
        "subject_vec": subject_vec,
        "stimulus_vec": subject_vec,
        "motivation_mapping": map_trace,
        "arbiter_trace": arb_trace,
    }
    motivation_signal.agent_name = observer.agent_name
    motivation_decision.agent_name = observer.agent_name
    l4.agent_name = observer.agent_name

    return RunResult(
        motivation_signal=motivation_signal,
        motivation_decision=motivation_decision,
        l4=l4,
        trace=trace,
    )


def _build_bindings(obs: Observation, observer: ObserverState) -> Dict[str, Any]:
    ev = obs.evidence_profile.evidence
    return {
        "duration_s": float(ev.duration_s),
        "capacity": float(ev.capacity),
        "occupied": bool(ev.occupied),
        "progress_count": float(ev.progress_count),

        # ✅ 对齐你的新规范命名
        "psych_state": Namespace(observer.psych_state),
        "cog_attr": Namespace(observer.cog_attr),

        # （可选）兼容旧 DSL：psych/cog
        "psych": Namespace(observer.psych_state),
        "cog": Namespace(observer.cog_attr),
    }


def _bindings_for_trace(bindings: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in bindings.items():
        if k in {"psych", "cog"}:
            out[k] = dict(v._d)
        else:
            out[k] = v
    return out


def _map_to_motivation_space(cfg: EngineConfig, event_vec: StimulusVec):
    # MVP：每个 action 的 drive=正向总和，resist=负向总和（可替换为研究者自定义映射）
    pos = sum(max(float(v), 0.0) for v in event_vec.values())
    neg = sum(max(-float(v), 0.0) for v in event_vec.values())

    drive = {act: float(pos) for act in cfg.intent_actions}
    resist = {act: float(neg) for act in cfg.intent_actions}

    trace = {"rule": "pos_sum/neg_sum per action", "pos_sum": pos, "neg_sum": neg}
    return drive, resist, trace


def _render_utterance(decision: MotivationDecision) -> str:
    if not decision.top_motivations:
        return "我现在没有明显的动机。"
    m, _s = max(decision.top_motivations.items(), key=lambda kv: kv[1])
    return f"我现在主要想要「{m}」，动机强度为 {decision.level}。"