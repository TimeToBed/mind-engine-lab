from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple


class MotivationArbiter(Protocol):
    def decide(self, drive: Dict[str, float], resist: Dict[str, float], params: Dict[str, Any]) -> Tuple[Dict[str, float], str, Dict[str, Any]]: ...


class DefaultArbiter:
    def decide(self, drive: Dict[str, float], resist: Dict[str, float], params: Dict[str, Any]):
        k = int(params.get("top_k", 3))
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 1.0))
        nonlin = params.get("nonlin", "none")  # none|tanh

        need: Dict[str, float] = {}
        keys = set(drive.keys()) | set(resist.keys())
        for m in keys:
            x = alpha * float(drive.get(m, 0.0)) - beta * float(resist.get(m, 0.0))
            if nonlin == "tanh":
                import math
                x = math.tanh(x)
            need[m] = x

        ranked = sorted(need.items(), key=lambda kv: kv[1], reverse=True)
        top = dict(ranked[:max(k, 1)])

        hi = float(params.get("level_hi", 0.7))
        mid = float(params.get("level_mid", 0.3))
        primary = ranked[0][1] if ranked else 0.0
        level = "high" if primary >= hi else ("mid" if primary >= mid else "low")

        trace = {
            "k": k, "alpha": alpha, "beta": beta, "nonlin": nonlin,
            "need_all": need,
            "ranked": ranked,
            "level_thresholds": {"mid": mid, "high": hi},
        }
        return top, level, trace


ARBITERS = {"default": DefaultArbiter()}