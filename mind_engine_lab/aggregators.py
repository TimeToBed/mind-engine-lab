from __future__ import annotations

from typing import Dict, List, Protocol, Any
import math

StimulusVec = Dict[str, float]


class ActionAggregator(Protocol):
    def aggregate(self, dim_to_values: Dict[str, List[float]], params: Dict[str, Any]) -> StimulusVec: ...


class WeightedSumActionAggregator:
    def aggregate(self, dim_to_values: Dict[str, List[float]], params: Dict[str, Any]) -> StimulusVec:
        return {d: float(sum(vals)) for d, vals in dim_to_values.items()}


class MaxPoolActionAggregator:
    def aggregate(self, dim_to_values: Dict[str, List[float]], params: Dict[str, Any]) -> StimulusVec:
        return {d: float(max(vals) if vals else 0.0) for d, vals in dim_to_values.items()}


class SoftmaxPoolActionAggregator:
    def aggregate(self, dim_to_values: Dict[str, List[float]], params: Dict[str, Any]) -> StimulusVec:
        tau = max(float(params.get("tau", 1.0)), 1e-6)
        out: StimulusVec = {}
        for d, vals in dim_to_values.items():
            if not vals:
                out[d] = 0.0
                continue
            m = max(vals)
            exps = [math.exp((v - m) / tau) for v in vals]
            out[d] = float(sum(v * e for v, e in zip(vals, exps)) / (sum(exps) + 1e-9))
        return out


ACTION_AGGREGATORS = {
    "weighted_sum": WeightedSumActionAggregator(),
    "max_pool": MaxPoolActionAggregator(),
    "softmax_pool": SoftmaxPoolActionAggregator(),
}
