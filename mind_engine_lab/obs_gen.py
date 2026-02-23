from __future__ import annotations

from typing import List, Optional
import random
import time
import math
from .models import WorldModel, Observation, EvidenceProfile, Evidence, Deltas, ObservationGenConfig

from datetime import datetime

def _iso(ts: float) -> str:
    # 无时区；毫秒三位
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _infer_capacity_occupied(world: WorldModel, facility_id: Optional[str]) -> tuple[int, bool]:
    if not facility_id:
        return 0, False
    for f in world.facilities:
        if f.node_id == facility_id:
            cap = int(f.capacity or 0)
            occ = bool(f.occupied) if f.occupied is not None else False
            return cap, occ
    return 0, False


def _infer_body_energy_delta(world: WorldModel, facility_id: Optional[str], object_ids: List[str], duration_s: float) -> float:
    # 这里按“测试规范”生成 [-1,1] 的小范围物理变化：先把原始累计转为一个压缩值
    minutes = duration_s / 60.0
    raw = 0.0

    def add(rate_per_min: Optional[float]):
        nonlocal raw
        if rate_per_min is None:
            return
        raw += float(rate_per_min) * minutes

    if facility_id:
        for f in world.facilities:
            if f.node_id == facility_id:
                add(f.bodyEnergy_per_min)
                break

    if object_ids:
        s = set(object_ids)
        for o in world.objects:
            if o.node_id in s:
                add(o.bodyEnergy_per_min)

    # 压缩到 [-1,1]：用 tanh 类似的饱和（也可改为 clamp(raw, -1,1)）
    return math.tanh(raw)

def generate_observations(world: WorldModel, cfg: ObservationGenConfig, agent_id: Optional[str] = None) -> List[Observation]:
    rnd = random.Random(cfg.seed)

    agent_ids = [a.agent_id for a in world.agents] or list(world.agent_name_index.keys()) or ["Agent_01"]
    if agent_id is not None:
        agent_ids = [agent_id]

    loc_ids = [l.node_id for l in world.locations]
    fac_ids = [f.node_id for f in world.facilities]
    obj_ids = [o.node_id for o in world.objects]

    obs_list: List[Observation] = []
    t0 = time.time()

    for i in range(cfg.n):
        aid = rnd.choice(agent_ids)
        duration = float(rnd.uniform(*cfg.duration_s_range))
        t_start = float(t0 + i * 10.0)
        t_end = float(t_start + duration)

        location_id = rnd.choice(loc_ids) if loc_ids else None
        facility_id = rnd.choice(fac_ids) if fac_ids else None

        k_obj = rnd.randint(0, max(cfg.choose_objects_max, 0))
        chosen_obj_ids = rnd.sample(obj_ids, k=min(k_obj, len(obj_ids))) if obj_ids and k_obj > 0 else []

        capacity, occupied = _infer_capacity_occupied(world, facility_id)

        evidence = Evidence(
            duration_s=round(duration, 1),
            progress_count=int(rnd.randint(*cfg.progress_count_range)),
            capacity=int(capacity or 0),
            occupied=bool(occupied),
            action=None
        )

        body_delta = _infer_body_energy_delta(world, facility_id, chosen_obj_ids, duration)
        body_delta = float(_clamp(body_delta, -1.0, 1.0))

        ep = EvidenceProfile(
            evidence=evidence,
            deltas=Deltas(agent_phys={"bodyEnergy": round(body_delta, 3)}, extra={}),
            related_actions=None,   # 关键：不出现该键
            extra={},
        )

        subjects = world.agent_name_index.get(aid, aid)
        location = [world.entity_name_index.get(location_id, location_id)] if location_id else []

        obs_list.append(Observation(
            obs_id=f"obs_{i:05d}",
            t_start=_iso(t_start),
            t_end=_iso(t_end),
            subjects=subjects,
            location=location,
            evidence_profile=ep,
            status="final",
        ))

    return obs_list

