from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import EngineConfig, WorldModel, Observation
from .world_import import import_world_from_xlsx
from .obs_gen import generate_observations
from .yaml_io import save_config_yaml, load_config_yaml
from .engine import run_engine


def main():
    p = argparse.ArgumentParser(prog="mind-engine", description="Mind Engine Lab CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_import = sub.add_parser("import-world", help="Import world config from xlsx")
    p_import.add_argument("--xlsx", required=True)
    p_import.add_argument("--out", required=True)

    p_gen = sub.add_parser("gen-obs", help="Generate random observations from world")
    p_gen.add_argument("--world", required=True)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--n", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--agent-id", default=None)

    p_init = sub.add_parser("init-config", help="Create a starter YAML config")
    p_init.add_argument("--out", required=True)

    p_test = sub.add_parser("test", help="Run engine on one observation")
    p_test.add_argument("--config", required=True)
    p_test.add_argument("--world", required=True)
    p_test.add_argument("--obs", required=True)
    p_test.add_argument("--index", type=int, default=0)

    args = p.parse_args()

    if args.cmd == "import-world":
        world = import_world_from_xlsx(args.xlsx)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(world.model_dump(by_alias=True), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote world to {args.out}")
        return

    if args.cmd == "gen-obs":
        world = WorldModel.model_validate(json.loads(Path(args.world).read_text(encoding="utf-8")))
        gcfg = EngineConfig().observation_gen
        gcfg.n = args.n
        gcfg.seed = args.seed
        obs_list = generate_observations(world, gcfg, agent_id=args.agent_id)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([o.model_dump() for o in obs_list], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote observations to {args.out}")
        return

    if args.cmd == "init-config":
        cfg = starter_config()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_config_yaml(cfg, args.out)
        print(f"Wrote config to {args.out}")
        return

    if args.cmd == "test":
        cfg = load_config_yaml(args.config)
        world = WorldModel.model_validate(json.loads(Path(args.world).read_text(encoding="utf-8")))
        obs_list = [Observation.model_validate(o) for o in json.loads(Path(args.obs).read_text(encoding="utf-8"))]
        obs = obs_list[args.index]
        res = run_engine(cfg, world, obs)
        print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
        return


def starter_config() -> EngineConfig:
    from .models import EngineConfig, ActionMeaning, MeaningDim, AgentState
    cfg = EngineConfig()

    cfg.agents = [
        AgentState(agent_id="Agent_01", psych_state={"stress": 0.2, "energy": 0.6}, cog_attr={"curiosity": 0.5})
    ]

    cfg.action_meanings = [
        ActionMeaning(
            action="watch_game",
            meas={
                "social_intensity": MeaningDim(desc="crowd exposure", value="min(capacity, nearby_agents)", polarity="+", weight=0.6),
                "info_exposure": MeaningDim(desc="time on content", value="duration_s/60", polarity="+", weight=0.5),
                "comfort": MeaningDim(desc="fatigue", value="(-duration_s/600) + psych.energy", polarity="+", weight=0.4),
            },
        ),
        ActionMeaning(
            action="chat",
            meas={
                "social_intensity": MeaningDim(desc="conversation density", value="nearby_agents", polarity="+", weight=0.7),
                "info_exposure": MeaningDim(desc="new info", value="cog.curiosity * duration_s/120", polarity="+", weight=0.5),
                "comfort": MeaningDim(desc="social cost", value="psych.stress * nearby_agents/10", polarity="-", weight=0.4),
            },
        ),
        ActionMeaning(
            action="rest",
            meas={
                "comfort": MeaningDim(desc="recover", value="clamp(1 - duration_s/600, 0, 1) + (1-psych.stress)", polarity="+", weight=0.8),
            },
        ),
    ]

    cfg.plugins.params = {
        "entity_aggregator": {"w_subject": 1.0, "w_objects": 0.3},
        "arbiter": {"top_k": 3, "alpha": 1.0, "beta": 1.0, "nonlin": "none", "level_mid": 0.3, "level_hi": 0.7},
    }
    return cfg