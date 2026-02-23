from __future__ import annotations

from typing import Any, Dict
import yaml

from .models import EngineConfig


def save_config_yaml(cfg: EngineConfig, path: str) -> None:
    data = cfg.model_dump(by_alias=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_config_yaml(path: str) -> EngineConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return EngineConfig.model_validate(data)


def dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_yaml_to_cfg(yaml_text: str) -> EngineConfig:
    data = yaml.safe_load(yaml_text) or {}
    return EngineConfig.model_validate(data)


def compile_yaml(cfg: EngineConfig) -> str:
    data = cfg.model_dump(by_alias=True, exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)