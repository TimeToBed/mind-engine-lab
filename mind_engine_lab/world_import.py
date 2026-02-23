from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from .models import WorldModel, Location, Facility, WorldObject, AgentProfile

SHEET_LOCATIONS = "Locations"
SHEET_FACILITIES = "Facilities"
SHEET_OBJECTS = "Objects"
SHEET_AGENTS = "Agents"


def import_world_from_xlsx(xlsx_path: str) -> WorldModel:
    xl = pd.ExcelFile(xlsx_path)

    def read_sheet(name: str) -> pd.DataFrame:
        if name not in xl.sheet_names:
            return pd.DataFrame()
        df = xl.parse(name)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    loc_df = read_sheet(SHEET_LOCATIONS)
    fac_df = read_sheet(SHEET_FACILITIES)
    obj_df = read_sheet(SHEET_OBJECTS)
    ag_df = read_sheet(SHEET_AGENTS)

    world = WorldModel()

    if not loc_df.empty:
        for _, r in loc_df.iterrows():
            world.locations.append(Location(
                node_id=str(r.get("node_id")),
                node_name=str(r.get("node_name")),
                location_type=_maybe_str(r.get("location_type")),
                parent_id=_maybe_str(r.get("parent_id")),
                path=_maybe_str(r.get("path")),
                x=_maybe_float(r.get("x")),
                y=_maybe_float(r.get("y")),
            ))

    if not fac_df.empty:
        for _, r in fac_df.iterrows():
            world.facilities.append(Facility(
                node_id=str(r.get("node_id")),
                node_name=str(r.get("node_name")),
                facility_type=_maybe_str(r.get("facility_type")),
                parent_id=_maybe_str(r.get("parent_id")),
                capacity=_maybe_float(r.get("capacity")),
                occupied=_maybe_float(r.get("occupied")),
                **{"bodyEnergy(/min)": _maybe_float(r.get("bodyEnergy(/min)"))},
            ))

    if not obj_df.empty:
        for _, r in obj_df.iterrows():
            world.objects.append(WorldObject(
                node_id=str(r.get("node_id")),
                node_name=str(r.get("node_name")),
                object_type=_maybe_str(r.get("object_type")),
                parent_id=_maybe_str(r.get("parent_id")),
                is_consumable=_maybe_bool(r.get("is_consumable")),
                **{"bodyEnergy(/min)": _maybe_float(r.get("bodyEnergy(/min)"))},
            ))

    if not ag_df.empty:
        for _, r in ag_df.iterrows():
            world.agents.append(AgentProfile(
                agent_id=str(r.get("agent_id")),
                agent_name=_maybe_str(r.get("agent_name")),
                describe=_maybe_str(r.get("describe")),
            ))

    world.entity_index = _build_entity_index(world)
    world.agent_index = {a.agent_id: a.model_dump() for a in world.agents}

    world.entity_name_index = {}
    for l in world.locations:
        world.entity_name_index[l.node_id] = l.node_name or l.node_id
    for f in world.facilities:
        world.entity_name_index[f.node_id] = f.node_name or f.node_id
    for o in world.objects:
        world.entity_name_index[o.node_id] = o.node_name or o.node_id

    world.agent_name_index = {}
    for a in world.agents:
        world.agent_name_index[a.agent_id] = a.agent_name or a.agent_id
        
    return world


def _build_entity_index(world: WorldModel) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for l in world.locations:
        idx[l.node_id] = {"type": "location", **l.model_dump()}
    for f in world.facilities:
        idx[f.node_id] = {"type": "facility", **f.model_dump(by_alias=True)}
    for o in world.objects:
        idx[o.node_id] = {"type": "object", **o.model_dump(by_alias=True)}
    return idx


def _maybe_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s if s else None


def _maybe_float(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _maybe_bool(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None