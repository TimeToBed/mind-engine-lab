from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from mind_engine_lab.models import EngineConfig, ObserverState, L2Input
from mind_engine_lab.world_import import import_world_from_xlsx
from mind_engine_lab.obs_gen import generate_observations
from mind_engine_lab.engine import run_engine
from mind_engine_lab.yaml_io import dump_yaml


# -----------------------------
# Dict <-> DataEditor helpers
# -----------------------------
def _dict_to_kv_df(d: Dict[str, Any], value_type: str) -> pd.DataFrame:
    """
    value_type: "number" | "text"
    """
    rows = []
    d = d or {}
    for k, v in d.items():
        if value_type == "number":
            try:
                v = float(v)
            except Exception:
                v = 0.0
        else:
            v = "" if v is None else str(v)
        rows.append({"key": str(k), "value": v})
    if not rows:
        # keep at least one empty row for UX
        rows = [{"key": "", "value": 0.0 if value_type == "number" else ""}]
    return pd.DataFrame(rows)


def _kv_df_to_dict(df: pd.DataFrame, value_type: str) -> Dict[str, Any]:
    """
    Convert edited kv DataFrame -> dict
    - drops empty keys
    - for number: converts to float, invalid -> ignored (or 0.0)
    - for text: converts to str
    """
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        k = str(r.get("key", "")).strip()
        if not k:
            continue
        v = r.get("value", None)
        if value_type == "number":
            try:
                out[k] = float(v)
            except Exception:
                # ignore invalid numeric cell
                continue
        else:
            out[k] = "" if v is None else str(v)
    return out


def _edit_kv_table(
    title: str,
    state_key: str,
    value_type: str,
    default_dict: Dict[str, Any],
    height: int = 220,
) -> Dict[str, Any]:
    """
    Render a data_editor for a dict and store in session_state[state_key] as dict.
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = dict(default_dict or {})

    df_init = _dict_to_kv_df(st.session_state[state_key], value_type=value_type)

    st.caption(title)
    edited_df = st.data_editor(
        df_init,
        width="stretch",
        height=height,
        num_rows="dynamic",  # allow add/remove rows
        column_config={
            "key": st.column_config.TextColumn("key", required=False),
            "value": (
                st.column_config.NumberColumn("value", step=0.01)
                if value_type == "number"
                else st.column_config.TextColumn("value")
            ),
        },
        hide_index=True,
        key=f"editor_{state_key}",
    )

    parsed = _kv_df_to_dict(edited_df, value_type=value_type)
    st.session_state[state_key] = parsed
    return parsed


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(page_title="Mind Engine Lab", layout="wide")
    st.title("Mind Engine Lab (L1–L4) — Researcher WebUI")

    if "world" not in st.session_state:
        st.session_state["world"] = None
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = EngineConfig()
    if "observations" not in st.session_state:
        st.session_state["observations"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # Observer defaults
    if "observer_agent_name" not in st.session_state:
        st.session_state["observer_agent_name"] = "贾小红"

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("1) World")
        up = st.file_uploader("Upload world xlsx", type=["xlsx"])
        if up is not None:
            try:
                p = Path(".streamlit_tmp_world.xlsx")
                p.write_bytes(up.getvalue())
                world = import_world_from_xlsx(str(p))
                st.session_state["world"] = world
                st.success(
                    f"Imported: {len(world.locations)} loc, {len(world.facilities)} fac, "
                    f"{len(world.objects)} obj, {len(world.agents)} agents"
                )
            except Exception as e:
                st.error("Import failed")
                st.exception(e)

        if st.session_state["world"] is not None:
            st.download_button(
                "Download world.json",
                data=json.dumps(
                    st.session_state["world"].model_dump(by_alias=True),
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="world.json",
                mime="application/json",
            )

        st.header("2) Observation Generator")
        gcfg = st.session_state["cfg"].observation_gen
        gcfg.seed = int(st.number_input("seed", value=int(gcfg.seed), step=1))
        gcfg.n = int(st.number_input("n", value=int(gcfg.n), step=1, min_value=1, max_value=500))
        lo, hi = gcfg.duration_s_range
        lo = float(st.number_input("duration_s min", value=float(lo), step=10.0))
        hi = float(st.number_input("duration_s max", value=float(hi), step=10.0))
        gcfg.duration_s_range = (lo, max(hi, lo + 1.0))
        gcfg.choose_objects_max = int(
            st.number_input("max objects per obs", value=int(gcfg.choose_objects_max), step=1, min_value=0, max_value=10)
        )

        if st.button("Generate Observations", disabled=st.session_state["world"] is None):
            try:
                obs_list = generate_observations(st.session_state["world"], gcfg)
                st.session_state["observations"] = obs_list
                st.success(f"Generated {len(obs_list)} observations")
            except Exception as e:
                st.error("Generate observations failed")
                st.exception(e)

        if st.session_state["observations"]:
            st.download_button(
                "Download observations.json",
                data=json.dumps(
                    [o.model_dump(exclude_none=True) for o in st.session_state["observations"]],
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="observations.json",
                mime="application/json",
            )

    # ---------------- Main Layout ----------------
    col1, col2 = st.columns([1, 1])

    # --------- Config YAML (advanced) ----------
    with col1:
        st.header("3) Engine Config (YAML)")
        st.caption("建议：日常用右侧 ObserverState/测试；YAML 仅作为高级编辑入口。")

        cfg_yaml = dump_yaml(st.session_state["cfg"].model_dump(by_alias=True))
        edited = st.text_area("config.yaml", value=cfg_yaml, height=420)

        if st.button("Apply YAML"):
            try:
                import yaml as _yaml

                data = _yaml.safe_load(edited) or {}
                st.session_state["cfg"] = EngineConfig.model_validate(data)
                st.success("Config applied")
            except Exception:
                st.error("Failed to parse/apply YAML")
                st.exception(Exception("YAML parse/apply error"))

        st.download_button(
            "Compile & Download config.yaml",
            data=edited.encode("utf-8"),
            file_name="config.yaml",
            mime="text/yaml",
        )

    # --------- Test Runner ----------
    with col2:
        st.header("4) Test Runner")

        # ---- Observer State Editor (visual tables) ----
        st.subheader("Observer State (independent of Observation)")

        world = st.session_state["world"]
        agent_options: List[str] = []
        if world is not None and getattr(world, "agents", None):
            for a in world.agents:
                if getattr(a, "agent_name", None):
                    agent_options.append(a.agent_name)
                else:
                    agent_options.append(a.agent_id)

        if agent_options:
            idx0 = agent_options.index(st.session_state["observer_agent_name"]) if st.session_state["observer_agent_name"] in agent_options else 0
            sel = st.selectbox("observer_state.agent_name", options=agent_options, index=idx0)
            st.session_state["observer_agent_name"] = sel
        else:
            st.session_state["observer_agent_name"] = st.text_input(
                "observer_state.agent_name",
                value=st.session_state["observer_agent_name"],
            )

        # default dicts
        default_phys = {"bodyEnergy": 0.62, "mindEnergy": 0.51}
        default_psych = {"comfort": 0.55, "achievement": 0.40, "belonging": 0.30, "security": 0.65}
        default_cog = {"身份": "学生", "性格": "内向"}

        with st.expander("phys_state", expanded=True):
            phys_state = _edit_kv_table(
                title="phys_state: 数值型 (float)",
                state_key="observer_phys_state",
                value_type="number",
                default_dict=default_phys,
                height=220,
            )

        with st.expander("psych_state", expanded=True):
            psych_state = _edit_kv_table(
                title="psych_state: 数值型 (float)",
                state_key="observer_psych_state",
                value_type="number",
                default_dict=default_psych,
                height=260,
            )

        with st.expander("cog_attr", expanded=True):
            cog_attr = _edit_kv_table(
                title="cog_attr: 文本型 (string)",
                state_key="observer_cog_attr",
                value_type="text",
                default_dict=default_cog,
                height=260,
            )

        observer_state = ObserverState(
            agent_name=st.session_state["observer_agent_name"],
            phys_state=phys_state,
            psych_state=psych_state,
            cog_attr=cog_attr,
        )

        st.divider()

        # ---- Observation selection ----
        if st.session_state["observations"]:
            idx = int(
                st.number_input(
                    "Observation index",
                    value=0,
                    step=1,
                    min_value=0,
                    max_value=len(st.session_state["observations"]) - 1,
                )
            )
            obs = st.session_state["observations"][idx]
            st.subheader("Selected Observation")
            st.json(obs.model_dump(exclude_none=True))

            if st.button("Run Test", disabled=st.session_state["world"] is None):
                try:
                    l2_in = L2Input(observation=obs, observer_state=observer_state)
                    res = run_engine(st.session_state["cfg"], st.session_state["world"], l2_in)
                    st.session_state["last_result"] = res
                    st.success("Run completed")
                except Exception as e:
                    st.error("Run failed")
                    st.exception(e)
        else:
            st.info("Generate observations first (sidebar).")

    # ---------------- Outputs ----------------
    st.divider()
    st.header("5) Outputs & Trace")

    res = st.session_state["last_result"]
    if res is not None:
        st.subheader("L2: motivation_signal")
        st.json(res.motivation_signal.model_dump())
        st.subheader("L3: motivation_decision")
        st.json(res.motivation_decision.model_dump())
        st.subheader("L4: utterance")
        st.write(res.l4.utterance)
        with st.expander("Trace", expanded=False):
            st.json(res.trace)
    else:
        st.info("Run a test to see outputs.")


if __name__ == "__main__":
    main()