from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from mind_engine_lab.models import (
    EngineConfig,
    ObserverState,
    L2Input,
    ActionMeaning,
    MeaningDim,
)
from mind_engine_lab.world_import import import_world_from_xlsx
from mind_engine_lab.obs_gen import generate_observations
from mind_engine_lab.engine import run_engine
from mind_engine_lab.yaml_io import load_yaml_to_cfg, compile_yaml

# -----------------------------
# Helpers: kv table editors
# -----------------------------
def dict_to_kv_df(d: Dict[str, Any], value_type: str) -> pd.DataFrame:
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
        rows = [{"key": "", "value": 0.0 if value_type == "number" else ""}]

    df = pd.DataFrame(rows)

    # 强制 RangeIndex，避免 dynamic rows + hide_index 抖动/警告
    df = df.reset_index(drop=True)

    # 固定 dtype，减少 NA/类型漂移
    if value_type == "number":
        df = df.astype({"key": "string", "value": "float64"})
    else:
        df = df.astype({"key": "string", "value": "string"})

    return df


def kv_df_to_dict(df: pd.DataFrame, value_type: str) -> Dict[str, Any]:
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
                continue
        else:
            out[k] = "" if v is None else str(v)
    return out


def edit_kv_table(
    title: str,
    state_key: str,
    value_type: str,
    default: Dict[str, Any],
    height: int = 220,
):
    """
    最稳版本：data_editor 放进 form，用户点 Save 才写回 state_key。
    """
    df_key = f"{state_key}__df"
    form_key = f"form_{state_key}"
    editor_key = f"editor_{state_key}"

    if df_key not in st.session_state:
        init_dict = st.session_state.get(state_key, default or {})
        st.session_state[df_key] = dict_to_kv_df(init_dict, value_type=value_type)

    st.caption(title)

    # 每次渲染前保证 RangeIndex（重要：dynamic + hide_index）
    st.session_state[df_key] = st.session_state[df_key].reset_index(drop=True)

    with st.form(form_key, clear_on_submit=False):
        edited_df = st.data_editor(
            st.session_state[df_key],
            width="stretch",
            height=height,
            num_rows="dynamic",
            column_config={
                "key": st.column_config.TextColumn("key"),
                "value": st.column_config.NumberColumn("value", step=0.01)
                if value_type == "number"
                else st.column_config.TextColumn("value"),
            },
            hide_index=True,
            key=editor_key,
        )

        submitted = st.form_submit_button("Save")

    if submitted:
        edited_df = edited_df.reset_index(drop=True)

        # 维持 dtype（避免提交后 dtype 漂移）
        if value_type == "number":
            edited_df = edited_df.astype({"key": "string", "value": "float64"})
        else:
            edited_df = edited_df.astype({"key": "string", "value": "string"})

        st.session_state[df_key] = edited_df
        st.session_state[state_key] = kv_df_to_dict(edited_df, value_type=value_type)

    return st.session_state.get(state_key, {})


# -----------------------------
# Helpers: Intent actions editor
# -----------------------------
def actions_to_df(actions: List[str]) -> pd.DataFrame:
    actions = actions or []
    if not actions:
        actions = [""]
    df = pd.DataFrame([{"action": a} for a in actions]).reset_index(drop=True)
    df = df.astype({"action": "string"})
    return df


def df_to_actions(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        a = str(r.get("action", "")).strip()
        if a:
            out.append(a)
    # 去重但保持顺序
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


# -----------------------------
# Helpers: ActionMeaning editor
# -----------------------------
def build_actionmeaning_index(cfg: EngineConfig) -> Dict[str, ActionMeaning]:
    idx: Dict[str, ActionMeaning] = {}
    for am in cfg.action_meanings:
        idx[am.action] = am
    return idx


def meaning_to_df(am: ActionMeaning, stimulus_dims: List[str]) -> pd.DataFrame:
    rows = []
    meas = am.meas or {}
    for d in stimulus_dims:
        md = meas.get(d)
        if md is None:
            rows.append({"dim": d, "desc": "", "value": "", "polarity": "+", "weight": 1.0})
        else:
            rows.append(
                {
                    "dim": d,
                    "desc": md.desc or "",
                    "value": md.value or "",
                    "polarity": md.polarity,
                    "weight": float(md.weight),
                }
            )

    df = pd.DataFrame(rows).reset_index(drop=True)

    # 固定 dtype（减少抖动/重建）
    df = df.astype(
        {
            "dim": "string",
            "desc": "string",
            "value": "string",
            "polarity": "string",
            "weight": "float64",
        }
    )
    return df


def df_to_meaning(df: pd.DataFrame) -> Dict[str, MeaningDim]:
    out: Dict[str, MeaningDim] = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        dim = str(r.get("dim", "")).strip()
        if not dim:
            continue
        value = str(r.get("value", "")).strip()
        # 允许 value 为空：表示该 dim 未定义
        if not value:
            continue
        desc = str(r.get("desc", "")).strip()
        polarity = str(r.get("polarity", "+")).strip()
        if polarity not in {"+", "-"}:
            polarity = "+"
        try:
            weight = float(r.get("weight", 1.0))
        except Exception:
            weight = 1.0
        out[dim] = MeaningDim(desc=desc or None, value=value, polarity=polarity, weight=weight)
    return out



def reset_ui_caches():
    # 1) 所有 data_editor 的 df 缓存
    df_keys = [
        "intent_actions__df",
        "stimulus_dims__df",
        "observer_phys_state__df",
        "observer_psych_state__df",
        "observer_cog_attr__df",
        "plugin_params_action__df",
        "plugin_params_arbiter__df",
    ]
    for k in df_keys:
        if k in st.session_state:
            del st.session_state[k]

    # 2) ActionMeaning 每个 action 的 df 缓存：meaning_df__{act}
    for k in list(st.session_state.keys()):
        if str(k).startswith("meaning_df__"):
            del st.session_state[k]

    # 3) multiselect 的 UI 状态（Interactive Allocation）
    # 你为每个 action 用 key=f"alloc_{act}"，导入新 cfg 后这些也要清，否则 default 不会按新 cfg 刷新
    for k in list(st.session_state.keys()):
        if str(k).startswith("alloc_"):
            del st.session_state[k]

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Mind Engine Lab", layout="wide")
    st.title("Mind Engine Lab — Researcher WebUI (MVP)")

    # Session state init
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
    if "observer_phys_state" not in st.session_state:
        st.session_state["observer_phys_state"] = {"bodyEnergy": 0.62, "mindEnergy": 0.51}
    if "observer_psych_state" not in st.session_state:
        st.session_state["observer_psych_state"] = {
            "comfort": 0.55,
            "achievement": 0.40,
            "belonging": 0.30,
            "security": 0.65,
        }
    if "observer_cog_attr" not in st.session_state:
        st.session_state["observer_cog_attr"] = {"身份": "学生", "性格": "内向"}

    # -----------------------------
    # (1) World Import + (2) Obs Gen
    # -----------------------------
    with st.sidebar:
        st.header("导入世界配置 (xlsx)")
        up = st.file_uploader("Upload Object_Table_world_config_v0_4.xlsx", type=["xlsx"])
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
                data=json.dumps(st.session_state["world"].model_dump(by_alias=True), ensure_ascii=False, indent=2),
                file_name="world.json",
                mime="application/json",
            )

        st.divider()
        st.header("导入配置 (YAML)")
        cfg_up = st.file_uploader("Upload compiled_config.yaml", type=["yaml", "yml"])

        if cfg_up is not None:
            yaml_text = cfg_up.getvalue().decode("utf-8")

            coly1, coly2 = st.columns(2)
            with coly1:
                if st.button("Load YAML (Apply)"):
                    try:
                        new_cfg = load_yaml_to_cfg(yaml_text)
                        st.session_state["cfg"] = new_cfg

                        # 导入后：清空 UI 缓存，保证所有 data_editor/multiselect 用新 cfg 重建
                        reset_ui_caches()

                        st.success("Config loaded. Refreshing UI...")
                        st.rerun()
                    except Exception as e:
                        st.error("Load failed")
                        st.exception(e)

            with coly2:
                with st.expander("Preview uploaded YAML", expanded=False):
                    st.code(yaml_text, language="yaml")

        st.divider()
        st.header("Observations 生成配置")
        gcfg = st.session_state["cfg"].observation_gen
        gcfg.seed = int(st.number_input("seed", value=int(gcfg.seed), step=1))
        gcfg.n = int(st.number_input("n", value=int(gcfg.n), step=1, min_value=1, max_value=500))
        lo, hi = gcfg.duration_s_range
        lo = float(st.number_input("duration_s min", value=float(lo), step=10.0))
        hi = float(st.number_input("duration_s max", value=float(hi), step=10.0))
        gcfg.duration_s_range = (lo, max(hi, lo + 1.0))

        if st.button("Generate Observations", disabled=st.session_state["world"] is None):
            try:
                obs_list = generate_observations(st.session_state["world"], st.session_state["cfg"])
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

    # -----------------------------
    # Main panels: (3) Config  (4) Compile  (5) Test
    # -----------------------------
    cfg: EngineConfig = st.session_state["cfg"]

    st.header("研究者配置")

    tab_obs, tab_actions, tab_plugins = st.tabs(["Observer 定义", "Intent + ActionMeaning", "Plugins (可选)"])

    # -------- Observer tab --------
    with tab_obs:
        st.subheader("ObserverState（观察者状态）")

        world = st.session_state["world"]
        agent_options: List[str] = []
        if world is not None and getattr(world, "agents", None):
            for a in world.agents:
                agent_options.append(a.agent_name or a.agent_id)

        colA, colB = st.columns([1, 2])
        with colA:
            if agent_options:
                idx0 = agent_options.index(st.session_state["observer_agent_name"]) if st.session_state[
                    "observer_agent_name"
                ] in agent_options else 0
                st.session_state["observer_agent_name"] = st.selectbox("agent_name", options=agent_options, index=idx0)
            else:
                st.session_state["observer_agent_name"] = st.text_input(
                    "agent_name", value=st.session_state["observer_agent_name"]
                )

        col1, col2, col3 = st.columns(3)
        with col1:
            edit_kv_table("phys_state（数值）", "observer_phys_state", "number", st.session_state["observer_phys_state"], height=240)
        with col2:
            edit_kv_table("psych_state（数值）", "observer_psych_state", "number", st.session_state["observer_psych_state"], height=240)
        with col3:
            edit_kv_table("cog_attr（文本）", "observer_cog_attr", "text", st.session_state["observer_cog_attr"], height=240)

        st.info("注意：observer_state 与 observation.subjects 没有必然关系。")

    # -------- Actions + Meaning tab --------
    with tab_actions:
        st.subheader("Intent_actions（动作集合）")

        # 用 form 提交，避免编辑中途 rerun
        actions_df_key = "intent_actions__df"
        if actions_df_key not in st.session_state:
            st.session_state[actions_df_key] = actions_to_df(cfg.intent_actions)

        st.session_state[actions_df_key] = st.session_state[actions_df_key].reset_index(drop=True).astype({"action": "string"})

        with st.form("form_intent_actions", clear_on_submit=False):
            actions_df_edit = st.data_editor(
                st.session_state[actions_df_key],
                width="stretch",
                height=220,
                num_rows="dynamic",
                column_config={"action": st.column_config.TextColumn("action")},
                hide_index=True,
                key="editor_intent_actions",
            )
            submitted = st.form_submit_button("Apply Intent_actions")

        if submitted:
            actions_df_edit = actions_df_edit.reset_index(drop=True).astype({"action": "string"})
            st.session_state[actions_df_key] = actions_df_edit
            cfg.intent_actions = df_to_actions(actions_df_edit)

        st.divider()
        st.subheader("Interactive Allocation (交互分配)")

        if not st.session_state["world"]:
            st.warning("请先在侧边栏导入世界配置 (xlsx) 以进行物体分配。")
        elif not cfg.intent_actions:
            st.warning("请先定义 Intent_actions。")
        else:
            # 1. 准备可选物体列表 (Facilities + Objects + Agents)
            world = st.session_state["world"]
            available_entities = []
            entity_display_map = {}  # node_id -> "name (id)"

            for f in world.facilities:
                label = f"设施: {f.node_name} ({f.node_id})"
                available_entities.append(f.node_id)
                entity_display_map[f.node_id] = label

            for o in world.objects:
                label = f"物体: {o.node_name} ({o.node_id})"
                available_entities.append(o.node_id)
                entity_display_map[o.node_id] = label

            for a in world.agents:
                label = f"智能体: {a.agent_name} ({a.agent_id})"
                available_entities.append(a.agent_id)
                entity_display_map[a.agent_id] = label

            if cfg.intent_allocations is None:
                cfg.intent_allocations = {}

            alloc_col1, alloc_col2 = st.columns(2)
            for i, act in enumerate(cfg.intent_actions):
                target_col = alloc_col1 if i % 2 == 0 else alloc_col2

                current_selected = cfg.intent_allocations.get(act, [])
                current_selected = [eid for eid in current_selected if eid in entity_display_map]

                selected_ids = target_col.multiselect(
                    f"分配物体给: {act}",
                    options=available_entities,
                    default=current_selected,
                    format_func=lambda x: entity_display_map.get(x, x),
                    key=f"alloc_{act}",
                )
                cfg.intent_allocations[act] = selected_ids

        st.divider()
        st.subheader("ActionMeaning（动作 → 维度 meas 公式）")

        # stimulus dims (platform/global) —— 用 form 提交
        st.caption("stimulus_dims（维度列表）")
        dims_df_key = "stimulus_dims__df"
        if dims_df_key not in st.session_state:
            st.session_state[dims_df_key] = pd.DataFrame([{"dim": d} for d in (cfg.stimulus_dims or [])] or [{"dim": ""}]).reset_index(drop=True).astype({"dim": "string"})

        st.session_state[dims_df_key] = st.session_state[dims_df_key].reset_index(drop=True).astype({"dim": "string"})

        with st.form("form_stimulus_dims", clear_on_submit=False):
            dims_df_edit = st.data_editor(
                st.session_state[dims_df_key],
                width="stretch",
                height=200,
                num_rows="dynamic",
                column_config={"dim": st.column_config.TextColumn("dim")},
                hide_index=True,
                key="editor_stimulus_dims",
            )
            dims_submitted = st.form_submit_button("Apply stimulus_dims")

        if dims_submitted:
            dims_df_edit = dims_df_edit.reset_index(drop=True).astype({"dim": "string"})
            st.session_state[dims_df_key] = dims_df_edit

            dims = []
            for _, r in dims_df_edit.iterrows():
                d = str(r.get("dim", "")).strip()
                if d:
                    dims.append(d)

            seen = set()
            dims2 = []
            for d in dims:
                if d not in seen:
                    dims2.append(d)
                    seen.add(d)
            cfg.stimulus_dims = dims2

        am_index = build_actionmeaning_index(cfg)

        if not cfg.intent_actions:
            st.warning("请先在上方添加至少一个 intent action。")
        elif not cfg.stimulus_dims:
            st.warning("请先添加至少一个 stimulus dim。")
        else:
            for act in cfg.intent_actions:
                if act not in am_index:
                    am_index[act] = ActionMeaning(action=act, meas={})

                with st.expander(f"Action: {act}", expanded=False):
                    am = am_index[act]

                    df_key = f"meaning_df__{act}"
                    form_key = f"form_meas__{act}"
                    editor_key = f"editor_meas_{act}"

                    if df_key not in st.session_state:
                        st.session_state[df_key] = meaning_to_df(am, cfg.stimulus_dims)

                    # 固定 RangeIndex + dtype
                    st.session_state[df_key] = st.session_state[df_key].reset_index(drop=True).astype(
                        {
                            "dim": "string",
                            "desc": "string",
                            "value": "string",
                            "polarity": "string",
                            "weight": "float64",
                        }
                    )

                    with st.form(form_key, clear_on_submit=False):
                        df_edit = st.data_editor(
                            st.session_state[df_key],
                            width="stretch",
                            height=260,
                            num_rows="fixed",  # 你这里是固定行（dims 行），OK
                            column_config={
                                "dim": st.column_config.TextColumn("dim", disabled=True),
                                "desc": st.column_config.TextColumn("desc"),
                                "value": st.column_config.TextColumn("value (DSL expr)"),
                                "polarity": st.column_config.SelectboxColumn("polarity", options=["+", "-"]),
                                "weight": st.column_config.NumberColumn("weight", step=0.1),
                            },
                            hide_index=True,
                            key=editor_key,
                        )
                        submitted_meas = st.form_submit_button("Save ActionMeaning")

                    if submitted_meas:
                        df_edit = df_edit.reset_index(drop=True).astype(
                            {
                                "dim": "string",
                                "desc": "string",
                                "value": "string",
                                "polarity": "string",
                                "weight": "float64",
                            }
                        )
                        st.session_state[df_key] = df_edit
                        am.meas = df_to_meaning(df_edit)

            # 写回 cfg.action_meanings（只保留 intent_actions 内的）
            cfg.action_meanings = [am_index[a] for a in cfg.intent_actions if a in am_index]

        st.info("related_actions：若未定义 ActionMeaning，可让 observation 中不出现该键。定义后可在生成/导入时填入。")

    # -------- Plugins tab --------
    with tab_plugins:
        st.subheader("Plugins（可选）")

        c1, c2 = st.columns(2)
        with c1:
            cfg.plugins.action_aggregator = st.selectbox(
                "ActionAggregator",
                options=["weighted_sum", "max_pool", "softmax_pool"],
                index=["weighted_sum", "max_pool", "softmax_pool"].index(cfg.plugins.action_aggregator)
                if cfg.plugins.action_aggregator in ["weighted_sum", "max_pool", "softmax_pool"]
                else 0,
            )
        with c2:
            cfg.plugins.arbiter = st.selectbox(
                "MotivationArbiter",
                options=["default"],
                index=0,
            )

        st.divider()
        st.caption("Plugin params（键值表）")

        if cfg.plugins.params is None:
            cfg.plugins.params = {}

        p_action = cfg.plugins.params.get("action_aggregator", {})
        p_arbiter = cfg.plugins.params.get("arbiter", {})

        colp1, colp2 = st.columns(2)
        with colp1:
            cfg.plugins.params["action_aggregator"] = edit_kv_table(
                "action_aggregator params（数值）",
                "plugin_params_action",
                "number",
                p_action,
                height=220,
            )
        with colp2:
            cfg.plugins.params["arbiter"] = edit_kv_table(
                "arbiter params（数值）",
                "plugin_params_arbiter",
                "number",
                p_arbiter,
                height=220,
            )

    # Save cfg back
    st.session_state["cfg"] = cfg

    st.header("配置 Compile（导出 YAML）")
    compiled = compile_yaml(cfg)
    st.download_button(
        "Download compiled_config.yaml",
        data=compiled.encode("utf-8"),
        file_name="compiled_config.yaml",
        mime="text/yaml",
    )
    with st.expander("Preview YAML", expanded=False):
        st.code(compiled, language="yaml")

    st.header(" 测试（L2 → L3 → L4）")

    # Build observer_state from UI tables
    observer_state = ObserverState(
        agent_name=st.session_state["observer_agent_name"],
        phys_state=st.session_state.get("observer_phys_state", {}),
        psych_state=st.session_state.get("observer_psych_state", {}),
        cog_attr=st.session_state.get("observer_cog_attr", {}),
    )

    if not st.session_state["observations"]:
        st.info("请先在左侧生成 Observations。")
        return

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

    colX, colY = st.columns([1, 1])
    with colX:
        st.subheader("Selected Observation")
        st.json(obs.model_dump(exclude_none=True))
    with colY:
        st.subheader("ObserverState")
        st.json(observer_state.model_dump(exclude_none=True))

    if st.button("Run Test", disabled=st.session_state["world"] is None):
        try:
            l2_in = L2Input(observation=obs, observer_state=observer_state)
            res = run_engine(st.session_state["cfg"], st.session_state["world"], l2_in)
            st.session_state["last_result"] = res
            st.success("Run completed")
        except Exception as e:
            st.error("Run failed")
            st.exception(e)

    st.divider()
    st.subheader("Outputs")

    res = st.session_state["last_result"]
    if res is not None:
        st.markdown("### L2: motivation_signal")
        st.json(res.motivation_signal.model_dump())
        st.markdown("### L3: motivation_decision")
        st.json(res.motivation_decision.model_dump())
        st.markdown("### L4: utterance")
        st.write(res.l4.utterance)
        with st.expander("Trace", expanded=False):
            st.json(res.trace)
    else:
        st.info("点击 Run Test 查看输出。")


if __name__ == "__main__":
    main()