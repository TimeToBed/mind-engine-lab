#models.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# ---- World model (imported from Excel) ----
class Location(BaseModel):
    node_id: str
    node_name: str
    node_type: Literal["location"] = "location"
    location_type: Optional[str] = None
    parent_id: Optional[str] = None
    path: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None


class Facility(BaseModel):
    node_id: str
    node_name: str
    node_type: Literal["facility"] = "facility"
    facility_type: Optional[str] = None
    parent_id: Optional[str] = None
    capacity: Optional[float] = None
    occupied: Optional[float] = None
    bodyEnergy_per_min: Optional[float] = Field(default=None, alias="bodyEnergy(/min)")


class WorldObject(BaseModel):
    node_id: str
    node_name: str
    node_type: Literal["object"] = "object"
    object_type: Optional[str] = None
    parent_id: Optional[str] = None
    is_consumable: Optional[bool] = None
    bodyEnergy_per_min: Optional[float] = Field(default=None, alias="bodyEnergy(/min)")


class AgentProfile(BaseModel):
    agent_id: str
    agent_name: Optional[str] = None
    describe: Optional[str] = None


class WorldModel(BaseModel):
    locations: List[Location] = Field(default_factory=list)
    facilities: List[Facility] = Field(default_factory=list)
    objects: List[WorldObject] = Field(default_factory=list)
    agents: List[AgentProfile] = Field(default_factory=list)

    # derived indexes (facts only)
    entity_index: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_index: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    entity_name_index: Dict[str, str] = Field(default_factory=dict)
    agent_name_index: Dict[str, str] = Field(default_factory=dict)


# ---- Observation & evidence profile (L1 output / L2 input) ----
class Evidence(BaseModel):
    duration_s: float = 0.0
    progress_count: int = 0
    # 允许接收 int 或 None，且默认值为 None
    capacity: Optional[int] = None 
    occupied: Optional[bool] = None
    action: Optional[str] = None

class Deltas(BaseModel):
    # objective physical deltas
    agent_phys: Dict[str, float] = Field(default_factory=dict)


class EvidenceProfile(BaseModel):
    evidence: Evidence = Field(default_factory=Evidence)
    deltas: Deltas = Field(default_factory=Deltas)
    # 注意：用户未定义 Action 时，此键必须不存在
    related_actions: Optional[List[str]] = None


class Observation(BaseModel):
    obs_id: str
    t_start: str
    t_end: str

    subjects: str   # name
    location: List[str] = Field(default_factory=list)   # name

    evidence_profile: EvidenceProfile = Field(default_factory=EvidenceProfile)
    status: Literal["draft", "final"] = "final"


class ObserverState(BaseModel):
    agent_name: str
    phys_state: Dict[str, float] = Field(default_factory=dict)
    psych_state: Dict[str, float] = Field(default_factory=dict)
    cog_attr: Dict[str, Any] = Field(default_factory=dict)

class L2Input(BaseModel):
    observation: Observation
    observer_state: ObserverState

# ---- Config: ActionMeaning + plugins + generator config ----
class MeaningDim(BaseModel):
    desc: Optional[str] = None
    value: str  # DSL expression
    polarity: Literal["+", "-"] = "+"
    weight: float = 1.0


class ActionMeaning(BaseModel):
    action: str
    meas: Dict[str, MeaningDim] = Field(default_factory=dict)  # dims aligned with stimulus_vec


class PluginConfig(BaseModel):
    action_aggregator: str = "weighted_sum"
    arbiter: str = "default"
    params: Dict[str, Any] = Field(default_factory=dict)


class ObservationGenConfig(BaseModel):
    seed: int = 42
    n: int = 20
    duration_s_range: Tuple[float, float] = (5.0, 50.0)
    progress_count_range: Tuple[int, int] = (0, 20)


class EngineConfig(BaseModel):
    schema_version: str = "0.1"
    stimulus_dims: List[str] = Field(default_factory=lambda: ["achievement", "info_exposure", "comfort"])
    intent_actions: List[str] = Field(default_factory=lambda: ["get_info", "socialize", "exercise", "rest", "create"])

    # 新增：建立 Intent -> List[node_id] 的映射关系
    intent_allocations: Dict[str, List[str]] = Field(default_factory=dict)

    action_meanings: List[ActionMeaning] = Field(default_factory=list)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    observers: List[ObserverState] = Field(default_factory=list)
    observation_gen: ObservationGenConfig = Field(default_factory=ObservationGenConfig)

# ---- L2/L3/L4 outputs ----
class MotivationSignal(BaseModel):
    obs_id: str
    agent_name: str
    stimulus_vec: Dict[str, float] = Field(default_factory=dict)
    drive: Dict[str, float] = Field(default_factory=dict)
    resist: Dict[str, float] = Field(default_factory=dict)


class MotivationDecision(BaseModel):
    obs_id: str
    agent_name: str
    top_motivations: Dict[str, float] = Field(default_factory=dict)  # need scores
    level: str = "low"  # low/mid/high


class L4Utterance(BaseModel):
    obs_id: str
    agent_name: str
    utterance: str


class RunResult(BaseModel):
    motivation_signal: MotivationSignal
    motivation_decision: MotivationDecision
    l4: L4Utterance
    trace: Dict[str, Any] = Field(default_factory=dict)