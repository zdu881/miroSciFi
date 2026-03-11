from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    observation_analysis: str = Field(
        description="对方刚才的话语、动作或沉默里，我察觉到的危险、漏洞或权力结构。"
    )
    emotional_shift: str = Field(
        description="我当下的生理反应和情绪底色，例如胃部抽紧、麻木、想发作却压住。"
    )
    hidden_agenda: str = Field(
        description="我真正想达成的目的，必须和我的核心创伤或终极渴望相关。"
    )
    micro_expression: str = Field(
        description="我试图掩饰但仍微微流露出来的表情、停顿或小动作。"
    )
    action_and_dialogue: str = Field(
        description="我实际做出的动作和说出的话，写成一个紧凑段落，不要解释。"
    )


class CharacterProfile(BaseModel):
    name: str
    role: str
    worldview: str
    core_goal: str
    core_wound: str
    ultimate_desire: str
    public_mask: str
    system_prompt: str


class CharacterResourceState(BaseModel):
    stats: dict[str, int]
    decay_per_round: dict[str, int]
    failure_condition: str
    pressure_note: str


class BeatItem(BaseModel):
    round_index: int
    dramatic_function: str
    forced_event: str
    target_shift: str


class ShowrunnerPlan(BaseModel):
    scene_brief: str
    scene_purpose: str
    target_ending: str
    core_conflict: str
    hidden_foreshadowing: str
    tone_guardrail: str
    forced_beats: list[BeatItem]


class SymbolismCue(BaseModel):
    motif: str
    sensory_surface: str
    emotional_mapping: str
    usage_instruction: str


class SymbolismPlan(BaseModel):
    scene_subtext: str
    imagery_cues: list[SymbolismCue]
    gesture_rewrites: list[str]
    forbidden_explicit_phrases: list[str]


class PublicTurnRecord(BaseModel):
    speaker: str
    round_index: int
    micro_expression: str
    action_and_dialogue: str


class CharacterSceneEvent(BaseModel):
    event_type: Literal["character"] = "character"
    speaker: str
    round_index: int
    observation_analysis: str
    emotional_shift: str
    hidden_agenda: str
    micro_expression: str
    action_and_dialogue: str
    resource_snapshot: dict[str, int]


class DirectorSceneEvent(BaseModel):
    event_type: Literal["director"] = "director"
    speaker: str = "Director"
    round_index: int
    beat_focus: str
    content: str


class SceneState(TypedDict):
    world_context: str
    scene_brief: str
    showrunner_plan: dict[str, Any]
    short_term_window: list[dict[str, Any]]
    public_trace: list[dict[str, Any]]
    private_memory: dict[str, list[dict[str, Any]]]
    dynamic_relationships: dict[str, dict[str, str]]
    core_anchors: dict[str, dict[str, str]]
    resource_state: dict[str, dict[str, Any]]
    scene_log: list[dict[str, Any]]
    director_log: list[str]
    symbolism_plan: dict[str, Any]
    subtext_guide: str
    scene_data: str
    chapter_text: str
    chapter_target: str
    turn_count: int
    max_turns: int
    short_window_size: int
