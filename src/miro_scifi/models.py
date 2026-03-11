from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentOutput(BaseModel):
    inner_thought: str = Field(
        description="角色的内心独白，只能写给自己，其他角色绝对不可见。"
    )
    public_action: str = Field(
        description="角色当下可被他人观察到的动作、神态或环境互动。"
    )
    public_dialogue: str = Field(
        description="角色对外说出口的话；如果选择沉默，也要明确写出沉默方式。"
    )


class CharacterProfile(BaseModel):
    name: str
    role: str
    worldview: str
    core_goal: str
    system_prompt: str


class PublicTurnRecord(BaseModel):
    speaker: str
    round_index: int
    public_action: str
    public_dialogue: str


class CharacterSceneEvent(BaseModel):
    event_type: Literal["character"] = "character"
    speaker: str
    round_index: int
    inner_thought: str
    public_action: str
    public_dialogue: str


class DirectorSceneEvent(BaseModel):
    event_type: Literal["director"] = "director"
    speaker: str = "Director"
    round_index: int
    content: str


class SceneState(TypedDict):
    world_context: str
    public_history: list[dict[str, Any]]
    private_memory: dict[str, list[dict[str, Any]]]
    scene_log: list[dict[str, Any]]
    director_log: list[str]
    turn_count: int
    max_turns: int
