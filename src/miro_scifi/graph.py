from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .engine import CharacterEngine
from .models import (
    AgentOutput,
    CharacterProfile,
    CharacterSceneEvent,
    DirectorSceneEvent,
    PublicTurnRecord,
    SceneState,
)
from .prompts import DIRECTOR_OPENING_BROADCAST, WORLD_CONTEXT_PROMPT, build_checkpoint_broadcast


def create_initial_state(
    *,
    characters: list[CharacterProfile],
    max_turns: int = 3,
    world_context: str | None = None,
) -> SceneState:
    return {
        "world_context": (world_context or WORLD_CONTEXT_PROMPT).strip(),
        "public_history": [],
        "private_memory": {character.name: [] for character in characters},
        "scene_log": [],
        "director_log": [],
        "turn_count": 0,
        "max_turns": max_turns,
    }


def build_scene_graph(
    *,
    character_a: CharacterProfile,
    character_b: CharacterProfile,
    engine: CharacterEngine,
):
    graph = StateGraph(SceneState)
    graph.add_node("director_setup", director_setup)
    graph.add_node("character_a_turn", make_character_node(character_a, engine))
    graph.add_node("character_b_turn", make_character_node(character_b, engine))
    graph.add_node("director_checkpoint", director_checkpoint)

    graph.add_edge(START, "director_setup")
    graph.add_edge("director_setup", "character_a_turn")
    graph.add_edge("character_a_turn", "character_b_turn")
    graph.add_edge("character_b_turn", "director_checkpoint")
    graph.add_conditional_edges(
        "director_checkpoint",
        should_continue_scene,
        {"continue": "character_a_turn", "end": END},
    )
    return graph.compile()


def director_setup(state: SceneState) -> dict:
    if state["director_log"]:
        return {}

    opening_record = PublicTurnRecord(
        speaker="Director",
        round_index=0,
        public_action="采样站天花板的旧喇叭发出轻微电流噪音，灰蓝色政务字幕从终端底部缓慢滚动。",
        public_dialogue=DIRECTOR_OPENING_BROADCAST,
    ).model_dump()
    director_event = DirectorSceneEvent(
        round_index=0,
        content=DIRECTOR_OPENING_BROADCAST,
    ).model_dump()

    return {
        "world_context": state["world_context"].strip() or WORLD_CONTEXT_PROMPT,
        "public_history": state["public_history"] + [opening_record],
        "scene_log": state["scene_log"] + [director_event],
        "director_log": state["director_log"] + [DIRECTOR_OPENING_BROADCAST],
    }


def make_character_node(profile: CharacterProfile, engine: CharacterEngine):
    def node(state: SceneState) -> dict:
        output = engine.invoke(profile=profile, state=state)
        return apply_character_output(state, profile.name, output)

    return node


def apply_character_output(
    state: SceneState,
    speaker: str,
    output: AgentOutput,
) -> dict:
    round_index = state["turn_count"] + 1
    public_record = PublicTurnRecord(
        speaker=speaker,
        round_index=round_index,
        public_action=output.public_action,
        public_dialogue=output.public_dialogue,
    ).model_dump()
    private_record = CharacterSceneEvent(
        speaker=speaker,
        round_index=round_index,
        inner_thought=output.inner_thought,
        public_action=output.public_action,
        public_dialogue=output.public_dialogue,
    ).model_dump()

    updated_private_memory = {
        name: list(memory)
        for name, memory in state["private_memory"].items()
    }
    updated_private_memory.setdefault(speaker, []).append(private_record)

    return {
        "public_history": state["public_history"] + [public_record],
        "private_memory": updated_private_memory,
        "scene_log": state["scene_log"] + [private_record],
    }


def director_checkpoint(state: SceneState) -> dict:
    next_turn = state["turn_count"] + 1
    broadcast = build_checkpoint_broadcast(next_turn, state["max_turns"])
    public_record = PublicTurnRecord(
        speaker="Director",
        round_index=next_turn,
        public_action="走廊深处的屏幕再度亮起，保安靴底摩擦地面的声音从玻璃门外传进来。",
        public_dialogue=broadcast,
    ).model_dump()
    director_event = DirectorSceneEvent(
        round_index=next_turn,
        content=broadcast,
    ).model_dump()

    return {
        "turn_count": next_turn,
        "public_history": state["public_history"] + [public_record],
        "scene_log": state["scene_log"] + [director_event],
        "director_log": state["director_log"] + [broadcast],
    }


def should_continue_scene(state: SceneState) -> Literal["continue", "end"]:
    if state["turn_count"] >= state["max_turns"]:
        return "end"
    return "continue"
