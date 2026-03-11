from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .engine import CharacterEngine, ShowrunnerEngine, SymbolismEngine
from .models import (
    AgentAction,
    CharacterProfile,
    CharacterSceneEvent,
    DirectorSceneEvent,
    PublicTurnRecord,
    SceneState,
)
from .prompts import (
    DEFAULT_SCENE_BRIEF,
    DIRECTOR_OPENING_BROADCAST,
    WORLD_CONTEXT_PROMPT,
    default_resource_state,
    format_symbolism_plan,
)
from .writer import SceneWriter, build_scene_data, write_chapter


def create_initial_state(
    *,
    characters: list[CharacterProfile],
    max_turns: int = 3,
    scene_brief: str = DEFAULT_SCENE_BRIEF,
    world_context: str | None = None,
    short_window_size: int = 4,
    chapter_target: str = "1000 字左右",
) -> SceneState:
    return {
        "world_context": (world_context or WORLD_CONTEXT_PROMPT).strip(),
        "scene_brief": scene_brief,
        "showrunner_plan": {},
        "short_term_window": [],
        "public_trace": [],
        "private_memory": {character.name: [] for character in characters},
        "dynamic_relationships": initialize_relationships(characters),
        "core_anchors": {
            character.name: {
                "core_wound": character.core_wound,
                "ultimate_desire": character.ultimate_desire,
                "public_mask": character.public_mask,
            }
            for character in characters
        },
        "resource_state": {
            name: resource.model_dump()
            for name, resource in default_resource_state().items()
        },
        "scene_log": [],
        "director_log": [],
        "symbolism_plan": {},
        "subtext_guide": "",
        "scene_data": "",
        "chapter_text": "",
        "chapter_target": chapter_target,
        "turn_count": 0,
        "max_turns": max_turns,
        "short_window_size": short_window_size,
    }


def build_scene_graph(
    *,
    character_a: CharacterProfile,
    character_b: CharacterProfile,
    character_engine: CharacterEngine,
    showrunner_engine: ShowrunnerEngine,
    symbolism_engine: SymbolismEngine,
    writer: SceneWriter,
):
    characters = [character_a, character_b]
    graph = StateGraph(SceneState)
    graph.add_node(
        "showrunner_node",
        make_showrunner_node(characters, showrunner_engine),
    )
    graph.add_node("director_setup", director_setup)
    graph.add_node(
        "character_a_turn",
        make_character_node(character_a, character_engine),
    )
    graph.add_node(
        "character_b_turn",
        make_character_node(character_b, character_engine),
    )
    graph.add_node("director_checkpoint", director_checkpoint)
    graph.add_node("symbolism_node", make_symbolism_node(symbolism_engine))
    graph.add_node("writer_node", make_writer_node(writer))

    graph.add_edge(START, "showrunner_node")
    graph.add_edge("showrunner_node", "director_setup")
    graph.add_edge("director_setup", "character_a_turn")
    graph.add_edge("character_a_turn", "character_b_turn")
    graph.add_edge("character_b_turn", "director_checkpoint")
    graph.add_conditional_edges(
        "director_checkpoint",
        should_continue_scene,
        {"continue": "character_a_turn", "subtext": "symbolism_node"},
    )
    graph.add_edge("symbolism_node", "writer_node")
    graph.add_edge("writer_node", END)
    return graph.compile()


def initialize_relationships(
    characters: list[CharacterProfile],
) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for observer in characters:
        mapping[observer.name] = {}
        for target in characters:
            if observer.name == target.name:
                continue
            if "矿工" in observer.role and "审核员" in target.role:
                mapping[observer.name][target.name] = "把对方视作掐着自己账户命门的制度接口。"
            elif "审核员" in observer.role and "矿工" in target.role:
                mapping[observer.name][target.name] = "把对方视作高波动、可替代、需要分级处理的样本提供者。"
            else:
                mapping[observer.name][target.name] = "陌生、互不信任、只在制度流程中接触。"
    return mapping


def make_showrunner_node(
    characters: list[CharacterProfile],
    engine: ShowrunnerEngine,
):
    def node(state: SceneState) -> dict:
        if state["showrunner_plan"]:
            return {}
        plan = engine.plan(
            scene_brief=state["scene_brief"],
            world_context=state["world_context"],
            characters=characters,
            max_turns=state["max_turns"],
        )
        return {"showrunner_plan": plan.model_dump()}

    return node


def director_setup(state: SceneState) -> dict:
    if state["director_log"]:
        return {}

    opening_text = (
        f"{DIRECTOR_OPENING_BROADCAST} "
        f"审核台边上搁着一只杯沿带细裂的保温杯，没人提它。"
    )
    opening_record = PublicTurnRecord(
        speaker="Director",
        round_index=0,
        micro_expression="采样站天花板的旧喇叭发出轻微电流噪音，玻璃门上的雾气迟迟不散。",
        action_and_dialogue=opening_text,
    ).model_dump()
    director_event = DirectorSceneEvent(
        round_index=0,
        beat_focus="场景开场",
        content=opening_text,
    ).model_dump()

    return append_public_and_director(state, opening_record, director_event, opening_text)


def make_character_node(profile: CharacterProfile, engine: CharacterEngine):
    def node(state: SceneState) -> dict:
        output = engine.invoke(profile=profile, state=state)
        return apply_character_output(state, profile.name, output)

    return node


def make_symbolism_node(engine: SymbolismEngine):
    def node(state: SceneState) -> dict:
        scene_data = build_scene_data(state)
        plan = engine.plan(
            scene_data=scene_data,
            showrunner_plan=state["showrunner_plan"],
            state=state,
        )
        dumped = plan.model_dump()
        return {
            "scene_data": scene_data,
            "symbolism_plan": dumped,
            "subtext_guide": format_symbolism_plan(dumped),
        }

    return node


def make_writer_node(writer: SceneWriter):
    def node(state: SceneState) -> dict:
        return write_chapter(state=state, writer=writer)

    return node


def append_public_and_director(
    state: SceneState,
    public_record: dict,
    director_event: dict,
    broadcast: str,
) -> dict:
    updated_public_trace = state["public_trace"] + [public_record]
    updated_short_window = clamp_list(
        state["short_term_window"] + [public_record],
        state["short_window_size"],
    )
    return {
        "public_trace": updated_public_trace,
        "short_term_window": updated_short_window,
        "scene_log": state["scene_log"] + [director_event],
        "director_log": state["director_log"] + [broadcast],
    }


def apply_character_output(
    state: SceneState,
    speaker: str,
    output: AgentAction,
) -> dict:
    round_index = state["turn_count"] + 1
    public_record = PublicTurnRecord(
        speaker=speaker,
        round_index=round_index,
        micro_expression=output.micro_expression,
        action_and_dialogue=output.action_and_dialogue,
    ).model_dump()
    resource_snapshot = dict(state["resource_state"][speaker]["stats"])
    private_record = CharacterSceneEvent(
        speaker=speaker,
        round_index=round_index,
        observation_analysis=output.observation_analysis,
        emotional_shift=output.emotional_shift,
        hidden_agenda=output.hidden_agenda,
        micro_expression=output.micro_expression,
        action_and_dialogue=output.action_and_dialogue,
        resource_snapshot=resource_snapshot,
    ).model_dump()

    updated_private_memory = {
        name: list(memory) for name, memory in state["private_memory"].items()
    }
    updated_private_memory.setdefault(speaker, [])
    updated_private_memory[speaker] = clamp_list(
        updated_private_memory[speaker] + [private_record],
        3,
    )

    return {
        "public_trace": state["public_trace"] + [public_record],
        "short_term_window": clamp_list(
            state["short_term_window"] + [public_record],
            state["short_window_size"],
        ),
        "private_memory": updated_private_memory,
        "scene_log": state["scene_log"] + [private_record],
    }


def director_checkpoint(state: SceneState) -> dict:
    next_turn = state["turn_count"] + 1
    updated_resources = apply_resource_decay(state["resource_state"])
    updated_relationships = update_dynamic_relationships(
        state["dynamic_relationships"],
        state["scene_log"],
        next_turn,
    )
    beat = get_beat_for_round(state["showrunner_plan"], next_turn)
    broadcast = build_director_intervention(beat, updated_resources)
    public_record = PublicTurnRecord(
        speaker="Director",
        round_index=next_turn,
        micro_expression="走廊尽头的提示灯一闪一灭，保安靴底擦过地面的声音隔着玻璃传进来。",
        action_and_dialogue=broadcast,
    ).model_dump()
    director_event = DirectorSceneEvent(
        round_index=next_turn,
        beat_focus=beat.get("dramatic_function", "场景推进"),
        content=broadcast,
    ).model_dump()

    update = append_public_and_director(state, public_record, director_event, broadcast)
    update["turn_count"] = next_turn
    update["resource_state"] = updated_resources
    update["dynamic_relationships"] = updated_relationships
    return update


def get_beat_for_round(
    showrunner_plan: dict[str, object],
    round_index: int,
) -> dict[str, object]:
    for beat in showrunner_plan.get("forced_beats", []):
        if beat["round_index"] == round_index:
            return beat
    return {
        "round_index": round_index,
        "dramatic_function": "场景推进",
        "forced_event": "制度继续向前，不打算等任何人想明白。",
        "target_shift": "角色只能带着更少余地继续说话。",
    }


def build_director_intervention(
    beat: dict[str, object],
    resource_state: dict[str, dict[str, object]],
) -> str:
    pressure_lines = [
        build_resource_warning(name, pool) for name, pool in resource_state.items()
    ]
    return (
        f"[系统介入] {beat['forced_event']} {beat['target_shift']} "
        + " ".join(pressure_lines)
    )


def build_resource_warning(name: str, pool: dict[str, object]) -> str:
    stats = pool["stats"]
    if "debt" in stats and "san_value" in stats:
        return (
            f"{name} 的催缴界面显示：debt={stats['debt']}，"
            f"san_value={stats['san_value']}。"
        )
    if "quota_clock" in stats and "discipline_risk" in stats:
        return (
            f"{name} 的审核终端亮起红线：quota_clock={stats['quota_clock']}，"
            f"discipline_risk={stats['discipline_risk']}。"
        )
    return f"{name} 的生存压力继续上升。"


def apply_resource_decay(
    resource_state: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    updated: dict[str, dict[str, object]] = {}
    for name, pool in resource_state.items():
        new_pool = {
            "stats": dict(pool["stats"]),
            "decay_per_round": dict(pool["decay_per_round"]),
            "failure_condition": pool["failure_condition"],
            "pressure_note": pool["pressure_note"],
        }
        for stat_name, delta in new_pool["decay_per_round"].items():
            new_value = new_pool["stats"].get(stat_name, 0) + delta
            if stat_name in {"san_value", "dignity", "humanity_residue", "quota_clock"}:
                new_value = max(0, new_value)
            new_pool["stats"][stat_name] = new_value
        updated[name] = new_pool
    return updated


def update_dynamic_relationships(
    relationships: dict[str, dict[str, str]],
    scene_log: list[dict[str, object]],
    round_index: int,
) -> dict[str, dict[str, str]]:
    updated = {observer: dict(mapping) for observer, mapping in relationships.items()}
    recent_events = [
        entry
        for entry in scene_log
        if entry.get("event_type") == "character" and entry.get("round_index") == round_index
    ]
    for entry in recent_events:
        speaker = entry["speaker"]
        targets = [name for name in updated.keys() if name != speaker]
        if not targets:
            continue
        target = targets[0]
        updated[speaker][target] = infer_relationship_label(entry)
    return updated


def infer_relationship_label(entry: dict[str, object]) -> str:
    combined = " ".join(
        str(entry[key])
        for key in [
            "observation_analysis",
            "emotional_shift",
            "hidden_agenda",
            "micro_expression",
        ]
    )
    if any(token in combined for token in ["钱", "债", "账户", "贴片", "活下去"]):
        return "绝望驱动的依赖与试探正在加深。"
    if any(token in combined for token in ["流程", "配额", "风控", "纪律", "指标"]):
        return "程序化审视里的压制意味更重了。"
    if any(token in combined for token in ["裂纹", "停顿", "记住", "迟疑"]):
        return "看似无事，实际已经出现一丝被压下去的动摇。"
    if any(token in combined for token in ["麻木", "屈辱", "发热", "反胃"]):
        return "屈辱正在把关系推成更赤裸的功能交换。"
    return "对彼此的警惕没有减轻，只是更懂得怎样利用对方了。"


def clamp_list(items: list[dict], size: int) -> list[dict]:
    if size <= 0:
        return []
    return items[-size:]


def should_continue_scene(state: SceneState) -> Literal["continue", "subtext"]:
    if state["turn_count"] >= state["max_turns"]:
        return "subtext"
    return "continue"
