from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .engine import CharacterEngine, ContinuityEngine, ShowrunnerEngine, SymbolismEngine
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
    WORLD_CONTEXT_PROMPT,
    format_symbolism_plan,
    memory_state_for_characters,
    resource_state_for_characters,
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
    resource_state_override: dict[str, dict] | None = None,
    cognition_mode: str = "standard",
    memory_state_override: dict[str, dict[str, object]] | None = None,
) -> SceneState:
    resource_payload = resource_state_override or {
        name: resource.model_dump()
        for name, resource in resource_state_for_characters(characters).items()
    }
    memory_payload = memory_state_override or {
        name: {
            **payload,
            "loaded_contexts": list(payload.get("loaded_contexts", [])),
            "resident_memories": list(payload.get("resident_memories", [])),
        }
        for name, payload in memory_state_for_characters(characters).items()
    }
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
        "resource_state": resource_payload,
        "cognition_mode": cognition_mode,
        "memory_state": memory_payload,
        "memory_archive": {character.name: [] for character in characters},
        "memory_eviction_log": [],
        "scene_log": [],
        "director_log": [],
        "symbolism_plan": {},
        "continuity_summary": {},
        "chapter_history": [],
        "carryover_threads": [],
        "last_scene_summary": "",
        "current_location": "",
        "time_marker": "",
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
    continuity_engine: ContinuityEngine,
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
    graph.add_node("continuity_node", make_continuity_node(continuity_engine))

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
    graph.add_edge("writer_node", "continuity_node")
    graph.add_edge("continuity_node", END)
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
            elif "领航员" in observer.role and "审核官" in target.role:
                mapping[observer.name][target.name] = "把对方视作有权决定自己该忘掉什么的冷接口。"
            elif "审核官" in observer.role and "领航员" in target.role:
                mapping[observer.name][target.name] = "把对方视作一块高价值但随时可能失稳的人脑载体。"
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
            state=state,
        )
        return {
            "showrunner_plan": plan.model_dump(),
            "current_location": plan.opening_location,
            "time_marker": plan.opening_time_marker,
        }

    return node


def build_opening_broadcast(state: SceneState) -> tuple[str, str]:
    plan = state.get("showrunner_plan", {})
    time_marker = plan.get("opening_time_marker") or state.get("time_marker") or "凌晨四点十二分"
    location = plan.get("opening_location") or state.get("current_location") or "雾港第七码头的情绪采样站"
    continuity_mode = plan.get("continuity_mode", "retain")
    if state.get("last_scene_summary"):
        bridge = (
            "上一场留下的沉默还没散，队列已经继续往前走。"
            if continuity_mode == "retain"
            else "上一场留下的后果已经换了地方继续发作。"
        )
    else:
        bridge = ""
    environment = build_location_detail(location)
    opening_text = (
        f"{time_marker}，{location}。{bridge}"
        "系统提示：‘逾期、违约与异常标签不会因为换了地点就被撤销。’ "
        f"{environment}"
    )
    micro_expression = build_location_micro_expression(location)
    return opening_text, micro_expression


def build_location_detail(location: str) -> str:
    if "装载" in location or "白舱" in location or "远航局" in location or "许可" in location:
        return "舱壁上的预算条一格一格亮灭，像有人在替大脑计数。"
    if "宿舍" in location or "回访" in location:
        return "空气里有冷却胶和旧金属混在一起的味道，像一段刚被擦掉的梦还没散干净。"
    if "公寓" in location or "疗养仓" in location:
        return "走廊尽头的氧雾机隔几秒喘一下，像一台不太愿意继续工作的肺。"
    if "回收站" in location or "黑市" in location:
        return "顶棚滴下来的冷凝水敲在铁桶边沿，像有人在暗处替系统计时。"
    return "审核台边上搁着一只杯沿带细裂的保温杯，没人提它。"


def build_location_micro_expression(location: str) -> str:
    if "装载" in location or "白舱" in location or "远航局" in location or "许可" in location:
        return "白色读写灯在观察窗里一跳一跳，像神经元被迫给别的东西腾地方。"
    if "宿舍" in location or "回访" in location:
        return "走廊尽头的门锁偶尔自检，红点掠过每张脸时都像在核对谁还记得自己的名字。"
    if "公寓" in location or "疗养仓" in location:
        return "狭窄走廊里的感应灯亮一阵灭一阵，门缝里漏出的氧雾在脚边拖成一层淡白。"
    if "回收站" in location or "黑市" in location:
        return "铁门后的排风扇转得忽快忽慢，潮气贴着墙皮往下流。"
    return "采样站天花板的旧喇叭发出轻微电流噪音，玻璃门上的雾气迟迟不散。"


def director_setup(state: SceneState) -> dict:
    if state["director_log"]:
        return {}

    opening_text, micro_expression = build_opening_broadcast(state)
    opening_record = PublicTurnRecord(
        speaker="Director",
        round_index=0,
        micro_expression=micro_expression,
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


def make_continuity_node(engine: ContinuityEngine):
    def node(state: SceneState) -> dict:
        scene_data = state["scene_data"] or build_scene_data(state)
        summary = engine.summarize(scene_data=scene_data, state=state)
        dumped = summary.model_dump()
        return {
            "continuity_summary": dumped,
            "chapter_history": state["chapter_history"] + [dumped],
            "carryover_threads": dumped["carryover_threads"],
            "last_scene_summary": dumped["chapter_summary"],
            "current_location": dumped["ending_location"],
            "time_marker": dumped["ending_time_marker"],
        }

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
    updated_resource_state = {
        name: {
            "stats": dict(payload["stats"]),
            "decay_per_round": dict(payload["decay_per_round"]),
            "failure_condition": payload["failure_condition"],
            "pressure_note": payload["pressure_note"],
        }
        for name, payload in state["resource_state"].items()
    }
    updated_memory_state, updated_memory_archive, updated_eviction_log, memory_note = apply_cognitive_eviction(
        state,
        speaker,
        output,
        round_index,
        updated_resource_state,
    )
    resource_snapshot = dict(updated_resource_state[speaker]["stats"])
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
    if output.context_load_label:
        private_record["context_load_label"] = output.context_load_label
        private_record["context_load_cost"] = output.context_load_cost
    if output.evicted_memory_label:
        private_record["evicted_memory_label"] = output.evicted_memory_label
        private_record["evicted_memory_summary"] = output.evicted_memory_summary
        private_record["evicted_memory_cost"] = output.evicted_memory_cost
    if memory_note:
        private_record["memory_note"] = memory_note

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
        "resource_state": updated_resource_state,
        "memory_state": updated_memory_state,
        "memory_archive": updated_memory_archive,
        "memory_eviction_log": updated_eviction_log,
        "scene_log": state["scene_log"] + [private_record],
    }


def apply_cognitive_eviction(
    state: SceneState,
    speaker: str,
    output: AgentAction,
    round_index: int,
    resource_state: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], list[dict[str, object]], str]:
    memory_state = {
        name: {
            **payload,
            "loaded_contexts": list(payload.get("loaded_contexts", [])),
            "resident_memories": list(payload.get("resident_memories", [])),
        }
        for name, payload in state.get("memory_state", {}).items()
    }
    memory_archive = {
        name: list(items) for name, items in state.get("memory_archive", {}).items()
    }
    eviction_log = list(state.get("memory_eviction_log", []))
    if state.get("cognition_mode") != "eviction_budget":
        return memory_state, memory_archive, eviction_log, ""
    speaker_memory = memory_state.get(speaker)
    if not speaker_memory or int(speaker_memory.get("capacity", 0)) <= 0:
        return memory_state, memory_archive, eviction_log, ""

    note_parts: list[str] = []
    if output.context_load_label:
        load_cost = max(0, int(output.context_load_cost or 0))
        speaker_memory["used"] = int(speaker_memory.get("used", 0)) + load_cost
        speaker_memory["loaded_contexts"] = clamp_list(
            speaker_memory.get("loaded_contexts", [])
            + [
                {
                    "label": output.context_load_label,
                    "weight": load_cost,
                    "source": "scene_turn",
                }
            ],
            8,
        )
        note_parts.append(f"装载了 {output.context_load_label}（{load_cost}）")

    if output.evicted_memory_label:
        freed = evict_named_memory(
            speaker_memory,
            memory_archive,
            eviction_log,
            speaker,
            round_index,
            output.evicted_memory_label,
            output.evicted_memory_summary,
            int(output.evicted_memory_cost or 0),
            auto_generated=False,
        )
        if freed:
            speaker_memory["used"] = max(0, int(speaker_memory.get("used", 0)) - freed)
            note_parts.append(f"清退了 {output.evicted_memory_label}（回收 {freed}）")

    capacity = int(speaker_memory.get("capacity", 0))
    reserve_floor = int(speaker_memory.get("reserve_floor", 0))
    allowed_used = max(0, capacity - reserve_floor)
    while int(speaker_memory.get("used", 0)) > allowed_used and speaker_memory.get("resident_memories"):
        fallback = choose_fallback_memory(speaker_memory)
        if not fallback:
            break
        freed = evict_named_memory(
            speaker_memory,
            memory_archive,
            eviction_log,
            speaker,
            round_index,
            str(fallback.get("label", "未知记忆")),
            str(fallback.get("summary", "一段被系统自动清退的私人内容。")),
            int(fallback.get("weight", 0)),
            auto_generated=True,
        )
        speaker_memory["used"] = max(0, int(speaker_memory.get("used", 0)) - freed)
        note_parts.append(f"系统自动清退了 {fallback.get('label', '未知记忆')}（回收 {freed}）")

    sync_memory_stats(resource_state, speaker, speaker_memory, note_parts)
    return memory_state, memory_archive, eviction_log, "；".join(note_parts)


def choose_fallback_memory(memory_state: dict[str, object]) -> dict[str, object] | None:
    resident = list(memory_state.get("resident_memories", []))
    if not resident:
        return None
    return max(resident, key=lambda item: int(item.get("weight", 0)))


def evict_named_memory(
    speaker_memory: dict[str, object],
    memory_archive: dict[str, list[dict[str, object]]],
    eviction_log: list[dict[str, object]],
    speaker: str,
    round_index: int,
    label: str,
    summary: str,
    cost: int,
    *,
    auto_generated: bool,
) -> int:
    resident = list(speaker_memory.get("resident_memories", []))
    matched = None
    for index, item in enumerate(resident):
        if item.get("label") == label:
            matched = resident.pop(index)
            break
    if matched is None:
        matched = {
            "label": label,
            "weight": max(0, cost or 120),
            "summary": summary or "一段没有被保住的私人记忆。",
        }
    speaker_memory["resident_memories"] = resident
    payload = {
        "speaker": speaker,
        "round_index": round_index,
        "evicted_memory_label": label or matched.get("label", "未知记忆"),
        "evicted_memory_summary": summary or matched.get("summary", "一段没有被保住的私人记忆。"),
        "evicted_memory_cost": max(0, cost or int(matched.get("weight", 0))),
        "auto_generated": auto_generated,
    }
    memory_archive.setdefault(speaker, [])
    memory_archive[speaker] = memory_archive[speaker] + [payload]
    eviction_log.append(payload)
    return int(payload["evicted_memory_cost"])


def sync_memory_stats(
    resource_state: dict[str, dict[str, object]],
    speaker: str,
    speaker_memory: dict[str, object],
    note_parts: list[str],
) -> None:
    stats = resource_state.get(speaker, {}).get("stats", {})
    free_margin = int(speaker_memory.get("capacity", 0)) - int(speaker_memory.get("used", 0))
    if "memory_margin" in stats:
        stats["memory_margin"] = free_margin
    if note_parts and "self_coherence" in stats:
        stats["self_coherence"] = max(0, int(stats.get("self_coherence", 0)) - 4)
    if note_parts and "empathy_residue" in stats:
        stats["empathy_residue"] = max(0, int(stats.get("empathy_residue", 0)) - 2)




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
    if "memory_margin" in stats and "self_coherence" in stats:
        return (
            f"{name} 的认知面板显示：memory_margin={stats['memory_margin']}，"
            f"self_coherence={stats['self_coherence']}。"
        )
    if "audit_quota" in stats and "liability_risk" in stats:
        return (
            f"{name} 的清退终端显示：audit_quota={stats['audit_quota']}，"
            f"liability_risk={stats['liability_risk']}。"
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
            if stat_name in {"san_value", "dignity", "humanity_residue", "quota_clock", "memory_margin", "self_coherence", "audit_quota", "empathy_residue"}:
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
    if any(token in combined for token in ["记忆", "清退", "装载", "上下文", "失认", "航线"]):
        return "双方都在拿人格连续性给制度让位，关系越来越像一次冷的外科手术。"
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
