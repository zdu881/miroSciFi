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
    communication_mode: str = "shared_public",
    delay_profile: dict[str, dict[str, int]] | None = None,
) -> SceneState:
    resource_payload = resource_state_override or {
        name: resource.model_dump()
        for name, resource in resource_state_for_characters(characters).items()
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
        "scene_log": [],
        "director_log": [],
        "symbolism_plan": {},
        "continuity_summary": {},
        "chapter_history": [],
        "carryover_threads": [],
        "last_scene_summary": "",
        "current_location": "",
        "time_marker": "",
        "communication_mode": communication_mode,
        "local_inboxes": {character.name: [] for character in characters},
        "pending_transmissions": [],
        "transmission_log": [],
        "delay_profile": delay_profile or {},
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
    graph.add_node("showrunner_node", make_showrunner_node(characters, showrunner_engine))
    graph.add_node("director_setup", director_setup)
    graph.add_node("character_a_turn", make_character_node(character_a, character_engine))
    graph.add_node("character_b_turn", make_character_node(character_b, character_engine))
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
            elif "地球主权协调局" in observer.role and "总督" in target.role:
                mapping[observer.name][target.name] = "把对方视作必须在迟到之前重新套回法统的边缘行政接口。"
            elif "总督" in observer.role and "地球主权协调局" in target.role:
                mapping[observer.name][target.name] = "把对方视作永远晚到的中心回波，只在文书上强大。"
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
    time_marker = plan.get("opening_time_marker") or state.get("time_marker") or "未知时刻"
    location = plan.get("opening_location") or state.get("current_location") or "未知地点"
    continuity_mode = plan.get("continuity_mode", "retain")
    if state.get("last_scene_summary"):
        bridge = (
            "上一场留下的回波还在路上。"
            if state.get("communication_mode") == "delayed_inbox"
            else (
                "上一场留下的沉默还没散。"
                if continuity_mode == "retain"
                else "上一场留下的后果已经换了地方继续发作。"
            )
        )
    else:
        bridge = ""
    if state.get("communication_mode") == "delayed_inbox":
        opening_text = (
            f"{time_marker}，{location}。{bridge} "
            "终端提示：‘没有人拥有同一时刻的太阳系；每个人拿到的都只是别处已经过去的消息。’ "
            f"{build_location_detail(location)}"
        )
    else:
        opening_text = (
            f"{time_marker}，{location}。{bridge} "
            "系统提示：‘逾期、违约与异常标签不会因为换了地点就被撤销。’ "
            f"{build_location_detail(location)}"
        )
    return opening_text, build_location_micro_expression(location)


def build_location_detail(location: str) -> str:
    if "治理局" in location or "同步厅" in location:
        return "墙面上的太阳系时延图持续刷新，每一条闪烁轨迹都像一根已经来不及拉直的神经。"
    if "木卫三" in location or "卡利斯托" in location or "港" in location:
        return "低重力舱壁偶尔轻轻鸣响，像有旧命令在金属里迟到地回声。"
    return "空气里总有一点被旧数据烤过的冷味，像一条慢半拍的新闻带。"


def build_location_micro_expression(location: str) -> str:
    if "治理局" in location or "同步厅" in location:
        return "光时延迟图在玻璃墙上轮番变色，值班员的视线跟着那些晚到的亮点一格一格移动。"
    if "木卫三" in location or "卡利斯托" in location or "港" in location:
        return "港口穹顶外的工业灯隔着厚层冰壳折回来，照得每张脸都像比实际更晚一步。"
    return "远处的信号中继塔偶尔吐出一声短促蜂鸣，像谁的未来又被别人提早收到了。"


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
        transmission_target=output.transmission_target,
        transmission_content=output.transmission_content,
    ).model_dump()

    updated_private_memory = {
        name: list(memory) for name, memory in state["private_memory"].items()
    }
    updated_private_memory.setdefault(speaker, [])
    updated_private_memory[speaker] = clamp_list(updated_private_memory[speaker] + [private_record], 3)

    result: dict[str, object] = {
        "private_memory": updated_private_memory,
        "scene_log": state["scene_log"] + [private_record],
    }
    if state.get("communication_mode") != "delayed_inbox":
        result["public_trace"] = state["public_trace"] + [public_record]
        result["short_term_window"] = clamp_list(
            state["short_term_window"] + [public_record],
            state["short_window_size"],
        )

    if output.transmission_target and output.transmission_content:
        queued = build_queued_transmission(
            state=state,
            sender=speaker,
            target=output.transmission_target,
            content=output.transmission_content,
            round_index=round_index,
        )
        result["pending_transmissions"] = state.get("pending_transmissions", []) + [queued]
        result["transmission_log"] = state.get("transmission_log", []) + [queued]
    return result


def build_queued_transmission(
    *,
    state: SceneState,
    sender: str,
    target: str,
    content: str,
    round_index: int,
) -> dict[str, object]:
    delay = state.get("delay_profile", {}).get(sender, {}).get(target, 0)
    return {
        "sender": sender,
        "target": target,
        "content": content,
        "queued_round": round_index,
        "remaining_delay": delay,
        "original_delay": delay,
        "status": "queued",
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
        micro_expression="提示灯在延迟图上依次亮起，像一串彼此看不见彼此现在的坐标。",
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

    if state.get("pending_transmissions"):
        inboxes, pending, log_updates, delivery_events, delivery_logs = tick_pending_transmissions(
            state,
            delivered_round=next_turn,
        )
        update["local_inboxes"] = inboxes
        update["pending_transmissions"] = pending
        update["transmission_log"] = state.get("transmission_log", []) + log_updates
        if delivery_events:
            update["scene_log"] = update["scene_log"] + delivery_events
            update["director_log"] = update["director_log"] + delivery_logs
    return update


def tick_pending_transmissions(
    state: SceneState,
    *,
    delivered_round: int,
) -> tuple[
    dict[str, list[dict[str, object]]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
]:
    inboxes = {
        name: list(items)
        for name, items in state.get("local_inboxes", {}).items()
    }
    still_pending: list[dict[str, object]] = []
    log_updates: list[dict[str, object]] = []
    delivery_events: list[dict[str, object]] = []
    delivery_logs: list[str] = []
    for item in state.get("pending_transmissions", []):
        remaining = int(item.get("remaining_delay", 0)) - 1
        if remaining <= 0:
            delivered_item = {
                **item,
                "remaining_delay": 0,
                "status": "delivered",
                "delivered_round": delivered_round,
            }
            target = str(item.get("target", ""))
            inboxes.setdefault(target, [])
            inboxes[target] = clamp_list(inboxes[target] + [delivered_item], 5)
            log_updates.append(delivered_item)
            text = f"[延迟消息送达] {item.get('sender')} -> {target}：{item.get('content')}"
            delivery_events.append(
                DirectorSceneEvent(
                    round_index=delivered_round,
                    beat_focus="信息回波",
                    content=text,
                ).model_dump()
            )
            delivery_logs.append(text)
        else:
            still_pending.append({**item, "remaining_delay": remaining, "status": "queued"})
    return inboxes, still_pending, log_updates, delivery_events, delivery_logs


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
    pressure_lines = [build_resource_warning(name, pool) for name, pool in resource_state.items()]
    return f"[系统介入] {beat['forced_event']} {beat['target_shift']} " + " ".join(pressure_lines)


def build_resource_warning(name: str, pool: dict[str, object]) -> str:
    stats = pool["stats"]
    if "debt" in stats and "san_value" in stats:
        return f"{name} 的催缴界面显示：debt={stats['debt']}，san_value={stats['san_value']}。"
    if "quota_clock" in stats and "discipline_risk" in stats:
        return f"{name} 的审核终端亮起红线：quota_clock={stats['quota_clock']}，discipline_risk={stats['discipline_risk']}。"
    if "credibility" in stats and "response_window" in stats:
        return f"{name} 的主权面板显示：credibility={stats['credibility']}，response_window={stats['response_window']}。"
    if "militia_loyalty" in stats and "dock_control" in stats:
        return f"{name} 的港口监测屏显示：militia_loyalty={stats['militia_loyalty']}，dock_control={stats['dock_control']}。"
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
            if stat_name in {
                "san_value",
                "dignity",
                "humanity_residue",
                "quota_clock",
                "response_window",
                "secession_window",
            }:
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
        str(entry.get(key, ""))
        for key in [
            "observation_analysis",
            "emotional_shift",
            "hidden_agenda",
            "micro_expression",
            "transmission_content",
        ]
    )
    if any(token in combined for token in ["钱", "债", "账户", "贴片", "活下去"]):
        return "绝望驱动的依赖与试探正在加深。"
    if any(token in combined for token in ["流程", "配额", "风控", "纪律", "指标"]):
        return "程序化审视里的压制意味更重了。"
    if any(token in combined for token in ["延迟", "光时", "回波", "在途", "盲区"]):
        return "双方都在拿彼此收不到的现在下注，猜疑被时间放大了。"
    if any(token in combined for token in ["港口", "独立", "法统", "命令"]):
        return "法统和既成事实正在分离，关系越来越像一场隔空夺权。"
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
