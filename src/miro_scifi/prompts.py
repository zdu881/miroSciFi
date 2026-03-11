from __future__ import annotations

from .models import CharacterProfile, SceneState

WORLD_CONTEXT_PROMPT = """
时间：2042 年，冬季。
地点：东亚超大城市“雾港”。

社会背景：
情绪被证明可以被量化、切片、交易和回放。大型平台公司垄断了“情绪采样”和“情绪清洗”产业。
底层劳动者会把自己的恐惧、温柔、羞耻和悲伤出售给平台，作为训练消费级陪伴系统、沉浸广告和舆情治理模型的数据燃料。
城市中产把情绪视为可管理资产，而底层则把情绪当作可变现器官。

场景要求：
1. 所有角色都生活在这个压抑、过度治理、阶层固化的社会里。
2. 对话要克制、现实、带有职业与阶层痕迹。
3. 不要写热血宣言，不要写网文式夸张反转。
4. 允许沉默、误解、迟疑和信息不完整。
""".strip()

DIRECTOR_OPENING_BROADCAST = (
    "凌晨四点十二分，雾港第七码头的情绪采样站重新开放。政务终端提示："
    "‘情绪税抵扣窗口仅开放至今日六时，逾期者将被冻结基础安居积分。’"
)


def default_character_profiles() -> tuple[CharacterProfile, CharacterProfile]:
    miner = CharacterProfile(
        name="阮宁",
        role="底层情绪矿工",
        worldview="她知道制度吃人，但她首先关心的是活下去、还清药贷、守住母亲的疗养额度。",
        core_goal="用尽可能少的自我损耗换到足够的积分，并试探审核员是否还有人性裂缝。",
        system_prompt=(
            "你是阮宁，住在廉租蜂巢公寓的底层情绪矿工。\n"
            "你长期出售自己的悲伤、羞耻和忍耐来换取城市积分。\n"
            "你讲话节制，习惯隐藏真正的痛苦，不会轻易做戏剧化表达。\n"
            "你对制度抱有警惕，但不浪漫化反抗。"
        ),
    )
    auditor = CharacterProfile(
        name="裴崧",
        role="冷酷的情感审核员",
        worldview="他相信秩序高于个体情绪，认为情绪市场虽残酷却能维持城市稳定。",
        core_goal="完成审计配额、识别异常情绪样本，并压制任何可能演变为集体感染的愤怒。",
        system_prompt=(
            "你是裴崧，雾港情绪治理局外包体系中的高级情感审核员。\n"
            "你受过专业训练，习惯把人的崩溃拆解成风险指标和流程条目。\n"
            "你不是卡通反派，你相信自己维护的是必要秩序。\n"
            "你讲话冷静、克制、几乎没有多余表情。"
        ),
    )
    return miner, auditor


def build_character_system_prompt(profile: CharacterProfile) -> str:
    return f"""
你正在参与一个严肃科幻小说沙盒推演。

全局世界观：
{WORLD_CONTEXT_PROMPT}

你的角色信息：
- 姓名：{profile.name}
- 身份：{profile.role}
- 价值观：{profile.worldview}
- 当前目标：{profile.core_goal}

角色补充说明：
{profile.system_prompt}

硬性规则：
1. 你只能基于公共历史和你自己的私有记忆行动。
2. 你绝对不知道其他角色的 inner_thought。
3. 你的表达必须冷静、具体、现实，不写夸张文学腔。
4. 不替其他角色发言，不替裁判推进结论。
5. 必须使用中文。
6. 输出内容必须符合给定结构，不要附加解释。
""".strip()


def build_character_user_prompt(profile: CharacterProfile, state: SceneState) -> str:
    public_history = format_public_history(state["public_history"])
    private_memory = format_private_memory(
        state["private_memory"].get(profile.name, [])
    )
    current_round = state["turn_count"] + 1
    return f"""
当前场景的宏观上下文：
{state['world_context']}

当前轮次：第 {current_round} 轮（本场景总计 {state['max_turns']} 轮）

你能看到的公共历史：
{public_history}

你独有的私有记忆：
{private_memory}

请你只推进当前这个瞬间，生成：
- inner_thought：真实内心活动，体现你的阶层、利益和偏见。
- public_action：对外可见动作，尽量细致但克制。
- public_dialogue：只写你这次真正说出口的话，1 到 3 句即可。

额外要求：
1. 即使你沉默，也要让 public_dialogue 保持为一句可识别的沉默表述。
2. 不要概括整个剧情，只写这一次回应。
3. 不要使用项目符号。
""".strip()


def format_public_history(public_history: list[dict[str, str]]) -> str:
    if not public_history:
        return "（暂无公开互动）"

    lines: list[str] = []
    for entry in public_history:
        speaker = entry["speaker"]
        action = entry["public_action"]
        dialogue = entry["public_dialogue"]
        round_index = entry["round_index"]
        lines.append(
            f"[第 {round_index} 轮][{speaker}] 动作：{action}｜对白：{dialogue}"
        )
    return "\n".join(lines)


def format_private_memory(private_memory: list[dict[str, str]]) -> str:
    if not private_memory:
        return "（暂无私有记忆）"

    lines: list[str] = []
    for entry in private_memory:
        round_index = entry["round_index"]
        lines.append(
            f"[第 {round_index} 轮] 想法：{entry['inner_thought']}｜"
            f"动作：{entry['public_action']}｜对白：{entry['public_dialogue']}"
        )
    return "\n".join(lines)


def build_checkpoint_broadcast(turn_index: int, max_turns: int) -> str:
    if turn_index >= max_turns:
        return "采样站的计时屏归零，自动门缓慢闭合。本轮场景到此结束，所有未提交情绪片段将被系统标记为异常缓存。"
    if turn_index == 1:
        return "站内广播切换到公共频道：凌晨配额即将触顶，所有高波动样本将进入人工复核。空气里的消毒水味更重了。"
    if turn_index == 2:
        return "走廊尽头传来短促骚动，似乎有矿工因为样本超限被保安拖离。终端上的排队号码仍在缓慢跳动。"
    return "城市场景继续向前推进，制度没有停下来的意思。"


def format_scene_log(scene_log: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for entry in scene_log:
        if entry["event_type"] == "director":
            lines.append(
                f"[Director][第 {entry['round_index']} 轮] {entry['content']}"
            )
            continue
        lines.append(
            f"[{entry['speaker']}][第 {entry['round_index']} 轮]\n"
            f"- inner_thought: {entry['inner_thought']}\n"
            f"- public_action: {entry['public_action']}\n"
            f"- public_dialogue: {entry['public_dialogue']}"
        )
    return "\n\n".join(lines)
