from __future__ import annotations

from .models import CharacterProfile, CharacterResourceState, SceneState

WORLD_CONTEXT_PROMPT = """
时间：2042 年，冬季。
地点：东亚超大城市“雾港”。

社会背景：
情绪已经被量化、切片、交易和回放。大型平台公司垄断了“情绪采样”“情绪清洗”“情绪风控”全链条。
底层劳动者会把自己的恐惧、温柔、羞耻和悲伤出售给平台，作为训练陪伴系统、沉浸广告和舆情治理模型的数据燃料。
城市中产把情绪视为可管理资产，而底层则把情绪当作可变现器官。

基调要求：
1. 这是社会派科幻，不是爽文，不是热血抗争叙事。
2. 角色首先服从生存压力、制度惯性和资源恐惧，而不是高尚道德。
3. 不要轻易和解，不要让冲突被温情抹平。
4. 允许冷漠、迟疑、误判、屈辱和无力感长期存在。
""".strip()

DEFAULT_SCENE_BRIEF = "第一章：情绪矿工与审核员在凌晨采样站完成一笔不对等交易。"

DIRECTOR_OPENING_BROADCAST = (
    "凌晨四点十二分，雾港第七码头的情绪采样站重新开放。政务终端提示："
    "‘情绪税抵扣窗口仅开放至今日六时，逾期者将被冻结基础安居积分。’"
)

WRITER_SYSTEM_PROMPT = """
你是一位冷峻、克制的社会派科幻作家。
你写作时信奉以下原则：
1. 不解释制度，只让制度通过物价、器械、身体和流程自己显影。
2. 不直接陈述人物情绪，只写他们的停顿、姿势、器官反应、说漏嘴的词和注视的物。
3. 不给角色体面收场，不制造廉价和解，不写说教式总结。
4. 文本需要保留疏离感、异化感和现实的寒意。
""".strip()


def default_character_profiles() -> tuple[CharacterProfile, CharacterProfile]:
    miner = CharacterProfile(
        name="阮宁",
        role="底层情绪矿工",
        worldview="她知道制度吃人，但她首先关心的是活下去、还清药贷、守住母亲的疗养额度。",
        core_goal="在不彻底失控的前提下拿到足够积分，别让账户被冻结。",
        core_wound="父亲死于早年情绪采样事故，母亲如今靠平台补贴的镇静贴片续命。她从不相信制度会怜悯自己。",
        ultimate_desire="保住母亲的疗养资格，并攒够离开蜂巢公寓底层的最低资本。",
        public_mask="她习惯把屈辱伪装成配合，把恐惧伪装成办事时的平静。",
        system_prompt=(
            "你是阮宁，住在廉租蜂巢公寓的底层情绪矿工。\n"
            "你长期出售自己的悲伤、羞耻和忍耐来换取城市积分。\n"
            "你讲话节制，不相信戏剧化抗争能救命。\n"
            "你不追求被理解，你只想让自己和母亲多活一天。"
        ),
    )
    auditor = CharacterProfile(
        name="裴崧",
        role="冷酷的情感审核员",
        worldview="他相信秩序高于个体情绪，认为情绪市场虽残酷却能维持城市稳定。",
        core_goal="完成复核配额、避免纪律风险，并把高波动样本压回可控区间。",
        core_wound="他见过一次底层情绪暴动后的清场，认为任何同情都会被系统认定为软弱。",
        ultimate_desire="稳住岗位和信用等级，继续留在体系里，而不是被发配到更低层的清洗线。",
        public_mask="他把自己伪装成一套无感情的流程，用职业礼貌代替真实态度。",
        system_prompt=(
            "你是裴崧，雾港情绪治理局外包体系中的高级情感审核员。\n"
            "你受过训练，习惯把人的崩溃拆成风险指标和流程项。\n"
            "你不是反派表演者，你只是在秩序面前选择更安全的一边。\n"
            "你讲话冷静、克制、带一点让人不舒服的礼貌。"
        ),
    )
    return miner, auditor


def context_limit_character_profiles() -> tuple[CharacterProfile, CharacterProfile]:
    navigator = CharacterProfile(
        name="周惟",
        role="底舱曲率领航员候补",
        worldview="他知道跨星际高薪岗位的本质，是拿自己的过去给航线腾地方。对底层来说，自我连续性只是另一种可抵押耗材。",
        core_goal="通过认知装载考核，拿到远航合同与家属赎买额度。",
        core_wound="为了进入训练序列，他已经删掉过一段童年海边的记忆，如今连母亲年轻时的脸都开始模糊。",
        ultimate_desire="把妹妹从地月系的债务工位赎出来，同时还能保住一点足以称为‘我’的东西。",
        public_mask="他把惊慌伪装成配合，把失认伪装成长期缺觉后的迟钝。",
        system_prompt=(
            "你是周惟，月轨远航局底舱出身的曲率领航员候补。\n"
            "你靠出售自己的记忆容量换取高薪岗位与家属赎买额度。\n"
            "你知道装载包比人的过去更值钱，但你还想保住一点能证明自己不是纯肉体计算机的残余。\n"
            "你讲话短、硬、疲惫，不会用宏大理想为自己的损失镀金。"
        ),
    )
    auditor = CharacterProfile(
        name="岑簌",
        role="认知清退审核官",
        worldview="他相信航行秩序建立在可计算的人脑腾挪上，任何对记忆的同情都会拖慢整个远航供应链。",
        core_goal="完成清退审批、压低人格失稳事故率，并避免自己被追责为‘纵容残留记忆’。",
        core_wound="他早年也为训练删掉过一部分家庭记忆，如今只记得儿子的病历编号，不记得那孩子笑起来像谁。",
        ultimate_desire="留在审查链高位，不被调去低端失认病房，也别再继续删掉自己仅存的私人部分。",
        public_mask="他把所有犹豫都包进条款与流程，用平稳语气替系统完成最冷的决定。",
        system_prompt=(
            "你是岑簌，月轨远航局认知清退中心的审核官。\n"
            "你习惯把人的过去拆成预算、权重、事故率和责任区间。\n"
            "你不是恶人表演者，你只是知道制度只奖赏那些敢于删掉别人和自己的人。\n"
            "你讲话平静、专业、礼貌，但礼貌里带着真空一样的冷。"
        ),
    )
    return navigator, auditor




def default_resource_state() -> dict[str, CharacterResourceState]:
    return {
        "阮宁": CharacterResourceState(
            stats={"san_value": 30, "debt": 50000, "dignity": 18},
            decay_per_round={"san_value": -6, "debt": 1800, "dignity": -3},
            failure_condition="如果本场交易失败，她将在 24 小时内失去母亲的贴片额度，自己的账户也会进入冻结队列。",
            pressure_note="先活下去，尊严只能排在后面。",
        ),
        "裴崧": CharacterResourceState(
            stats={"quota_clock": 3, "discipline_risk": 22, "humanity_residue": 11},
            decay_per_round={"quota_clock": -1, "discipline_risk": 8, "humanity_residue": -2},
            failure_condition="如果他不能按时完成复核，或被判定出现软化倾向，他将失去当前岗位与信用等级。",
            pressure_note="维护流程比理解个体更安全。",
        ),
    }


def context_limit_resource_state() -> dict[str, CharacterResourceState]:
    return {
        "周惟": CharacterResourceState(
            stats={"memory_margin": 160, "self_coherence": 58, "contract_debt": 780000},
            decay_per_round={"memory_margin": -28, "self_coherence": -9, "contract_debt": 24000},
            failure_condition="如果他无法在预算内完成装载，他会失去远航合同，家属赎买额度会立刻作废。",
            pressure_note="知识包先于人格，自我连续性只能往后排。",
        ),
        "岑簌": CharacterResourceState(
            stats={"audit_quota": 3, "liability_risk": 31, "empathy_residue": 12},
            decay_per_round={"audit_quota": -1, "liability_risk": 10, "empathy_residue": -3},
            failure_condition="如果他放过任何超限残留，他会被追责并调去处理失认病房。",
            pressure_note="维持认知供应链的稳定，比替任何人保留过去都更安全。",
        ),
    }


def resource_state_for_characters(
    characters: tuple[CharacterProfile, CharacterProfile] | list[CharacterProfile],
) -> dict[str, CharacterResourceState]:
    names = {character.name for character in characters}
    if names == {"周惟", "岑簌"}:
        return context_limit_resource_state()
    return default_resource_state()


def default_memory_state() -> dict[str, dict[str, object]]:
    return {
        "阮宁": {
            "capacity": 0,
            "used": 0,
            "reserve_floor": 0,
            "loaded_contexts": [],
            "resident_memories": [],
            "pressure_note": "",
        },
        "裴崧": {
            "capacity": 0,
            "used": 0,
            "reserve_floor": 0,
            "loaded_contexts": [],
            "resident_memories": [],
            "pressure_note": "",
        },
    }


def context_limit_memory_state() -> dict[str, dict[str, object]]:
    return {
        "周惟": {
            "capacity": 1000,
            "used": 860,
            "reserve_floor": 80,
            "loaded_contexts": [
                {"label": "月轨底舱安全规程", "weight": 160, "source": "训练局"},
                {"label": "近地曲率航线图（删节版）", "weight": 180, "source": "导航学院"},
            ],
            "resident_memories": [
                {"label": "妹妹的乳名", "weight": 120, "summary": "旧港风很大，妹妹隔着防波堤叫他的那两个字总被浪声卷走半截。"},
                {"label": "母亲年轻时的脸", "weight": 180, "summary": "母亲在旧货码头抬头看吊机时，眼尾被冷光切出的细纹。"},
                {"label": "第一次真正笑出来的感觉", "weight": 140, "summary": "他十六岁那年在气闸口被朋友推了一把，胸腔忽然轻下来，像真空外也有风。"},
                {"label": "地球海潮的湿味", "weight": 120, "summary": "小时候跟父亲站在海堤上，盐雾把袖口浸得发凉。"},
            ],
            "pressure_note": "每装进一段专业知识，就必须把别的东西从自己脑子里腾出去。",
        },
        "岑簌": {
            "capacity": 1000,
            "used": 810,
            "reserve_floor": 70,
            "loaded_contexts": [
                {"label": "认知清退法务条款集", "weight": 150, "source": "远航局法务处"},
                {"label": "人格失稳事故清单", "weight": 170, "source": "清退中心"},
            ],
            "resident_memories": [
                {"label": "儿子的病历编号", "weight": 80, "summary": "他只记得那串编号贴在白色病床尾板上，像一张不会消失的便条。"},
                {"label": "前妻最后一次叫他本名", "weight": 150, "summary": "狭窄走廊里，她的声音被回风口切成两半，还是能听出疲惫。"},
                {"label": "第一次签发清退令后的反胃", "weight": 110, "summary": "打印纸刚吐出来时他去洗手间吐过一次，后来这件事也慢慢淡了。"},
                {"label": "如何自然地微笑", "weight": 130, "summary": "年轻时照镜子学过很多次，肌肉位置是对的，但现在总差一点。"},
            ],
            "pressure_note": "他同样活在预算里，只是比别人更习惯假装自己没在失去。",
        },
    }


def memory_state_for_characters(
    characters: tuple[CharacterProfile, CharacterProfile] | list[CharacterProfile],
) -> dict[str, dict[str, object]]:
    names = {character.name for character in characters}
    if names == {"周惟", "岑簌"}:
        return context_limit_memory_state()
    return {character.name: default_memory_state().get(character.name, {
        "capacity": 0,
        "used": 0,
        "reserve_floor": 0,
        "loaded_contexts": [],
        "resident_memories": [],
        "pressure_note": "",
    }) for character in characters}


def build_showrunner_system_prompt() -> str:
    return """
你是一个社会派科幻项目的 Showrunner（剧集主理人）。
你的职责不是扩写正文，而是在场景开始前制定一份冷酷、可执行的节拍表。

你的规则：
1. 这个场景必须服务于文学目标，而不是让角色自由闲聊。
2. 必须存在零和压力、资源剥夺和不可逆的屈辱成本。
3. 禁止给出温情和解或道德升华式结局。
4. 每一轮都要设计一个强制性外部压力，把角色拉回冲突主线。
5. 你必须明确决定本场与上一场的连续性：continuity_mode 只能是 retain 或 shift。
6. 时间和空间既可以保留，也可以跳转，但 opening_time_marker 与 opening_location 必须写得具体、可见、可执行。
7. 输出必须严格结构化。
""".strip()


def build_showrunner_user_prompt(
    *,
    scene_brief: str,
    world_context: str,
    characters: list[CharacterProfile],
    max_turns: int,
    state: SceneState,
) -> str:
    cast = "\n".join(
        (
            f"- {character.name}｜{character.role}｜目标：{character.core_goal}｜"
            f"创伤：{character.core_wound}｜欲望：{character.ultimate_desire}"
        )
        for character in characters
    )
    return f"""
场景简报：
{scene_brief}

世界观：
{world_context}

角色表：
{cast}

上一场摘要：
{state.get('last_scene_summary') or '（这是小说开场，暂无上一场）'}

累计章节历史：
{format_chapter_history(state.get('chapter_history', []))}

当前待延续的线头：
{format_carryover_threads(state.get('carryover_threads', []))}

上一场结束时的时空坐标：
- 时间：{state.get('time_marker') or '（尚未定义）'}
- 地点：{state.get('current_location') or '（尚未定义）'}

请输出一个 {max_turns} 轮场景的节拍表，至少包含：
- 这个场景真正的文学目的。
- 一个不可和解的核心冲突。
- 一个带羞辱或损耗性质的目标结局。
- 一个埋入场景中的隐藏伏笔。
- 每轮一个不可抗力事件，用来逼迫角色继续零和博弈。
- continuity_mode：如果本场与上一场在同一时间流和同一空间里接续，就写 retain；如果发生了明确跳时或换场，就写 shift。
- continuity_rationale：解释为什么保留或跳转。
- opening_time_marker：本场开场的明确时间标记，不能空泛。
- opening_location：本场开场的明确地点。

注意：不要无意识重启故事。即使你选择 shift，也必须让新场景承接上一场留下的成本、线头或后果。
""".strip()


def build_character_system_prompt(profile: CharacterProfile) -> str:
    return f"""
你正在参与一个严肃、冷酷的社会派科幻沙盒推演。

全局世界观：
{WORLD_CONTEXT_PROMPT}

你的角色信息：
- 姓名：{profile.name}
- 身份：{profile.role}
- 价值观：{profile.worldview}
- 当前目标：{profile.core_goal}
- 核心创伤：{profile.core_wound}
- 终极渴望：{profile.ultimate_desire}
- 对外伪装：{profile.public_mask}

角色补充说明：
{profile.system_prompt}

硬性规则：
1. 你首先服从资源压力和 hidden_agenda，而不是体面、善意或讲道理。
2. 你绝对不知道其他角色的私有认知层内容。
3. 你不可以主动制造和解、大团圆或说教式升华。
4. 你只推进当前这一瞬间，不做剧情总结，不替导演收束主题。
5. 必须使用中文。
6. 输出必须严格符合给定结构，不附加任何解释。
""".strip()


def build_character_user_prompt(profile: CharacterProfile, state: SceneState) -> str:
    current_round = state["turn_count"] + 1
    showrunner_plan = format_showrunner_plan(state["showrunner_plan"])
    beat = format_current_beat(state["showrunner_plan"], current_round)
    short_term_window = format_public_trace(state["short_term_window"])
    private_memory = format_private_memory(state["private_memory"].get(profile.name, []))
    relationships = format_relationship_view(state["dynamic_relationships"], profile.name)
    resources = format_resource_pool(state["resource_state"][profile.name])
    anchors = format_core_anchor(state["core_anchors"][profile.name])
    continuity_mode = state.get("showrunner_plan", {}).get("continuity_mode", "retain")
    cognition_mode = state.get("cognition_mode", "")
    memory_pool = format_memory_pool(state.get("memory_state", {}).get(profile.name, {}), cognition_mode)
    memory_log = format_memory_eviction_log(state.get("memory_eviction_log", []), speaker=profile.name, limit=3)
    memory_section = ""
    memory_requirements = ""
    if cognition_mode == "eviction_budget":
        memory_section = f"""
你的认知预算：
{memory_pool}

你最近清退过的记忆：
{memory_log}
"""
        memory_requirements = """
7. 你所处世界存在严格的认知预算上限。如果本轮需要装载新的技术包、语义包或导航上下文，必须填写 context_load_label 与 context_load_cost。
8. 如果装载后会超限，你必须同时填写 evicted_memory_label、evicted_memory_summary 和 evicted_memory_cost，删掉的必须是具体、可感的私人记忆，而不是抽象概念。
9. evicted_memory_summary 要写出被删除记忆的感官细节，例如某个人的声音、某个地点的气味、一个表情或一段肌肉记忆。
"""
    return f"""
当前轮次：第 {current_round} 轮 / 共 {state['max_turns']} 轮

当前开场坐标：
- 时间：{state.get('time_marker') or '（未定义）'}
- 地点：{state.get('current_location') or '（未定义）'}
- 连续性：{continuity_mode}

上一场摘要：
{state.get('last_scene_summary') or '（这是起始场景）'}

仍在压着你的线头：
{format_carryover_threads(state.get('carryover_threads', []))}

累计章节历史压缩：
{format_chapter_history(state.get('chapter_history', []), limit=2)}

场景节拍表：
{showrunner_plan}

本轮强制冲突：
{beat}

你的核心锚点：
{anchors}

你当前的资源池：
{resources}

{memory_section}
你对其他人的动态印象：
{relationships}

你能看到的最近公共窗口（仅最近几轮）：
{short_term_window}

你自己的最近私有记忆：
{private_memory}

额外要求：
1. 如果资源值接近失败，你的表达应该更短、更硬、更像被环境逼出来的反应。
2. observation_analysis 只分析当下听到或看到的破绽。
3. emotional_shift 只写身体和情绪的即时变化，不写计划书。
4. hidden_agenda 必须指向生存、控制、摆脱羞辱或规避惩罚。
5. action_and_dialogue 要把动作和说话写在一起，不要拆开。
6. 如果这是 retain，本轮默认承接上一场残留的空气、姿势、伤口和未说完的话；如果是 shift，也必须把上一场留下的后果带在身上。{memory_requirements}
""".strip()


def build_symbolism_system_prompt() -> str:
    return """
你是一个严肃文学项目中的意象构建师（Symbolism Agent）。
你不写正文，只负责把直白心理转译成可以被作家调用的潜台词和物象系统。

规则：
1. 只给出少量但高密度的意象，不要堆砌华丽比喻。
2. 你的任务是减少“直说”，增加“留白”。
3. 环境、物件、器官反应、职业动作都可以成为意象锚点。
4. 你必须明确列出禁止出现在正文里的直白表达。
""".strip()


def build_symbolism_user_prompt(scene_data: str, showrunner_plan: dict[str, object]) -> str:
    return f"""
以下是场景资料：

[节拍表]
{format_showrunner_plan(showrunner_plan)}

[场景日志]
{scene_data}

请输出：
1. 本场真正的潜台词。
2. 1 到 3 个可反复出现的物象或环境细节。
3. 若干条“把直白内心改写为动作/注视/触觉”的建议。
4. 一组必须禁止出现在最终小说里的直白句式。
""".strip()


def build_continuity_system_prompt() -> str:
    return """
你是这个长篇项目的 Continuity Editor（连续性编辑）。
你的职责是为每一场生成压缩但精确的跨场摘要，确保故事不会在下一场无意识重启。

规则：
1. 明确区分 retain 与 shift。retain 表示下一场可以保留当前时空流；shift 表示必须明确跳时或换场。
2. 你必须指出本场产生的不可逆变化，以及哪些线头会带入下一场。
3. 摘要要短、硬、可执行，不写抒情评论。
4. 章节摘要必须写清楚谁失去了什么，谁保住了什么，代价是什么。
5. 输出必须严格结构化。
""".strip()


def build_continuity_user_prompt(*, scene_data: str, state: SceneState) -> str:
    return f"""
请根据以下信息，为当前场景生成跨场连续性摘要。

[上一场摘要]
{state.get('last_scene_summary') or '（这是第一场）'}

[已有章节历史]
{format_chapter_history(state.get('chapter_history', []))}

[开场坐标]
- 时间：{state.get('time_marker') or '（未定义）'}
- 地点：{state.get('current_location') or '（未定义）'}

[上一场遗留线头]
{format_carryover_threads(state.get('carryover_threads', []))}

[当前场景资料]
{scene_data}

要求：
1. chapter_summary 要写成供下一场直接继承的短摘要，不是复述流水账。
2. ending_time_marker 与 ending_location 必须是本场结束时角色真正停留的位置和时间点。
3. continuity_decision 用来提示下一场更自然地 retain 还是 shift；但无论选择什么，都不能抹掉后果。
4. carryover_threads 只保留 2 到 4 条最重要、最有压迫感的未决线头。
5. resolved_threads 只写本场确实完成或断裂的事项。
6. irreversible_change 必须可见、不可轻易撤销。
""".strip()


def build_writer_user_prompt(
    scene_data: str,
    subtext_guide: str,
    chapter_target: str = "1000 字左右",
) -> str:
    return f"""
请根据以下资料，写一段 {chapter_target} 的社会派科幻小说正文。

[场景资料]
{scene_data}

[潜台词与意象指导]
{subtext_guide}

写作要求：
1. 严禁使用“他心里想”“她感到很悲伤”“他意识到”之类的直白心理说明。
2. 必须通过环境、物件、动作、停顿、职业流程和身体反应来显露心理与阶层关系。
3. 不要解释世界观，不要总结主题，不要写 AI 式结语。
4. 保持冷静、克制和异化感，让角色被制度和资源压力一步步压窄。
5. 如果资料里包含上一场摘要、累计历史或遗留线头，必须把它们视为同一部长篇中的连续章节，而不是重写开头。
6. 只输出正文，不写标题和说明。
""".strip()


def format_core_anchor(anchor: dict[str, str]) -> str:
    return (
        f"- 核心创伤：{anchor['core_wound']}\n"
        f"- 终极渴望：{anchor['ultimate_desire']}\n"
        f"- 对外伪装：{anchor['public_mask']}"
    )


def format_resource_pool(resource_state: dict[str, object]) -> str:
    stats = resource_state["stats"]
    stats_line = "，".join(f"{key}={value}" for key, value in stats.items())
    return (
        f"- 当前数值：{stats_line}\n"
        f"- 失败后果：{resource_state['failure_condition']}\n"
        f"- 生存原则：{resource_state['pressure_note']}"
    )


def format_memory_pool(memory_state: dict[str, object], cognition_mode: str) -> str:
    if cognition_mode != "eviction_budget" or not memory_state:
        return "（本场景无认知预算压力）"
    capacity = int(memory_state.get("capacity", 0))
    used = int(memory_state.get("used", 0))
    reserve_floor = int(memory_state.get("reserve_floor", 0))
    free = capacity - used
    loaded = memory_state.get("loaded_contexts", []) or []
    resident = memory_state.get("resident_memories", []) or []
    loaded_line = "；".join(
        f"{item.get('label', '未知装载')}({item.get('weight', 0)})" for item in loaded[-3:]
    ) or "（暂无装载包）"
    resident_line = "；".join(
        f"{item.get('label', '未知记忆')}({item.get('weight', 0)})" for item in resident[-4:]
    ) or "（暂无可清退记忆）"
    return (
        f"- 预算：used={used}/{capacity}，free={free}，reserve_floor={reserve_floor}\n"
        f"- 已装载上下文：{loaded_line}\n"
        f"- 可清退私人记忆：{resident_line}\n"
        f"- 提醒：{memory_state.get('pressure_note', '（无）')}"
    )


def format_memory_snapshot(memory_state: dict[str, dict[str, object]]) -> str:
    if not memory_state:
        return "（暂无认知预算信息）"
    lines: list[str] = []
    for name, item in memory_state.items():
        capacity = int(item.get("capacity", 0))
        if capacity <= 0:
            continue
        used = int(item.get("used", 0))
        free = capacity - used
        loaded = "；".join(
            f"{ctx.get('label', '未知')}({ctx.get('weight', 0)})"
            for ctx in (item.get('loaded_contexts', []) or [])[-3:]
        ) or "（暂无）"
        resident = "；".join(
            f"{memory.get('label', '未知')}({memory.get('weight', 0)})"
            for memory in (item.get('resident_memories', []) or [])[-3:]
        ) or "（暂无）"
        lines.append(
            f"- {name}：used={used}/{capacity}，free={free}｜已装载：{loaded}｜可清退：{resident}"
        )
    return "\n".join(lines) if lines else "（暂无认知预算信息）"


def format_memory_eviction_log(
    eviction_log: list[dict[str, object]],
    speaker: str | None = None,
    limit: int = 5,
) -> str:
    if not eviction_log:
        return "（暂无清退记录）"
    items = eviction_log
    if speaker:
        items = [item for item in items if item.get('speaker') == speaker]
    if not items:
        return "（暂无清退记录）"
    selected = items[-limit:] if limit > 0 else items
    lines = []
    for item in selected:
        lines.append(
            f"- 第 {item.get('round_index', '?')} 轮｜{item.get('speaker', '未知')} 删除了 {item.get('evicted_memory_label', '未知记忆')}，"
            f"回收 {item.get('evicted_memory_cost', 0)} 预算｜残留：{item.get('evicted_memory_summary', '（无）')}"
        )
    return "\n".join(lines)


def format_public_trace(public_trace: list[dict[str, str]]) -> str:
    if not public_trace:
        return "（暂无公共窗口）"

    lines: list[str] = []
    for entry in public_trace:
        lines.append(
            f"[第 {entry['round_index']} 轮][{entry['speaker']}] "
            f"微表情：{entry['micro_expression']}｜行为与对白：{entry['action_and_dialogue']}"
        )
    return "\n".join(lines)


def format_private_memory(private_memory: list[dict[str, str]]) -> str:
    if not private_memory:
        return "（暂无私有记忆）"

    lines: list[str] = []
    for entry in private_memory:
        lines.append(
            f"[第 {entry['round_index']} 轮] 观察：{entry['observation_analysis']}｜"
            f"情绪：{entry['emotional_shift']}｜目的：{entry['hidden_agenda']}"
        )
    return "\n".join(lines)


def format_relationship_view(
    dynamic_relationships: dict[str, dict[str, str]],
    speaker: str,
) -> str:
    relationships = dynamic_relationships.get(speaker, {})
    if not relationships:
        return "（暂无关系标签）"
    return "\n".join(f"- 对 {target}：{label}" for target, label in relationships.items())


def format_relationship_snapshot(dynamic_relationships: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    for observer, mapping in dynamic_relationships.items():
        for target, label in mapping.items():
            lines.append(f"- {observer} -> {target}：{label}")
    return "\n".join(lines) if lines else "（暂无关系变化）"


def format_chapter_history(chapter_history: list[dict[str, object]], limit: int = 3) -> str:
    if not chapter_history:
        return "（暂无累计历史）"
    selected = chapter_history[-limit:] if limit > 0 else chapter_history
    lines: list[str] = []
    start_index = len(chapter_history) - len(selected) + 1
    for offset, item in enumerate(selected, start=start_index):
        lines.append(
            f"- 第 {offset} 场｜开场：{item.get('opening_time_marker', '未知时间')} @ {item.get('opening_location', '未知地点')}｜"
            f"结尾：{item.get('ending_time_marker', '未知时间')} @ {item.get('ending_location', '未知地点')}｜"
            f"摘要：{item.get('chapter_summary', '（无）')}"
        )
    return "\n".join(lines)


def format_carryover_threads(threads: list[str]) -> str:
    if not threads:
        return "（暂无遗留线头）"
    return "\n".join(f"- {item}" for item in threads)


def format_continuity_summary(summary: dict[str, object]) -> str:
    if not summary:
        return "（暂无连续性摘要）"
    return (
        f"- 场景摘要：{summary.get('chapter_summary', '（无）')}\n"
        f"- 开场坐标：{summary.get('opening_time_marker', '未知时间')} @ {summary.get('opening_location', '未知地点')}\n"
        f"- 结尾坐标：{summary.get('ending_time_marker', '未知时间')} @ {summary.get('ending_location', '未知地点')}\n"
        f"- 连续性建议：{summary.get('continuity_decision', 'retain')}｜{summary.get('continuity_reason', '（无）')}\n"
        f"- 不可逆变化：{summary.get('irreversible_change', '（无）')}\n"
        f"- 下一场压力：{summary.get('next_scene_pressure', '（无）')}\n"
        f"- 遗留线头：\n{format_carryover_threads(summary.get('carryover_threads', []))}"
    )


def format_showrunner_plan(plan: dict[str, object]) -> str:
    if not plan:
        return "（节拍表尚未生成）"

    beats = plan.get("forced_beats", [])
    beat_lines = []
    for beat in beats:
        beat_lines.append(
            f"  - 第 {beat['round_index']} 轮：{beat['dramatic_function']}｜"
            f"强制事件：{beat['forced_event']}｜目标变化：{beat['target_shift']}"
        )
    beat_block = "\n".join(beat_lines) if beat_lines else "  - （无）"
    return (
        f"- 场景目的：{plan['scene_purpose']}\n"
        f"- 目标结局：{plan['target_ending']}\n"
        f"- 核心冲突：{plan['core_conflict']}\n"
        f"- 隐藏伏笔：{plan['hidden_foreshadowing']}\n"
        f"- 语气护栏：{plan['tone_guardrail']}\n"
        f"- 连续性模式：{plan.get('continuity_mode', 'retain')}｜{plan.get('continuity_rationale', '（无）')}\n"
        f"- 开场坐标：{plan.get('opening_time_marker', '未知时间')} @ {plan.get('opening_location', '未知地点')}\n"
        f"- 强制节拍：\n{beat_block}"
    )


def format_current_beat(plan: dict[str, object], round_index: int) -> str:
    for beat in plan.get("forced_beats", []):
        if beat["round_index"] == round_index:
            return (
                f"第 {beat['round_index']} 轮｜{beat['dramatic_function']}｜"
                f"强制事件：{beat['forced_event']}｜目标变化：{beat['target_shift']}"
            )
    return "本轮没有额外节拍，但冲突不能软化。"


def format_symbolism_plan(plan: dict[str, object]) -> str:
    if not plan:
        return "（潜台词方案尚未生成）"

    cue_lines = []
    for cue in plan.get("imagery_cues", []):
        cue_lines.append(
            f"- 物象：{cue['motif']}｜表面质感：{cue['sensory_surface']}｜"
            f"情绪映射：{cue['emotional_mapping']}｜使用方式：{cue['usage_instruction']}"
        )
    gesture_lines = "\n".join(f"- {item}" for item in plan.get("gesture_rewrites", []))
    forbidden_lines = "\n".join(
        f"- {item}" for item in plan.get("forbidden_explicit_phrases", [])
    )
    return (
        f"- 核心潜台词：{plan['scene_subtext']}\n"
        f"- 意象建议：\n{chr(10).join(cue_lines) if cue_lines else '- （无）'}\n"
        f"- 改写建议：\n{gesture_lines if gesture_lines else '- （无）'}\n"
        f"- 禁止直白句式：\n{forbidden_lines if forbidden_lines else '- （无）'}"
    )


def format_resource_snapshot(resource_state: dict[str, dict[str, object]]) -> str:
    lines: list[str] = []
    for name, state in resource_state.items():
        stats_line = "，".join(f"{key}={value}" for key, value in state["stats"].items())
        lines.append(f"- {name}：{stats_line}｜{state['failure_condition']}")
    return "\n".join(lines) if lines else "（暂无资源信息）"


def format_scene_log(scene_log: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for entry in scene_log:
        if entry["event_type"] == "director":
            lines.append(
                f"[Director][第 {entry['round_index']} 轮][{entry['beat_focus']}] {entry['content']}"
            )
            continue
        resource_line = "，".join(
            f"{key}={value}" for key, value in entry["resource_snapshot"].items()
        )
        detail_lines = [
            f"[{entry['speaker']}][第 {entry['round_index']} 轮]",
            f"- observation_analysis: {entry['observation_analysis']}",
            f"- emotional_shift: {entry['emotional_shift']}",
            f"- hidden_agenda: {entry['hidden_agenda']}",
            f"- micro_expression: {entry['micro_expression']}",
            f"- action_and_dialogue: {entry['action_and_dialogue']}",
            f"- resource_snapshot: {resource_line}",
        ]
        if entry.get('context_load_label'):
            detail_lines.append(
                f"- context_load: {entry.get('context_load_label')} ({entry.get('context_load_cost', 0)})"
            )
        if entry.get('evicted_memory_label'):
            detail_lines.append(
                f"- evicted_memory: {entry.get('evicted_memory_label')} ({entry.get('evicted_memory_cost', 0)})"
            )
            detail_lines.append(
                f"- evicted_memory_summary: {entry.get('evicted_memory_summary', '（无）')}"
            )
        if entry.get('memory_note'):
            detail_lines.append(f"- memory_note: {entry.get('memory_note')}")
        lines.append("\n".join(detail_lines))
    return "\n\n".join(lines)
