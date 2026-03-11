from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .engine import (
    LiveCharacterEngine,
    LiveContinuityEngine,
    LiveShowrunnerEngine,
    LiveSymbolismEngine,
    MockCharacterEngine,
    MockContinuityEngine,
    MockShowrunnerEngine,
    MockSymbolismEngine,
)
from .graph import build_scene_graph, create_initial_state
from .prompts import WORLD_CONTEXT_PROMPT, context_limit_character_profiles, default_character_profiles
from .writer import LiveSceneWriter, MockSceneWriter


@dataclass(frozen=True)
class SceneOutline:
    title: str
    brief: str
    turns: int = 2
    chapter_target: str = "1500 字左右"


@dataclass(frozen=True)
class NovelIdea:
    key: str
    title: str
    logline: str
    themes: list[str]
    tonal_guardrail: str
    world_context: str
    scene_outlines: list[SceneOutline]


def idea_echo_tax() -> NovelIdea:
    return NovelIdea(
        key="echo_tax",
        title="《回声税》",
        logline=(
            "在情绪被量化征税的雾港，底层矿工阮宁为了保住母亲的镇静贴片额度，"
            "一步步出售自己的记忆与体面；审核员裴崧则在维护配额与秩序的过程中，"
            "看着自己被制度训练出的冷静出现一丝几乎不足以称为良知的裂口。"
        ),
        themes=[
            "情绪资本主义与底层身体的可开采化",
            "制度如何诱导人主动修剪自我",
            "职业礼貌背后的冷暴力",
            "在生存压力下被迫达成的非对称交易",
        ],
        tonal_guardrail=(
            "绝不写成爽文、悬疑逆袭或温情和解；整部小说必须保持疲惫、冷硬、"
            "流程化的压迫感，让所有选择都带着损耗。"
        ),
        world_context=WORLD_CONTEXT_PROMPT,
        scene_outlines=[
            SceneOutline(
                title="第一章：异常签注",
                brief=(
                    "凌晨采样站，阮宁为了避免账户被冻结，被迫签下一张带异常标签的确认单。"
                    "场景目标是确立两人的非对称权力关系，以及‘杯沿裂纹’这个伏笔。"
                ),
            ),
            SceneOutline(
                title="第二章：贴片宽限期",
                brief=(
                    "阮宁回到蜂巢公寓，发现母亲的镇静贴片额度只剩最后一天宽限期；"
                    "她必须接受一笔更脏、更伤身的情绪采样单。裴崧在内部系统里再次看到她的异常标签。"
                ),
            ),
            SceneOutline(
                title="第三章：脏样本复核",
                brief=(
                    "阮宁提交的高波动样本触发人工复核，她不得不重新面对裴崧。"
                    "这一次冲突从压价升级为‘是否值得保留账户资格’。"
                ),
            ),
            SceneOutline(
                title="第四章：回放店",
                brief=(
                    "为了筹够贴片费用，阮宁去地下回放店出售一段关于父亲事故的旧记忆。"
                    "裴崧则接受上级关于‘静默令’的培训，被要求提高对异常情绪的拦截率。"
                ),
            ),
            SceneOutline(
                title="第五章：回声税试点",
                brief=(
                    "雾港开始试点‘回声税’：未被平台回收的私人悲伤会被计入个人负债。"
                    "阮宁所在楼层出现集体断供，裴崧则第一次负责与街区执行队联动。"
                ),
            ),
            SceneOutline(
                title="第六章：回收厂夜班",
                brief=(
                    "阮宁进入情绪回收厂夜班清洗脏样本，希望用工伤补贴换母亲的续费。"
                    "裴崧在工厂抽查时发现，自己正在审阅的是会把人越洗越薄的生产流程。"
                ),
            ),
            SceneOutline(
                title="第七章：静默区",
                brief=(
                    "一场局部骚动后，蜂巢公寓所在街区被划入临时静默区。"
                    "阮宁必须在封控前把母亲送进低配疗养仓，裴崧则被要求完成一批强制安抚审批。"
                ),
            ),
            SceneOutline(
                title="第八章：删除权",
                brief=(
                    "母亲的情况急转直下，阮宁只剩最后一种支付方式：出售一段仍未被平台收录的家庭记忆。"
                    "裴崧要在‘合规’和‘留下漏洞给自己’之间选更安全的那一个。"
                ),
            ),
            SceneOutline(
                title="第九章：天亮以前",
                brief=(
                    "天亮前的最终确认窗口开启。阮宁要决定是否用最后一段完整的自我换取账户续命，"
                    "裴崧则必须在系统追责前完成最终签注。结尾必须保住制度的运转，而不是保住人的完整。"
                ),
            ),
        ],
    )


def idea_cry_guarantee() -> NovelIdea:
    return NovelIdea(
        key="cry_guarantee",
        title="《哭声担保》",
        logline=(
            "在雾港，未被平台回收的悲伤会被视为公共风险负债。为了保住母亲的呼吸贴片与疗养仓资格，"
            "阮宁不得不把自己的公开哀悼权抵押出去；裴崧则负责给这些哭声定价，"
            "判断一个人还剩多少悲伤能被允许保留在人类范围之内。"
        ),
        themes=[
            "哀悼权的金融化与商品化",
            "私人哭声如何被制度改造成可担保资产",
            "职业礼貌包装下的程序暴力",
            "生存与体面之间不可逆的折旧关系",
        ],
        tonal_guardrail=(
            "这是一部更窄、更冷、更像债务文书缝里长出来的社会派科幻。"
            "不要让角色互相救赎，不要让制度被一时良知打断，只写人如何在规则里被一点点压缩。"
        ),
        world_context=(
            f"{WORLD_CONTEXT_PROMPT}\n\n"
            "新增制度背景：雾港开始试点‘公共哀悼担保协议’。平台认为未经回收的私人哭声会造成群体情绪感染，"
            "因此允许底层居民把自己的公开哀悼权、葬礼发言权、家庭旧录像中的哭声片段抵押给平台换取临时补贴。"
            "一旦违约，个人将失去在公开空间表达悲伤的资格，连葬礼也会被系统静音处理。"
        ),
        scene_outlines=[
            SceneOutline(
                title="第一章：担保窗口",
                brief=(
                    "阮宁来到采样站的担保窗口，想用自己的公开哀悼权换取母亲下周的呼吸贴片。"
                    "裴崧负责审核她是否有资格签署‘哭声担保协议’。场景必须建立：哭声也能抵押、制度如何定价悲伤、"
                    "以及两人之间的冷硬职业关系。"
                ),
                turns=2,
                chapter_target="1400 字左右",
            ),
            SceneOutline(
                title="第二章：家属静默单",
                brief=(
                    "阮宁回到蜂巢公寓，发现楼里一户工伤家属因拒绝交出葬礼录像被贴了静默单。"
                    "她母亲的疗养仓开始提示氧雾不足。裴崧在内部系统看到阮宁的担保资格被标成‘高违约风险’，"
                    "却仍要继续往下推进审批。"
                ),
                turns=2,
                chapter_target="1400 字左右",
            ),
            SceneOutline(
                title="第三章：哭声回收站",
                brief=(
                    "为了补足担保差额，阮宁去地下哭声回收站卖掉一段父亲葬礼上的旧录像音轨。"
                    "裴崧同时接到抽查命令，要核实她是否私自保留了未经备案的家庭哀悼素材。"
                    "两人的关系从交易窗口推进到互相知道对方会成为自己生存链条上一段难以拆掉的部件。"
                ),
                turns=2,
                chapter_target="1400 字左右",
            ),
            SceneOutline(
                title="第四章：天亮前的静音",
                brief=(
                    "最终确认窗口开启。阮宁只剩最后一项可抵押资产：自己在母亲死后公开哭出第一声的权利。"
                    "裴崧要在系统追责前完成签注。结尾必须冷：账户和贴片可以续上一段时间，"
                    "但阮宁的悲伤从此在制度层面失去外放许可；裴崧也没有改变什么，只是把裂了口的杯子重新放回工位。"
                ),
                turns=2,
                chapter_target="1600 字左右",
            ),
        ],
    )


def idea_dream_lease() -> NovelIdea:
    return NovelIdea(
        key="dream_lease",
        title="《梦层续租》",
        logline=(
            "在雾港，完整深睡期被平台当作城市预测系统的夜间基础设施出租。为了给母亲续上疗养仓的夜间供氧，"
            "阮宁不得不把自己最后还能连续做完的梦抵押出去；裴崧则负责判定她还剩多少睡眠完整性可以被合法切走。"
        ),
        themes=[
            "睡眠权如何被拆成可租赁的基础设施",
            "疲惫社会里身体连续性的金融化",
            "夜间劳动与家庭照护被同一套系统计息",
            "人为了续命，如何一步步失去做完整梦的资格",
        ],
        tonal_guardrail=(
            "保持冷、窄、失眠般的压迫感。不要把它写成赛博奇观或逆袭故事，"
            "只写夜里那些不得不签字、不得不熬着的人。"
        ),
        world_context=(
            f"{WORLD_CONTEXT_PROMPT}\n\n"
            "新增制度背景：雾港开始推行‘梦层续租协议’。平台把居民未受干扰的深度睡眠切片视为高价值基础设施，"
            "可租给城市预测模型、安抚系统和高端陪伴服务。底层居民可以抵押连续睡眠权、特定梦境层与醒来后的完整记忆，"
            "来换取氧雾、电力、疗养仓夜间供能和债务展期。违约者会被强制接入碎梦播报，失去在公共夜间休息区完整睡眠的资格。"
        ),
        scene_outlines=[
            SceneOutline(
                title="第一章：续租窗口",
                brief=(
                    "凌晨三点五十七分，阮宁来到第七码头的夜间睡眠续租窗口，想用自己最后一段完整深睡期给母亲续上疗养仓夜间供氧。"
                    "裴崧负责审核她的梦层纯度与违约风险。场景必须建立：睡眠也能被租、制度如何定价疲惫、以及两人之间冷硬的职业关系。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第二章：电梯井低鸣",
                brief=(
                    "阮宁离开窗口后没有真正脱身，她在蜂巢公寓电梯井口接到裴崧发来的远程追保催告：若不补交梦层稳定性证明，"
                    "疗养仓夜间供氧仍会在天亮前降级。这一章要尽量保留同一夜的时间流与上一场未散的疲惫。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第三章：拆梦工位",
                brief=(
                    "为了补足差额，阮宁在下一夜进入下城的拆梦工位，把一段仍能完整回到父亲事故前夕的梦切给黑市中介。"
                    "裴崧奉命抽查，核实她是否私自保留了未备案的深睡记忆。剧情应从上一夜 shift 到更脏的地下链条。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第四章：白昼回访席",
                brief=(
                    "天亮后的治理局回访席里，阮宁因为连续失眠出现梦游性断片，却还要接受裴崧的人工问询。"
                    "他们不再像第一次见面那样陌生，但这种熟悉只意味着彼此更懂对方会在哪一格表单上施压。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第五章：关灯协议",
                brief=(
                    "最终签注前夜，阮宁只剩最后一项可抵押资产：母亲死后她还能做一次完整梦的权利。"
                    "裴崧要在系统追责前完成签注。结尾必须冷：疗养仓和供氧可以暂时续上，"
                    "但阮宁在制度层面失去连续做梦的资格；裴崧也只是把那只裂了口的保温杯重新放回工位。"
                ),
                turns=1,
                chapter_target="2400 字左右",
            ),
        ],
    )


def idea_context_limit() -> NovelIdea:
    return NovelIdea(
        key="context_limit",
        title="《上下文法则》",
        logline=(
            "在曲率航行依赖人脑临时承载高维导航包的时代，远航高薪岗位只向愿意删掉过去的人开放。"
            "底舱领航员候补周惟和认知清退审核官岑簌，在一次次装载与失认之间，"
            "共同见证制度如何把‘我是谁’改造成一种可以腾挪的储存空间。"
        ),
        themes=[
            "认知带宽被金融化之后，人格如何成为可调度基础设施",
            "技能与记忆的零和博弈",
            "底层向上流动的代价不再只是身体，而是过去本身",
            "程序化审查如何把失去自我写成合规流程",
        ],
        tonal_guardrail=(
            "保持冷、窄、失认般的剥夺感。不要写英雄主义飞船奇观，不要写热血远航，"
            "只写人在装载包和清退令之间，怎样一点点失去能把自己叫回来的词。"
        ),
        world_context=(
            "时间：2197 年。\n"
            "地点：地月轨道远航工业带与太阳系外环启航走廊。\n\n"
            "社会背景：\n"
            "人类已经掌握曲率航行，但飞船无法把全部拓扑导航、引力折叠与异星语义协议外包给机器。"
            "最稳定的做法，是把高密度知识包直接压进经过基因筛选的人脑。\n"
            "颅骨容积和突触代谢上限固定不变，每装进一段新的上下文，就必须删掉等量的旧记忆。\n"
            "远航局把这种制度称为‘认知清退’：童年、亲属称谓、恋爱记忆、海潮气味，"
            "都可以被量化成可回收储位。高薪岗位和家属赎买额度，只发给愿意腾出更多脑内空间的人。\n\n"
            "基调要求：\n"
            "1. 这是社会派硬科幻，不是太空冒险爽文。\n"
            "2. 角色首先服从合同、预算、审查和生存压力，而不是理想主义。\n"
            "3. 不要写成‘牺牲自我换来伟大胜利’，只写制度如何让人把失去理解为工作流程。\n"
            "4. 允许时间连续，也允许空间跳转，但每一场都要保留‘还有什么已经想不起来’的空洞。"
        ),
        scene_outlines=[
            SceneOutline(
                title="第一章：装载窗口",
                brief=(
                    "月轨远航局 A-12 认知装载舱里，周惟为了拿到弧灯号的底舱领航合同，必须接收第一份高密度导航包。"
                    "岑簌负责审核他的预算是否足够，并要求他主动删除一段私人记忆腾空间。场景必须建立：记忆可以被清退、制度如何定价人格连续性、以及两人之间冷硬的职业关系。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第二章：失认宿舍",
                brief=(
                    "装载结束后的同一夜，周惟回到低重力宿舍，发现自己已经叫不出妹妹的乳名。"
                    "远航局又推来一份补丁包，要求他在天亮前再腾出一段可验证的私人记忆；岑簌通过远程复核链路追着他补齐预算。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第三章：语义压舱室",
                brief=(
                    "为了通过外环启航审批，周惟必须额外装载一份异星外交语义包。"
                    "这次需要腾出的不再是名字，而是某种更接近爱和安慰的感觉。岑簌奉命在白色压舱室里完成监督，确保没有任何私人残留占住预算。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第四章：白舱回访",
                brief=(
                    "启航前的白舱回访席里，周惟已经出现轻微失认和人格断片，却还要接受岑簌的人工问询。"
                    "他们不再陌生，但这种熟悉只是因为岑簌越来越清楚，周惟还能从哪里继续被切走。"
                ),
                turns=1,
                chapter_target="2200 字左右",
            ),
            SceneOutline(
                title="第五章：盲跳许可",
                brief=(
                    "最终盲跳许可签发前，弧灯号要求周惟装入终航星图与失稳补丁。"
                    "他只剩最后一段足以把自己叫回来的私人记忆可删；岑簌也必须在系统追责前签发许可。结尾必须冷：飞船可以起跳，合同和赎买额度也能兑现，但被删掉的那部分不会再回来。"
                ),
                turns=1,
                chapter_target="2400 字左右",
            ),
        ],
    )


def build_characters_for_idea(idea_key: str):
    if idea_key == "context_limit":
        return context_limit_character_profiles()
    return default_character_profiles()


def build_scene_runtime_overrides(idea_key: str) -> dict[str, Any]:
    if idea_key == "context_limit":
        return {
            "cognition_mode": "eviction_budget",
        }
    return {
        "cognition_mode": "standard",
    }




def build_idea_registry() -> dict[str, NovelIdea]:
    ideas = [idea_echo_tax(), idea_cry_guarantee(), idea_dream_lease(), idea_context_limit()]
    return {idea.key: idea for idea in ideas}


def default_novel_idea() -> NovelIdea:
    return build_idea_registry()["context_limit"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a longform social sci-fi novel from multi-scene LangGraph scenes."
    )
    parser.add_argument("--mode", choices=["live", "mock"], default="live")
    parser.add_argument(
        "--fast-lane",
        action="store_true",
        help="live 模式下只保留 Character + Writer 为真实模型，Showrunner / Symbolism / Continuity 使用内置快速策略。",
    )
    parser.add_argument(
        "--idea",
        default="context_limit",
        choices=sorted(build_idea_registry().keys()),
        help="选择内置小说 idea。默认使用新的 context_limit。",
    )
    parser.add_argument(
        "--character-model",
        default="openai:Pro/zai-org/GLM-5",
        help="角色节点模型。若使用 SiliconFlow，保留默认即可。",
    )
    parser.add_argument(
        "--showrunner-model",
        help="Showrunner 节点模型；默认跟随 --character-model。",
    )
    parser.add_argument(
        "--symbolism-model",
        help="Symbolism 节点模型；默认跟随 --character-model。",
    )
    parser.add_argument(
        "--writer-model",
        default="openai:Pro/zai-org/GLM-5",
        help="Writer 节点模型；若使用 SiliconFlow，保留默认即可。",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / "MyInvestment" / ".env.local",
        help="额外加载的 env 文件，默认使用 ~/MyInvestment/.env.local。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/novel_runner_novel.md"),
        help="小说输出路径。",
    )
    parser.add_argument(
        "--idea-output",
        type=Path,
        default=Path("outputs/novel_runner_idea.md"),
        help="Idea / outline 输出路径。",
    )
    parser.add_argument(
        "--states-dir",
        type=Path,
        default=Path("outputs/novel_runner_states"),
        help="每个场景 state JSON 的输出目录。",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="从 start-scene 开始最多跑 N 个场景；0 表示跑到结尾。",
    )
    parser.add_argument(
        "--start-scene",
        type=int,
        default=1,
        help="从第几个场景开始跑，默认从 1 开始。",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="显式指定上一场景的 state JSON，用于续跑。",
    )
    return parser


def maybe_load_siliconflow_env(env_file: Path | None) -> None:
    load_dotenv()
    if env_file and env_file.exists():
        load_dotenv(env_file, override=False)
    silicon_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    silicon_base = os.getenv("SILICONFLOW_BASE_URL", "").strip()
    if silicon_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = silicon_key
    if not os.getenv("OPENAI_BASE_URL"):
        if silicon_base:
            normalized = silicon_base.removesuffix("/chat/completions")
            os.environ["OPENAI_BASE_URL"] = normalized
        elif silicon_key:
            os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"


def build_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "live":
        if args.fast_lane:
            return {
                "character_engine": MockCharacterEngine(),
                "showrunner_engine": MockShowrunnerEngine(),
                "symbolism_engine": MockSymbolismEngine(),
                "continuity_engine": MockContinuityEngine(),
                "writer": LiveSceneWriter(
                    model=args.writer_model,
                    timeout=240,
                ),
            }
        return {
            "character_engine": LiveCharacterEngine(
                model=args.character_model,
                timeout=180,
            ),
            "showrunner_engine": LiveShowrunnerEngine(
                model=args.showrunner_model or args.character_model,
                timeout=180,
            ),
            "symbolism_engine": LiveSymbolismEngine(
                model=args.symbolism_model or args.character_model,
                timeout=180,
            ),
            "continuity_engine": LiveContinuityEngine(
                model=args.showrunner_model or args.character_model,
                timeout=180,
            ),
            "writer": LiveSceneWriter(
                model=args.writer_model,
                timeout=240,
            ),
        }
    return {
        "character_engine": MockCharacterEngine(),
        "showrunner_engine": MockShowrunnerEngine(),
        "symbolism_engine": MockSymbolismEngine(),
        "continuity_engine": MockContinuityEngine(),
        "writer": MockSceneWriter(),
    }


def seed_state_from_previous(
    state: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if not previous:
        return state
    state["resource_state"] = previous["resource_state"]
    state["dynamic_relationships"] = previous["dynamic_relationships"]
    state["core_anchors"] = previous["core_anchors"]
    state["continuity_summary"] = previous.get("continuity_summary", {})
    state["chapter_history"] = list(previous.get("chapter_history", []))
    state["carryover_threads"] = list(previous.get("carryover_threads", []))
    state["last_scene_summary"] = (
        previous.get("last_scene_summary")
        or previous.get("continuity_summary", {}).get("chapter_summary", "")
    )
    state["current_location"] = (
        previous.get("current_location")
        or previous.get("continuity_summary", {}).get("ending_location", "")
    )
    state["time_marker"] = (
        previous.get("time_marker")
        or previous.get("continuity_summary", {}).get("ending_time_marker", "")
    )
    state["cognition_mode"] = previous.get("cognition_mode", state.get("cognition_mode", "standard"))
    state["memory_state"] = previous.get("memory_state", state.get("memory_state", {}))
    state["memory_archive"] = previous.get("memory_archive", state.get("memory_archive", {}))
    state["memory_eviction_log"] = list(previous.get("memory_eviction_log", []))
    return state


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_resume_state(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.resume_state:
        return load_json(args.resume_state)
    if args.start_scene <= 1:
        return None
    candidate = args.states_dir / f"scene_{args.start_scene - 1:02d}.json"
    if candidate.exists():
        return load_json(candidate)
    return None


def load_existing_results(states_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not states_dir.exists():
        return results
    for path in sorted(states_dir.glob("scene_*.json")):
        try:
            results.append(load_json(path))
        except Exception:
            continue
    results.sort(key=lambda item: int(item.get("scene_index", 0)))
    return results


def run_novel(args: argparse.Namespace) -> tuple[NovelIdea, list[dict[str, Any]]]:
    maybe_load_siliconflow_env(args.env_file)
    idea_registry = build_idea_registry()
    idea = idea_registry[args.idea]
    character_a, character_b = build_characters_for_idea(idea.key)
    runtime = build_runtime(args)
    graph = build_scene_graph(
        character_a=character_a,
        character_b=character_b,
        character_engine=runtime["character_engine"],
        showrunner_engine=runtime["showrunner_engine"],
        symbolism_engine=runtime["symbolism_engine"],
        continuity_engine=runtime["continuity_engine"],
        writer=runtime["writer"],
    )

    start_scene = max(1, args.start_scene)
    outlines = idea.scene_outlines[start_scene - 1 :]
    if args.max_scenes and args.max_scenes > 0:
        outlines = outlines[: args.max_scenes]

    previous_state: dict[str, Any] | None = load_resume_state(args)
    scene_overrides = build_scene_runtime_overrides(idea.key)
    results: list[dict[str, Any]] = []
    for offset, scene in enumerate(outlines, start=start_scene):
        initial_state = create_initial_state(
            characters=[character_a, character_b],
            max_turns=scene.turns,
            scene_brief=f"{scene.title}：{scene.brief}",
            world_context=idea.world_context,
            chapter_target=scene.chapter_target,
            cognition_mode=scene_overrides["cognition_mode"],
        )
        seeded_state = seed_state_from_previous(initial_state, previous_state)
        result = graph.invoke(seeded_state)
        result["scene_index"] = offset
        result["scene_title"] = scene.title
        result["scene_outline"] = scene.brief
        results.append(result)
        previous_state = result
    return idea, results


def render_idea_markdown(idea: NovelIdea) -> str:
    lines = [
        f"# {idea.title}",
        "",
        f"**Idea Key**：{idea.key}",
        "",
        f"**Logline**：{idea.logline}",
        "",
        "## Themes",
    ]
    lines.extend(f"- {theme}" for theme in idea.themes)
    lines.extend(
        [
            "",
            "## Tonal Guardrail",
            idea.tonal_guardrail,
            "",
            "## Scene Outline",
        ]
    )
    for index, scene in enumerate(idea.scene_outlines, start=1):
        lines.append(f"### {index}. {scene.title}")
        lines.append(scene.brief)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_novel_markdown(idea: NovelIdea, results: list[dict[str, Any]]) -> str:
    parts = [
        f"# {idea.title}",
        "",
        f"> {idea.logline}",
        "",
    ]
    for result in results:
        parts.append(f"## {result['scene_title']}")
        parts.append("")
        parts.append(result["chapter_text"].strip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def count_non_whitespace_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def save_outputs(
    *,
    idea: NovelIdea,
    results: list[dict[str, Any]],
    output: Path,
    idea_output: Path,
    states_dir: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    idea_output.parent.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        state_path = states_dir / f"scene_{result['scene_index']:02d}.json"
        state_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    merged_results = load_existing_results(states_dir)
    idea_text = render_idea_markdown(idea)
    novel_text = render_novel_markdown(idea, merged_results)
    idea_output.write_text(idea_text, encoding="utf-8")
    output.write_text(novel_text, encoding="utf-8")

    chapter_body = "\n\n".join(result["chapter_text"] for result in merged_results)
    return {
        "idea_key": idea.key,
        "idea_path": str(idea_output),
        "novel_path": str(output),
        "states_dir": str(states_dir),
        "scene_count": len(merged_results),
        "char_count": count_non_whitespace_chars(chapter_body),
    }


def main() -> None:
    args = build_parser().parse_args()
    idea, results = run_novel(args)
    summary = save_outputs(
        idea=idea,
        results=results,
        output=args.output,
        idea_output=args.idea_output,
        states_dir=args.states_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
