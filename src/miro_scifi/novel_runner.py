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
    LiveShowrunnerEngine,
    LiveSymbolismEngine,
    MockCharacterEngine,
    MockShowrunnerEngine,
    MockSymbolismEngine,
)
from .graph import build_scene_graph, create_initial_state
from .prompts import WORLD_CONTEXT_PROMPT, default_character_profiles
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


def build_idea_registry() -> dict[str, NovelIdea]:
    ideas = [idea_echo_tax(), idea_cry_guarantee()]
    return {idea.key: idea for idea in ideas}


def default_novel_idea() -> NovelIdea:
    return build_idea_registry()["echo_tax"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a longform social sci-fi novel from multi-scene LangGraph scenes."
    )
    parser.add_argument("--mode", choices=["live", "mock"], default="live")
    parser.add_argument(
        "--idea",
        default="cry_guarantee",
        choices=sorted(build_idea_registry().keys()),
        help="选择内置小说 idea。默认使用新的 cry_guarantee。",
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
            "writer": LiveSceneWriter(
                model=args.writer_model,
                timeout=240,
            ),
        }
    return {
        "character_engine": MockCharacterEngine(),
        "showrunner_engine": MockShowrunnerEngine(),
        "symbolism_engine": MockSymbolismEngine(),
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
    character_a, character_b = default_character_profiles()
    runtime = build_runtime(args)
    graph = build_scene_graph(
        character_a=character_a,
        character_b=character_b,
        character_engine=runtime["character_engine"],
        showrunner_engine=runtime["showrunner_engine"],
        symbolism_engine=runtime["symbolism_engine"],
        writer=runtime["writer"],
    )

    start_scene = max(1, args.start_scene)
    outlines = idea.scene_outlines[start_scene - 1 :]
    if args.max_scenes and args.max_scenes > 0:
        outlines = outlines[: args.max_scenes]

    previous_state: dict[str, Any] | None = load_resume_state(args)
    results: list[dict[str, Any]] = []
    for offset, scene in enumerate(outlines, start=start_scene):
        initial_state = create_initial_state(
            characters=[character_a, character_b],
            max_turns=scene.turns,
            scene_brief=f"{scene.title}：{scene.brief}",
            world_context=idea.world_context,
            chapter_target=scene.chapter_target,
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
