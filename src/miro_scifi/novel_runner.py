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
    title: str
    logline: str
    themes: list[str]
    tonal_guardrail: str
    world_context: str
    scene_outlines: list[SceneOutline]


def default_novel_idea() -> NovelIdea:
    return NovelIdea(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a longform social sci-fi novel from multi-scene LangGraph scenes."
    )
    parser.add_argument("--mode", choices=["live", "mock"], default="live")
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
        default=Path("outputs/echo_tax_novel.md"),
        help="小说输出路径。",
    )
    parser.add_argument(
        "--idea-output",
        type=Path,
        default=Path("outputs/echo_tax_idea.md"),
        help="Idea / outline 输出路径。",
    )
    parser.add_argument(
        "--states-dir",
        type=Path,
        default=Path("outputs/echo_tax_states"),
        help="每个场景 state JSON 的输出目录。",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="只跑前 N 个场景；0 表示跑完整部。",
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


def seed_state_from_previous(state: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return state
    state["resource_state"] = previous["resource_state"]
    state["dynamic_relationships"] = previous["dynamic_relationships"]
    state["core_anchors"] = previous["core_anchors"]
    return state


def run_novel(args: argparse.Namespace) -> tuple[NovelIdea, list[dict[str, Any]]]:
    maybe_load_siliconflow_env(args.env_file)
    idea = default_novel_idea()
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

    outlines = idea.scene_outlines
    if args.max_scenes and args.max_scenes > 0:
        outlines = outlines[: args.max_scenes]

    previous_state: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for index, scene in enumerate(outlines, start=1):
        initial_state = create_initial_state(
            characters=[character_a, character_b],
            max_turns=scene.turns,
            scene_brief=f"{scene.title}：{scene.brief}",
            world_context=idea.world_context,
            chapter_target=scene.chapter_target,
        )
        seeded_state = seed_state_from_previous(initial_state, previous_state)
        result = graph.invoke(seeded_state)
        result["scene_index"] = index
        result["scene_title"] = scene.title
        result["scene_outline"] = scene.brief
        results.append(result)
        previous_state = result
    return idea, results


def render_idea_markdown(idea: NovelIdea) -> str:
    lines = [
        f"# {idea.title}",
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

    idea_text = render_idea_markdown(idea)
    novel_text = render_novel_markdown(idea, results)
    idea_output.write_text(idea_text, encoding="utf-8")
    output.write_text(novel_text, encoding="utf-8")

    for result in results:
        state_path = states_dir / f"scene_{result['scene_index']:02d}.json"
        state_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    chapter_body = "\n\n".join(result["chapter_text"] for result in results)
    return {
        "idea_path": str(idea_output),
        "novel_path": str(output),
        "states_dir": str(states_dir),
        "scene_count": len(results),
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
