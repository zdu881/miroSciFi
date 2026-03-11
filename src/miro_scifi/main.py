from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from .prompts import (
    DEFAULT_SCENE_BRIEF,
    default_character_profiles,
    format_public_trace,
    format_showrunner_plan,
)
from .writer import LiveSceneWriter, MockSceneWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI 社会派科幻小说推演系统：资源压迫 + Showrunner + Subtext Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock 为本地演示；live 会调用真实 LangChain ChatModel。",
    )
    parser.add_argument(
        "--character-model",
        default="openai:gpt-4o",
        help="角色节点使用的模型标识，例如 openai:gpt-4o。",
    )
    parser.add_argument(
        "--showrunner-model",
        help="Showrunner 节点使用的模型标识；默认跟随 --character-model。",
    )
    parser.add_argument(
        "--symbolism-model",
        help="Symbolism 节点使用的模型标识；默认跟随 --character-model。",
    )
    parser.add_argument(
        "--writer-model",
        default="anthropic:claude-3-5-sonnet-latest",
        help="Writer 节点使用的模型标识，例如 anthropic:claude-3-5-sonnet-latest。",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="场景总回合数，默认 3。",
    )
    parser.add_argument(
        "--scene-brief",
        default=DEFAULT_SCENE_BRIEF,
        help="交给 Showrunner 的场景简报。",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="将最终 State 结果保存到指定 JSON 文件。",
    )
    parser.add_argument(
        "--save-chapter",
        type=Path,
        help="将 Writer Node 生成的正文保存到文本文件。",
    )
    parser.add_argument(
        "--save-subtext",
        type=Path,
        help="将 Symbolism Node 生成的潜台词指导保存到文本文件。",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    character_a, character_b = default_character_profiles()
    initial_state = create_initial_state(
        characters=[character_a, character_b],
        max_turns=args.turns,
        scene_brief=args.scene_brief,
    )

    if args.mode == "live":
        character_engine = LiveCharacterEngine(model=args.character_model)
        showrunner_engine = LiveShowrunnerEngine(
            model=args.showrunner_model or args.character_model
        )
        symbolism_engine = LiveSymbolismEngine(
            model=args.symbolism_model or args.character_model
        )
        writer = LiveSceneWriter(model=args.writer_model)
    else:
        character_engine = MockCharacterEngine()
        showrunner_engine = MockShowrunnerEngine()
        symbolism_engine = MockSymbolismEngine()
        writer = MockSceneWriter()

    graph = build_scene_graph(
        character_a=character_a,
        character_b=character_b,
        character_engine=character_engine,
        showrunner_engine=showrunner_engine,
        symbolism_engine=symbolism_engine,
        writer=writer,
    )
    result = graph.invoke(initial_state)

    print("=== Showrunner Plan ===")
    print(format_showrunner_plan(result["showrunner_plan"]))
    print("\n=== Public Trace ===")
    print(format_public_trace(result["public_trace"]))
    print("\n=== Subtext Guide ===")
    print(result["subtext_guide"])
    print("\n=== Chapter Draft ===")
    print(result["chapter_text"])

    if args.save_json:
        args.save_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nState 已保存到: {args.save_json}")

    if args.save_chapter:
        args.save_chapter.write_text(result["chapter_text"], encoding="utf-8")
        print(f"章节草稿已保存到: {args.save_chapter}")

    if args.save_subtext:
        args.save_subtext.write_text(result["subtext_guide"], encoding="utf-8")
        print(f"潜台词指导已保存到: {args.save_subtext}")


if __name__ == "__main__":
    main()
