from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .engine import LiveCharacterEngine, MockCharacterEngine
from .graph import build_scene_graph, create_initial_state
from .prompts import default_character_profiles, format_public_history, format_scene_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI 社会派科幻小说推演系统：Phase 1 + Phase 2 原型"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock 为本地演示；live 会调用真实 LangChain ChatModel。",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-4o",
        help="live 模式下使用的模型标识，例如 openai:gpt-4o。",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="场景总回合数，默认 3。",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="将最终 State 结果保存到指定 JSON 文件。",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    character_a, character_b = default_character_profiles()
    initial_state = create_initial_state(
        characters=[character_a, character_b],
        max_turns=args.turns,
    )

    if args.mode == "live":
        engine = LiveCharacterEngine(model=args.model)
    else:
        engine = MockCharacterEngine()

    graph = build_scene_graph(
        character_a=character_a,
        character_b=character_b,
        engine=engine,
    )
    result = graph.invoke(initial_state)

    print("=== Public History ===")
    print(format_public_history(result["public_history"]))
    print("\n=== Full Scene Log ===")
    print(format_scene_log(result["scene_log"]))

    if args.save_json:
        args.save_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nState 已保存到: {args.save_json}")


if __name__ == "__main__":
    main()
