from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class ChapterSpec:
    title: str
    target_length: str
    brief: str
    continuity_facts: list[str]


@dataclass(frozen=True)
class NovelBlueprint:
    title: str
    logline: str
    world_context: str
    tonal_guardrail: str
    motifs: list[str]
    chapters: list[ChapterSpec]


def default_blueprint() -> NovelBlueprint:
    return NovelBlueprint(
        title="《回声税》",
        logline=(
            "在情绪可以被计税、出售和清洗的雾港，底层情绪矿工阮宁为了保住母亲的镇静贴片额度，"
            "一步步出售记忆、体面与自我边界；审核员裴崧则在维护配额和秩序的过程中，"
            "看着自己赖以生存的职业冷静出现一丝不足以改变任何结果的裂口。"
        ),
        world_context=(
            "2042 年的雾港，平台垄断了情绪采样、情绪税结算、静默令审批和异常样本清洗。"
            "贫穷者出售悲伤、恐惧、羞耻和温柔来换取积分、药物和住区资格；"
            "中层审核员用职业礼貌包装流程暴力。制度不靠公开恐吓运转，而靠配额、冻结、折扣、宽限期和一张张需要签字的确认单运转。"
        ),
        tonal_guardrail=(
            "整部小说必须冷峻、克制、有疲惫感，禁止写成热血反抗、悬疑爽文或互相理解的温情故事。"
            "不要说教，不要替角色总结主题，不要在结尾给出希望性金句。"
        ),
        motifs=[
            "杯沿裂纹",
            "发炎的后颈接口",
            "灰蓝色滚动字幕",
            "贴片宽限期倒计时",
            "自动门和计时屏",
        ],
        chapters=[
            ChapterSpec(
                title="第一章：异常签注",
                target_length="2600 字左右",
                brief=(
                    "写凌晨采样站的交易窗口。阮宁为了避免账户被冻结，被迫签下一张带异常标签的确认单。"
                    "她母亲的镇静贴片额度只剩一天宽限期。裴崧用职业礼貌完成压价与签注。"
                    "要把两人的权力差、阶层差、说话方式和互相利用的关系写出来。"
                ),
                continuity_facts=[
                    "阮宁是底层情绪矿工，住在蜂巢公寓，长期出售悲伤样本。",
                    "她母亲依赖平台补贴的镇静贴片，一旦断供会迅速恶化。",
                    "裴崧是情绪治理体系的审核员，把自己训练成流程接口。",
                    "审核台边上有一只杯沿带细裂的保温杯。",
                    "结尾必须是阮宁签下异常标签，保住账户，但体面受损。",
                ],
            ),
            ChapterSpec(
                title="第二章：回放店",
                target_length="2600 字左右",
                brief=(
                    "从阮宁回到蜂巢公寓写起，母亲贴片即将断供，蜂巢楼层里有人因为回声税试点而被追缴。"
                    "阮宁为了凑钱，去地下回放店出售一段关于父亲死于采样事故的旧记忆。"
                    "裴崧在单位接到新的静默令培训，被要求提高对高波动样本的拦截率。"
                    "两人不要正面长聊，但要通过制度和记录再次牵到一起。"
                ),
                continuity_facts=[
                    "承接上一章，阮宁账户暂时没被冻结，但被挂了异常标签。",
                    "她母亲已经开始认知错乱，分不清丈夫和女儿。",
                    "回声税开始试点：未被平台回收的私人悲伤会被折算成负债。",
                    "裴崧需要完成配额，否则岗位和信用等级都会下降。",
                    "杯沿裂纹和后颈接口要继续作为物象出现。",
                ],
            ),
            ChapterSpec(
                title="第三章：静默区",
                target_length="2600 字左右",
                brief=(
                    "雾港某片区因断供和追缴引发局部骚动，阮宁所在蜂巢公寓被划入临时静默区。"
                    "她进入情绪回收厂夜班，试图用工伤补贴和脏样本奖金换母亲的续费。"
                    "裴崧被派去处理静默区的审批和抽查，看见制度怎样把人的崩溃整理成表格。"
                    "这一章要让两人的关系从单纯交易，变成彼此都知道对方是自己生存链条上一截不舒服但必要的部件。"
                ),
                continuity_facts=[
                    "阮宁已经出售了一段父亲事故记忆，身体和情绪稳定性进一步下降。",
                    "裴崧知道她的异常标签还在系统里挂着，也知道她快被推向更危险的工种。",
                    "静默区内的人会被强制降低情绪波动，以维持街区秩序。",
                    "回收厂负责清洗脏样本，工人长期接触他人的悲伤、羞耻和恐惧。",
                    "这一章结尾要逼近一个无法拖延的最终支付节点。",
                ],
            ),
            ChapterSpec(
                title="第四章：天亮以前",
                target_length="2800 字左右",
                brief=(
                    "写最终确认窗口。母亲的状况急转直下，阮宁只剩最后一种支付方式：出售一段仍未被平台收录的家庭记忆。"
                    "裴崧需要在系统追责前完成签注，既不能被看成软化，也不能给自己留下明显责任。"
                    "结局必须冷：制度继续运转，账户可以续命，但阮宁失去一部分决定她是谁的东西；"
                    "裴崧没有成为拯救者，只是把那只裂了口的杯子带回工位。"
                ),
                continuity_facts=[
                    "阮宁母亲已经接近失去基本认知能力，疗养仓和贴片都需要续费。",
                    "阮宁的 san_value、dignity 和身体状态都已逼近下限。",
                    "裴崧的配额时钟和纪律风险都在红线边缘。",
                    "不要大团圆，不要英雄行为，不要公开反抗成功。",
                    "最后一句要冷，不要抒情拔高。",
                ],
            ),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a long social sci-fi novel in a few large model calls.")
    parser.add_argument("--env-file", type=Path, default=Path.home() / "MyInvestment" / ".env.local")
    parser.add_argument("--model", default=os.getenv("SILICONFLOW_MODEL", "Pro/zai-org/GLM-5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/echo_tax_novel.md"))
    parser.add_argument("--idea-output", type=Path, default=Path("outputs/echo_tax_idea.md"))
    parser.add_argument("--max-chapters", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def load_provider_env(env_file: Path | None) -> None:
    load_dotenv()
    if env_file and env_file.exists():
        load_dotenv(env_file, override=False)
    if os.getenv("SILICONFLOW_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["SILICONFLOW_API_KEY"]
    if not os.getenv("OPENAI_BASE_URL"):
        base = os.getenv("SILICONFLOW_BASE_URL", "").strip()
        if base:
            os.environ["OPENAI_BASE_URL"] = base.removesuffix("/chat/completions")
        elif os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"


def build_client(timeout: float) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未找到可用 API Key。请检查 ~/MyInvestment/.env.local 或 OPENAI_API_KEY。")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def build_chapter_prompt(blueprint: NovelBlueprint, chapter: ChapterSpec) -> str:
    motifs = "、".join(blueprint.motifs)
    facts = "\n".join(f"- {item}" for item in chapter.continuity_facts)
    return f"""
请为这部长篇小说写出当前章节正文。

[小说标题]
{blueprint.title}

[一句话梗概]
{blueprint.logline}

[世界观]
{blueprint.world_context}

[语气护栏]
{blueprint.tonal_guardrail}

[反复出现的物象]
{motifs}

[当前章节]
{chapter.title}

[章节目标篇幅]
{chapter.target_length}

[章节任务]
{chapter.brief}

[连续性事实]
{facts}

[写作硬约束]
1. 只输出本章正文，不要标题，不要解释，不要提纲。
2. 严禁使用“他心里想”“她感到很悲伤”“制度是残酷的”这类直白写法。
3. 用环境、职业动作、身体反应、停顿、价格、配额、终端提示、器械细节来写心理和制度压力。
4. 不要把任何角色写成正义化身，也不要让冲突被理解与温情轻易化解。
5. 文风要像严肃社会派科幻，而不是网文、悬疑爽文或散文化随笔。
""".strip()


def generate_chapter(client: OpenAI, model: str, blueprint: NovelBlueprint, chapter: ChapterSpec) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.9,
        max_tokens=7000,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位冷峻、克制的社会派科幻作家。"
                    "你擅长写制度如何在日常细节里碾压人，而不是靠宏大宣言。"
                    "你拒绝大团圆、拒绝说教、拒绝网文腔。"
                ),
            },
            {"role": "user", "content": build_chapter_prompt(blueprint, chapter)},
        ],
    )
    return response.choices[0].message.content.strip()


def render_idea_markdown(blueprint: NovelBlueprint) -> str:
    lines = [
        f"# {blueprint.title}",
        "",
        f"**Logline**：{blueprint.logline}",
        "",
        "## Themes / Motifs",
    ]
    lines.extend(f"- {item}" for item in blueprint.motifs)
    lines.extend(["", "## Chapters"])
    for idx, chapter in enumerate(blueprint.chapters, start=1):
        lines.append(f"### {idx}. {chapter.title}")
        lines.append(chapter.brief)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_novel_markdown(blueprint: NovelBlueprint, chapters: list[tuple[str, str]]) -> str:
    lines = [f"# {blueprint.title}", "", f"> {blueprint.logline}", ""]
    for title, body in chapters:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def count_non_whitespace_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def main() -> None:
    args = build_parser().parse_args()
    load_provider_env(args.env_file)
    blueprint = default_blueprint()
    client = build_client(args.timeout)

    chapters = blueprint.chapters
    if args.max_chapters and args.max_chapters > 0:
        chapters = chapters[: args.max_chapters]

    generated: list[tuple[str, str]] = []
    for chapter in chapters:
        text = generate_chapter(client, args.model, blueprint, chapter)
        generated.append((chapter.title, text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.idea_output.parent.mkdir(parents=True, exist_ok=True)
    args.idea_output.write_text(render_idea_markdown(blueprint), encoding="utf-8")
    novel_text = render_novel_markdown(blueprint, generated)
    args.output.write_text(novel_text, encoding="utf-8")

    body_text = "\n\n".join(text for _, text in generated)
    print(
        json.dumps(
            {
                "idea_path": str(args.idea_output),
                "novel_path": str(args.output),
                "chapter_count": len(generated),
                "char_count": count_non_whitespace_chars(body_text),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
