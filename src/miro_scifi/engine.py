from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .models import AgentOutput, CharacterProfile, SceneState
from .prompts import build_character_system_prompt, build_character_user_prompt


class CharacterEngine(Protocol):
    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentOutput:
        ...


@dataclass
class LiveCharacterEngine:
    model: str = "openai:gpt-4o"
    temperature: float = 0.8
    timeout: int = 60
    _structured_model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_credentials()
        chat_model = init_chat_model(
            self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=2,
        )
        self._structured_model = chat_model.with_structured_output(AgentOutput)

    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentOutput:
        result = self._structured_model.invoke(
            [
                SystemMessage(content=build_character_system_prompt(profile)),
                HumanMessage(content=build_character_user_prompt(profile, state)),
            ]
        )
        if isinstance(result, AgentOutput):
            return result
        return AgentOutput.model_validate(result)

    def _validate_credentials(self) -> None:
        provider = self.model.split(":", 1)[0] if ":" in self.model else ""
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "使用 OpenAI 模式时需要设置 OPENAI_API_KEY。"
            )
        if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "使用 Anthropic 模式时需要设置 ANTHROPIC_API_KEY，并安装对应 provider 包。"
            )


class MockCharacterEngine:
    def __init__(self) -> None:
        self._script = {
            "阮宁": [
                AgentOutput(
                    inner_thought="她知道这一次再被判定为低质样本，母亲下周的镇静贴片就会断供。她不想求任何人，但更不想被系统看见自己的慌。",
                    public_action="她把袖口往下扯了扯，露出手腕上反复穿刺留下的浅色疤痕，站到采样台前时肩膀有一瞬间发硬。",
                    public_dialogue="我按时来了，昨晚那批样本，你们系统是不是又给我压价了？",
                ),
                AgentOutput(
                    inner_thought="他不接她的眼神，像在看一张表。这样的人最危险，因为他们真的相信自己只是执行流程。",
                    public_action="她把下巴微微抬起，指尖却仍旧按在冰凉的采样台边缘，没有收回去。",
                    public_dialogue="如果你们要复核，就快一点。我后面还有一班清洗工。",
                ),
                AgentOutput(
                    inner_thought="门快关了。她忽然意识到，自己这一夜卖掉的不是情绪，是明天继续麻木下去的资格。",
                    public_action="她低头签完确认单，拇指在发炎的后颈接口上轻轻按了一下，像按住某种将要外溢的东西。",
                    public_dialogue="行，我提交。但异常标签你最好写清楚，别把我的账户一起冻上。",
                ),
            ],
            "裴崧": [
                AgentOutput(
                    inner_thought="她的愤怒值被压得太平，说明她已经学会在采样前自行修剪情绪。这样的人不稳定，但也最适合长期供给。",
                    public_action="他抬手调出阮宁的样本曲线，目光在悬浮屏上停了两秒，语气平得像播报天气。",
                    public_dialogue="压价是市场波动，不是针对你。你昨晚的悲伤样本纯度不够，系统只会按规则结算。",
                ),
                AgentOutput(
                    inner_thought="她在催，可配额已经逼近阈值。再放过一次高波动样本，今晚的报告就得他来背。秩序的代价从来不会落到系统头上。",
                    public_action="他把复核页面切到人工签注栏，手指停顿了一下，又继续往下滑。",
                    public_dialogue="流程快慢取决于你的样本稳定性，不取决于你的排班。保持平静，对你自己也有好处。",
                ),
                AgentOutput(
                    inner_thought="她并不特别，只是无数可替代个体中的一个。但他还是记住了她说‘别冻上我的账户’时那种过于平静的语气。",
                    public_action="他在终端上补了一条备注，随后把确认单推回给她，动作标准得近乎礼貌。",
                    public_dialogue="备注我会写明。至于冻结与否，由上层风控判断，不由我决定。",
                ),
            ],
        }

    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentOutput:
        round_index = state["turn_count"]
        scripted_turns = self._script[profile.name]
        return scripted_turns[min(round_index, len(scripted_turns) - 1)]
