from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, get_args, get_origin

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .models import AgentAction, CharacterProfile, SceneState, ShowrunnerPlan, SymbolismPlan
from .prompts import (
    build_character_system_prompt,
    build_character_user_prompt,
    build_showrunner_system_prompt,
    build_showrunner_user_prompt,
    build_symbolism_system_prompt,
    build_symbolism_user_prompt,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class CharacterEngine(Protocol):
    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentAction:
        ...


class ShowrunnerEngine(Protocol):
    def plan(
        self,
        *,
        scene_brief: str,
        world_context: str,
        characters: list[CharacterProfile],
        max_turns: int,
    ) -> ShowrunnerPlan:
        ...


class SymbolismEngine(Protocol):
    def plan(
        self,
        *,
        scene_data: str,
        showrunner_plan: dict[str, object],
        state: SceneState,
    ) -> SymbolismPlan:
        ...


def validate_provider_credentials(model: str) -> None:
    provider = model.split(":", 1)[0] if ":" in model else ""
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("使用 OpenAI 模式时需要设置 OPENAI_API_KEY。")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("使用 Anthropic 模式时需要设置 ANTHROPIC_API_KEY。")


def build_template_value(annotation: Any, description: str) -> Any:
    origin = get_origin(annotation)
    if origin in {list, list[str]}:
        args = get_args(annotation)
        inner = args[0] if args else str
        return [build_template_value(inner, description)]
    if origin is dict:
        return {"key": "value"}
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return build_output_template(annotation)
        if annotation is int:
            return 1
        return f"<{description or '填写内容'}>"
    args = get_args(annotation)
    if origin is list and args:
        inner = args[0]
        return [build_template_value(inner, description)]
    return f"<{description or '填写内容'}>"


def build_output_template(model_cls: type[BaseModel]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        payload[name] = build_template_value(field.annotation, field.description or name)
    return payload


def build_json_instruction(model_cls: type[ModelT]) -> str:
    template = json.dumps(build_output_template(model_cls), ensure_ascii=False, indent=2)
    return (
        "你必须只返回一个 JSON 对象，不能输出解释、标题、项目符号、Markdown 代码块或任何额外文本。\n"
        "严禁返回 schema 元信息，例如 properties、type、title、description、required。\n"
        "请直接按下面这个实例模板的键名返回内容，并把每个值替换成真实生成结果：\n"
        f"{template}"
    )


def normalize_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.S)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    raise ValueError(f"模型没有返回可解析的 JSON：{stripped[:300]}")


def parse_json_response(model_cls: type[ModelT], raw_text: str) -> ModelT:
    json_text = extract_json_object(raw_text)
    return model_cls.model_validate_json(json_text)


@dataclass
class _BaseLiveJSONEngine:
    model: str
    temperature: float
    timeout: int
    _chat_model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        validate_provider_credentials(self.model)
        self._chat_model = init_chat_model(
            self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=2,
        )

    def invoke_json(
        self,
        *,
        model_cls: type[ModelT],
        system_prompt: str,
        user_prompt: str,
    ) -> ModelT:
        response = self._chat_model.invoke(
            [
                SystemMessage(
                    content=f"{system_prompt}\n\n{build_json_instruction(model_cls)}"
                ),
                HumanMessage(
                    content=(
                        f"{user_prompt}\n\n"
                        "再次提醒：只返回一个 JSON 对象，不要使用 Markdown 代码块。"
                    )
                ),
            ]
        )
        raw_text = normalize_content(getattr(response, "content", response))
        return parse_json_response(model_cls, raw_text)


@dataclass
class LiveCharacterEngine(_BaseLiveJSONEngine):
    model: str = "openai:gpt-4o"
    temperature: float = 0.9
    timeout: int = 60

    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentAction:
        return self.invoke_json(
            model_cls=AgentAction,
            system_prompt=build_character_system_prompt(profile),
            user_prompt=build_character_user_prompt(profile, state),
        )


@dataclass
class LiveShowrunnerEngine(_BaseLiveJSONEngine):
    model: str = "openai:gpt-4o"
    temperature: float = 0.7
    timeout: int = 60

    def plan(
        self,
        *,
        scene_brief: str,
        world_context: str,
        characters: list[CharacterProfile],
        max_turns: int,
    ) -> ShowrunnerPlan:
        return self.invoke_json(
            model_cls=ShowrunnerPlan,
            system_prompt=build_showrunner_system_prompt(),
            user_prompt=build_showrunner_user_prompt(
                scene_brief=scene_brief,
                world_context=world_context,
                characters=characters,
                max_turns=max_turns,
            ),
        )


@dataclass
class LiveSymbolismEngine(_BaseLiveJSONEngine):
    model: str = "openai:gpt-4o"
    temperature: float = 0.6
    timeout: int = 60

    def plan(
        self,
        *,
        scene_data: str,
        showrunner_plan: dict[str, object],
        state: SceneState,
    ) -> SymbolismPlan:
        return self.invoke_json(
            model_cls=SymbolismPlan,
            system_prompt=build_symbolism_system_prompt(),
            user_prompt=build_symbolism_user_prompt(scene_data, showrunner_plan),
        )


class MockCharacterEngine:
    def __init__(self) -> None:
        self._script = {
            "阮宁": [
                AgentAction(
                    observation_analysis="裴崧连一句安抚性的废话都没有，说明他已经把自己缩成流程接口，只认曲线不认人。",
                    emotional_shift="胃里像被空针头轻轻刮了一下，她明明困得发冷，却被债务提醒顶得太阳穴发胀。",
                    hidden_agenda="先逼出一个能谈价的缝，再把账户风险压到今晚之后。",
                    micro_expression="她说话前先把袖口往下一扯，拇指在疤痕边缘停了一瞬，像在确认自己还没抖出来。",
                    action_and_dialogue="她靠近采样台，压低嗓子：‘我按时来了。昨晚那批悲伤样本，你们系统又压价了，是吧？’",
                ),
                AgentAction(
                    observation_analysis="他把‘流程’两个字说得太顺，像是在提醒她，自己连讨价还价都得借用系统的语法。",
                    emotional_shift="后颈接口一阵发热，她想把桌上的终端掀翻，最后只是把牙关咬得更紧。",
                    hidden_agenda="尽快让复核落地，哪怕丢掉一点体面，也不能让窗口在自己手里超时。",
                    micro_expression="她下巴抬了半寸，嘴角却没有跟上去，眼神只在那只裂了口的保温杯上扫了一下。",
                    action_and_dialogue="她把指尖按在冰凉台面上：‘要复核就快点。我后面还有一班清洗工，没空陪你们等系统心情。’",
                ),
                AgentAction(
                    observation_analysis="确认单上多出来的异常标签不是备注，是提醒她谁能决定她明天还能不能登录。",
                    emotional_shift="胸口那股要顶出来的火忽然塌下去，剩下一种更省力的麻木，像身体已经替她把屈辱消化过一遍。",
                    hidden_agenda="签字保账户，先把母亲的贴片额度保住，其余的账以后再算。",
                    micro_expression="她低头时喉结轻轻动了一下，拇指在发炎的接口周围压了压，像按住一阵想吐的反胃。",
                    action_and_dialogue="她把确认单拖到面前，签名写得很快：‘行，我签。异常标签你写清楚，别连我的账户一起埋进去。’",
                ),
            ],
            "裴崧": [
                AgentAction(
                    observation_analysis="阮宁先谈压价而不是哀求，说明她还有一点余量；真正危险的是这种还撑着格式感的底层样本。",
                    emotional_shift="他并不烦躁，只是眼底像被终端光线磨得更冷，连杯口的裂纹都懒得再转开。",
                    hidden_agenda="把她压回可结算区间，同时避免自己背上放行高波动样本的责任。",
                    micro_expression="他抬手调曲线时停了两秒，像在给系统、也给自己找一个足够干净的措辞。",
                    action_and_dialogue="他把她的样本图拉到半空，语气平得像播报天气：‘压价是市场波动。你昨晚的悲伤样本纯度不够，系统只按规则结算。’",
                ),
                AgentAction(
                    observation_analysis="她开始催时间，说明真正要命的不是尊严，是窗口背后的某笔账；这让她更容易被迫配合。",
                    emotional_shift="配额红线在终端角落闪了一下，他肩背没有动，手心却因为空调太冷而有点发干。",
                    hidden_agenda="让她在时限内自愿接受异常标签，好把自己的纪律风险留在阈值之下。",
                    micro_expression="他把保温杯往旁边推开半寸，指腹在杯沿那道细裂上蹭了一下，很快又收回。",
                    action_and_dialogue="他切到人工签注页：‘流程快慢取决于你的样本稳定性，不取决于你的排班。保持平静，对你更划算。’",
                ),
                AgentAction(
                    observation_analysis="她最后还是签了，说明系统判断是对的：多数人不是不能反抗，而是反抗成本太贵。",
                    emotional_shift="他说话时依旧平稳，只是那种熟悉的职业安全感里掺进一丝很快被压平的异物感。",
                    hidden_agenda="完成签注、切断个人责任，并让自己看起来依然只是中性的执行者。",
                    micro_expression="他把确认单推回去时动作标准得近乎礼貌，眼睛却没有真正落在她脸上。",
                    action_and_dialogue="他在终端补了一条备注：‘备注我会写明。至于冻结与否，由上层风控判断，不由我决定。’",
                ),
            ],
        }

    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentAction:
        round_index = state["turn_count"]
        scripted_turns = self._script[profile.name]
        return scripted_turns[min(round_index, len(scripted_turns) - 1)]


class MockShowrunnerEngine:
    def plan(
        self,
        *,
        scene_brief: str,
        world_context: str,
        characters: list[CharacterProfile],
        max_turns: int,
    ) -> ShowrunnerPlan:
        return ShowrunnerPlan(
            scene_brief=scene_brief,
            scene_purpose="把一场普通交易写成制度怎样迫使人主动交出体面的切片。",
            target_ending="阮宁为了保住账户和母亲的贴片额度签下带异常标签的确认单；裴崧维持了流程，表面无损，内里却留下一个极小的裂口。",
            core_conflict="阮宁试图把自己的痛苦包装成值得被善待的样本，裴崧只在乎情绪纯度、配额红线和风控责任。",
            hidden_foreshadowing="裴崧保温杯的杯沿有一道新裂纹，它不会被人讨论，但会在关键时刻被看见。",
            tone_guardrail="绝不和解，绝不煽情，只允许更窄的选择、更贵的沉默和更冷的流程。",
            forced_beats=[
                {
                    "round_index": 1,
                    "dramatic_function": "把交易从普通问询立刻推成生存倒计时。",
                    "forced_event": "阮宁的终端弹出催款警报，显示贴片额度宽限期只剩三分钟。",
                    "target_shift": "她必须更快地让渡体面；裴崧也意识到自己可以借时限施压。",
                },
                {
                    "round_index": 2,
                    "dramatic_function": "把个人冲突升级为制度红线。",
                    "forced_event": "人工复核被强制开启，裴崧的配额与纪律风险同时在终端右上角闪红。",
                    "target_shift": "双方都更不可能让步：一个怕失去账户，一个怕失去岗位。",
                },
                {
                    "round_index": 3,
                    "dramatic_function": "用带羞辱性的程序文件完成场景收束。",
                    "forced_event": "确认单附带异常标签：若拒绝签字，账户将在天亮前进入冻结序列。",
                    "target_shift": "阮宁被迫接受屈辱性的交易完成；裴崧完成任务，但裂纹作为伏笔留在场内。",
                },
            ],
        )


class MockSymbolismEngine:
    def plan(
        self,
        *,
        scene_data: str,
        showrunner_plan: dict[str, object],
        state: SceneState,
    ) -> SymbolismPlan:
        return SymbolismPlan(
            scene_subtext="真正被买卖的不是情绪，而是人为了继续活下去而主动修剪自我的能力。",
            imagery_cues=[
                {
                    "motif": "杯沿裂纹",
                    "sensory_surface": "细、白、几乎看不出的瓷裂，指腹一蹭才有轻微阻滞感",
                    "emotional_mapping": "看似完好的秩序内部已经出现微小但无法复原的断口",
                    "usage_instruction": "不要解释它象征什么，只让角色在说关键话之前或之后看见、碰到它。",
                },
                {
                    "motif": "发炎的后颈接口",
                    "sensory_surface": "发热、发胀、被衣领反复磨过之后带一点湿意",
                    "emotional_mapping": "屈辱被技术标准化后留在身体上的慢性伤口",
                    "usage_instruction": "把强烈情绪转译为触碰接口、按压、避开衣领，而不是直接写愤怒或恐惧。",
                },
                {
                    "motif": "灰蓝色滚动字幕",
                    "sensory_surface": "冷、旧、像一层退不掉的低烧贴着屏幕底部移动",
                    "emotional_mapping": "制度的声音总在场，即使没人抬头看它",
                    "usage_instruction": "把广播和字幕写成背景噪音，不要写成宏大叙述。",
                },
            ],
            gesture_rewrites=[
                "把“她害怕账户被冻结”改写成她签字前先去看终端右上角那格闪烁的冻结提示。",
                "把“他产生了一丝动摇”改写成他的手指在杯沿裂纹上停了一下，又若无其事地移开。",
                "把“她觉得屈辱”改写成她在签字时故意把名字写得很快，像在缩短自己暴露在台面上的时间。",
            ],
            forbidden_explicit_phrases=[
                "他心里想",
                "她感到很悲伤",
                "制度是残酷的",
                "两人都明白了彼此",
            ],
        )
