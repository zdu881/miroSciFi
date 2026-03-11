from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, get_args, get_origin

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .models import (
    AgentAction,
    CharacterProfile,
    ContinuitySummary,
    SceneState,
    ShowrunnerPlan,
    SymbolismPlan,
)
from .prompts import (
    build_character_system_prompt,
    build_character_user_prompt,
    build_continuity_system_prompt,
    build_continuity_user_prompt,
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
        state: SceneState,
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


class ContinuityEngine(Protocol):
    def summarize(self, *, scene_data: str, state: SceneState) -> ContinuitySummary:
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
        try:
            return parse_json_response(model_cls, raw_text)
        except Exception:
            repaired = self.repair_json(model_cls=model_cls, broken_text=raw_text)
            return parse_json_response(model_cls, repaired)

    def repair_json(self, *, model_cls: type[ModelT], broken_text: str) -> str:
        repair_response = self._chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一个 JSON 修复器。你的唯一任务是把用户给出的近似 JSON 文本修复成合法 JSON。\n"
                        "不要改写语义，不要补充解释，不要输出代码块，只输出修复后的 JSON 对象。\n"
                        f"{build_json_instruction(model_cls)}"
                    )
                ),
                HumanMessage(
                    content=(
                        "请修复下面这段不合法 JSON，使其能被标准 JSON 解析器读取：\n"
                        f"{broken_text}"
                    )
                ),
            ]
        )
        return normalize_content(getattr(repair_response, "content", repair_response))


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
        state: SceneState,
    ) -> ShowrunnerPlan:
        return self.invoke_json(
            model_cls=ShowrunnerPlan,
            system_prompt=build_showrunner_system_prompt(),
            user_prompt=build_showrunner_user_prompt(
                scene_brief=scene_brief,
                world_context=world_context,
                characters=characters,
                max_turns=max_turns,
                state=state,
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


@dataclass
class LiveContinuityEngine(_BaseLiveJSONEngine):
    model: str = "openai:gpt-4o"
    temperature: float = 0.3
    timeout: int = 60

    def summarize(self, *, scene_data: str, state: SceneState) -> ContinuitySummary:
        return self.invoke_json(
            model_cls=ContinuitySummary,
            system_prompt=build_continuity_system_prompt(),
            user_prompt=build_continuity_user_prompt(scene_data=scene_data, state=state),
        )


def build_mock_character_action(profile: CharacterProfile, state: SceneState) -> AgentAction | None:
    brief = state.get("scene_brief", "")
    location = state.get("current_location", "")
    is_miner = profile.name == "阮宁"
    is_navigator = profile.name == "周惟"
    is_cognitive_auditor = profile.name == "岑簌"

    if "装载窗口" in brief or "认知装载舱" in brief:
        if is_navigator:
            return AgentAction(
                observation_analysis="岑簌先看预算条再看自己，说明这份合同真正雇佣的不是人，而是自己脑子里还没被占满的那点空白。",
                emotional_shift="冷却胶贴在后颈时像一片薄冰，胃里却慢慢涌上来一股更旧的热意，像身体提前知道这次要丢掉什么。",
                hidden_agenda="把第一份导航包装进去，先拿到弧灯号底舱合同，让妹妹的赎买额度别再往下掉。",
                micro_expression="他把签字笔夹得很紧，视线在预算条和名字栏之间来回停了两次。",
                action_and_dialogue="他把腕骨压在读写台上：‘装吧。你们要的不是配合，是空间。我把能腾的先腾出来，合同别再往后拖。’",
                context_load_label="弧灯号底舱导航包·第一段",
                context_load_cost=180,
                evicted_memory_label="妹妹的乳名",
                evicted_memory_summary="旧港风太大，妹妹隔着防波堤喊他的那两个字，总被浪声吞掉半截。",
                evicted_memory_cost=120,
            )
        if is_cognitive_auditor:
            return AgentAction(
                observation_analysis="他连删掉什么都答得这样快，说明家属债务已经把他压到没资格挑记忆，只能挑还能换多少钱。",
                emotional_shift="白舱的无菌光让他眼周那层疲惫更薄，他没有表情，只觉得流程终于进入了自己熟悉的速度。",
                hidden_agenda="让周惟在第一轮就学会主动报出可删记忆，后面流程才会更顺。",
                micro_expression="他把清退条款翻到最后一页，指腹在‘自愿腾挪’那行字上停了一下。",
                action_and_dialogue="他平声提醒：‘导航包不会为你的人生留白。预算不足，就自己说一段能删的。系统只负责记录，不负责替你可惜。’",
            )

    if "失认宿舍" in brief or "低重力宿舍" in brief or "补丁包" in brief:
        if is_navigator:
            return AgentAction(
                observation_analysis="补丁包追到宿舍里来，说明刚才那次清退在系统眼里还远远不够，连睡一觉都得先腾空间。",
                emotional_shift="宿舍风口吹出来的冷气一阵一阵掠过耳后接口，他张了张嘴，舌尖却先碰到一小块空白。",
                hidden_agenda="把夜里的补丁装完，别让合同在启航前就被退回候补名单。",
                micro_expression="他盯着终端里那条妹妹发来的语音请求，指尖悬在回拨键上，迟迟没有按下去。",
                action_and_dialogue="他靠着床架回话：‘补丁发过来。我知道预算不够。把海那段拿走，别动别的——至少今晚别动。’",
                context_load_label="失稳补丁·夜间检核包",
                context_load_cost=120,
                evicted_memory_label="地球海潮的湿味",
                evicted_memory_summary="小时候跟父亲站在海堤上，盐雾把袖口一点点浸凉，他一直以为那股味道能陪自己更久。",
                evicted_memory_cost=120,
            )
        if is_cognitive_auditor:
            return AgentAction(
                observation_analysis="他没有再讨价还价，只要求别一次拿走太多，说明失认已经开始发生，只是他还想给自己留个可辨认的边。",
                emotional_shift="远程复核界面的黄灯稳定闪着，他的语气仍平，像在替一套自动程序补全最后一格人声。",
                hidden_agenda="把补丁推进去，别让周惟因为预算缺口在起飞前就掉出合规范围。",
                micro_expression="他把远程窗口缩小又放大，像在确认对方脸上那点迟钝是不是已经开始蔓延。",
                action_and_dialogue="他在链路里说：‘宿舍不是例外区。补丁今晚必须装完。你现在保住的每一段记忆，都会在明天的复核里变成理由。’",
            )

    if "语义压舱室" in brief or "异星外交语义包" in brief or "白色压舱室" in brief:
        if is_navigator:
            return AgentAction(
                observation_analysis="这次要删的不是称呼，而是某种更软的感觉，说明系统真正嫌占地方的，从来不是事实，而是人还能被什么打动。",
                emotional_shift="冷白色读写灯照得他牙根发酸，胸口像被什么东西轻轻掏空，又没有留下真正的伤口。",
                hidden_agenda="把语义包装进去，保住出航资格；至于那些更难命名的东西，先让它们消失。",
                micro_expression="他把下唇咬出一条很淡的白痕，手却稳稳按在确认片上。",
                action_and_dialogue="他看着岑簌：‘把那段也拿走。反正你们要的不是我会不会想人，是我会不会把外星语说准。’",
                context_load_label="异星外交语义包·第一类称呼",
                context_load_cost=140,
                evicted_memory_label="第一次真正笑出来的感觉",
                evicted_memory_summary="十六岁那年，有人从气闸口后面推了他一把，他踉跄着回头时，胸腔忽然轻得像真空外也有风。",
                evicted_memory_cost=140,
            )
        if is_cognitive_auditor:
            return AgentAction(
                observation_analysis="周惟已经学会自己说出该删什么，这说明训练正在成功：人格会主动替制度让位。",
                emotional_shift="他念条款时喉咙略干，像白舱里每一口空气都被过滤得过于干净，只剩规程的味道。",
                hidden_agenda="完成跨物种语义装载审批，并把所有可能被定义成私人残留的空隙提前堵死。",
                micro_expression="他把白手套的指尖在面板边缘轻轻抹了一下，像在擦去一粒并不存在的灰。",
                action_and_dialogue="他把压舱协议推过去：‘外环不会为你的私人感受减轻词汇密度。要么删，要么换人。你比我清楚候补名单有多长。’",
                context_load_label="跨物种责任豁免条款",
                context_load_cost=110,
                evicted_memory_label="第一次签发清退令后的反胃",
                evicted_memory_summary="打印纸从机器里吐出来时，他去洗手间吐过一次。后来这件事慢慢只剩下制度要求的那半句说明。",
                evicted_memory_cost=110,
            )

    if "白舱回访" in brief or "人格断片" in brief or "人工问询" in brief:
        if is_navigator:
            return AgentAction(
                observation_analysis="岑簌问的不是他还记不记得家人，而是他还能不能被稳定读取，说明自己已经更像一件即将出厂的部件。",
                emotional_shift="回访席的白光把他的眼底照得发麻，他明明坐着，却总觉得身体里有一块地方对不上号。",
                hidden_agenda="别在回访席上被判成失稳样本，至少先把起飞前最后一道门过掉。",
                micro_expression="他在回答前先摸了摸自己的胸牌，像要确认上面的名字还和自己有关。",
                action_and_dialogue="他看着录入灯：‘我不是故障。我只是有些词出来得慢一点。你们删掉的东西太多了，剩下的总要排队。’",
            )
        if is_cognitive_auditor:
            return AgentAction(
                observation_analysis="他已经开始靠摸胸牌确认自己，这种轻微失认正是最危险也最方便被归档的阶段。",
                emotional_shift="他把问询节奏放得更匀，心口却像被某个久远而模糊的家庭场景轻轻顶了一下，很快又没了。",
                hidden_agenda="让周惟在记录里显得‘仍可使用’，同时把责任边界写得足够干净。",
                micro_expression="他把回访问卷往前推了一厘米，眼神却没有真正落到周惟脸上。",
                action_and_dialogue="他平声说：‘只回答是否。你还能否在不调取私人残留的前提下完成盲跳前测试？系统要的是稳定，不是你的解释。’",
            )

    if "盲跳许可" in brief or "终航星图" in brief or "最终盲跳" in brief:
        if is_navigator:
            return AgentAction(
                observation_analysis="终航星图要占掉的那部分，已经逼到只剩最后一段能把自己叫回来的东西；合同真正要求的不是技术，而是愿不愿意把‘我’让到最后。",
                emotional_shift="许可闸门边的冷却雾贴着手背爬上来，他忽然很想记住点什么，却一时找不到该从哪里抓起。",
                hidden_agenda="拿到盲跳许可，让弧灯号起飞，也让赎买额度真的打到家属账户。至于自己会剩成什么样，先别算。",
                micro_expression="他盯着确认屏上的本名看了几秒，像那两个字已经开始和自己分家。",
                action_and_dialogue="他把手按上最终授权区：‘装终航星图。剩下那段也拿走。飞船得走，钱得到账，别让我在最后一步停下来。’",
                context_load_label="终航星图与失稳补丁",
                context_load_cost=180,
                evicted_memory_label="母亲年轻时的脸",
                evicted_memory_summary="旧货码头的吊机灯从她脸侧擦过去，眼尾那几道细纹像被海风压得很浅，他一直以为自己还能记得更久。",
                evicted_memory_cost=180,
            )
        if is_cognitive_auditor:
            return AgentAction(
                observation_analysis="周惟已经不再追问删掉之后还剩什么，这说明制度终于把人训练到了最省事的阶段。",
                emotional_shift="他念最终条款时嗓子发紧了一瞬，像有个旧名字在喉咙里碰了一下，又很快沉下去。",
                hidden_agenda="在责任追溯前完成签发，把飞船和自己都推离这间白舱。",
                micro_expression="他把许可章悬在面板上空半秒，随即很稳地按了下去。",
                action_and_dialogue="他低声确认：‘许可签发后，任何被清退内容都不再享有追索权。弧灯号会起跳，家属额度会结算，剩下的空白由你个人承担。’",
                context_load_label="盲跳事故责任矩阵",
                context_load_cost=150,
                evicted_memory_label="前妻最后一次叫他本名",
                evicted_memory_summary="狭窄走廊里，她的声音被回风口切成两半，还是能听出疲惫。后来那两个字越来越像一条与自己无关的旧记录。",
                evicted_memory_cost=150,
            )

    if "续租窗口" in brief or "深睡期" in brief:
        if is_miner:
            return AgentAction(
                observation_analysis="审核台上的人先看风险阈值再看人，说明这笔续租能不能过，取决于系统愿不愿承认她还剩一段完整睡眠。",
                emotional_shift="她已经熬到耳后发木，眼球像被细沙磨过，提到母亲的供氧表时胃里又绷了一下。",
                hidden_agenda="先把夜间供氧续上，哪怕把最后一段深睡期也押出去。",
                micro_expression="她把掌心压在窗口边缘，指腹无意识地蹭着那张快起毛的疗养仓缴费单。",
                action_and_dialogue="她把续租申请推过去：‘我能抵押的不是浅睡噪声，是整段深睡。你给我把疗养仓今晚的氧雾先续上。’",
            )
        return AgentAction(
            observation_analysis="她一开口就拿完整深睡期换供氧，说明她已经没有别的缓冲资产，只剩还能被切割的身体连续性。",
            emotional_shift="终端蓝光把他脸上的疲惫压得更薄，他没有困，只是习惯性地把人的夜晚读成一串可变现字段。",
            hidden_agenda="把她压进可续租但高违约的区间，让系统风险留在她身上，不落到自己工位上。",
            micro_expression="他把保温杯往手边挪了半寸，视线停在梦层波形图最深的那道下陷。",
            action_and_dialogue="他把她的梦层曲线拉开：‘完整深睡的价格今晚下调。要续供氧可以，但你得接受系统对连续睡眠权的冻结条款。’",
        )

    if "电梯井" in brief or "追保" in brief or "公共走廊" in location:
        if is_miner:
            return AgentAction(
                observation_analysis="他隔着远程追保频道还在追问稳定性证明，说明窗口签过的字根本不算数，真正的催款现在才开始。",
                emotional_shift="电梯井的回声把她的太阳穴敲得更空，母亲疗养仓的低鸣像贴着她肋骨喘气。",
                hidden_agenda="拖住降级时限，把供氧先保到天亮之后。",
                micro_expression="她用肩膀抵住剥落漆皮的墙面，眼睛却一直盯着腕机上那格灰下去的供氧条。",
                action_and_dialogue="她压低声音回话：‘稳定性证明我会补，但你先别让系统把夜间供氧往下砍。我妈撑不到你们白天班。’",
            )
        return AgentAction(
            observation_analysis="她没有否认违约风险，只求拖时间，说明疗养仓那头已经亮起更紧的告警。",
            emotional_shift="他隔着远程频道听见背景里的电梯噪音，语气仍平，手指却把催告页往下划得更快。",
            hidden_agenda="逼她在天亮前交齐证明或补押更多睡眠切片。",
            micro_expression="他看着追保界面的黄灯闪烁，嘴角几乎没有动，只把备注栏多开了一列。",
            action_and_dialogue="他在频道里说：‘系统不会等你的家庭状况。若补件失败，续租额度会回退，你现在最好决定还要不要再押一层梦。’",
        )

    if "拆梦工位" in brief or "黑市" in brief or "地下" in brief:
        if is_miner:
            return AgentAction(
                observation_analysis="黑市只收还能回到事故前夕的完整梦，这说明平台之外的人也知道，越完整的自我越值钱。",
                emotional_shift="她困得手腕发轻，父亲出事前一晚那点潮热的家常气味刚碰到嗓子，就被她硬咽下去。",
                hidden_agenda="把差额补齐，不让官方系统拿‘私藏梦层’直接掐死续租。",
                micro_expression="她把采样贴按在耳后时停了一秒，像在给某段快被切开的夜晚让路。",
                action_and_dialogue="她盯着拆梦台的玻璃罩：‘只切到事故前，后面的别碰。我要拿这段去补押，不是把整个人交给你们。’",
            )
        return AgentAction(
            observation_analysis="她连在黑市都还想保留一小段不被切开的梦，说明她真正舍不得的已经不是钱，是还能自证为人的那点连续性。",
            emotional_shift="黑市回传界面跳得很慢，他眼底那层职业性的平整感里掺进一丝极淡的不耐。",
            hidden_agenda="确认她是否私藏未备案梦层，并把所有风险节点固定到报告里。",
            micro_expression="他把抽查授权页翻到最后一屏，指节在桌面轻轻敲了一下。",
            action_and_dialogue="他通过抽查链路发去问询：‘私留梦层会直接触发违约。你如果还想保住续租资格，最好现在把未备案部分一并上传。’",
        )

    if "回访席" in brief or "问询" in brief:
        if is_miner:
            return AgentAction(
                observation_analysis="他现在问的不是昨夜发生了什么，而是她还能不能被系统稳定读取，这意味着她已经被看成一件故障中的设备。",
                emotional_shift="白昼把她脸上的夜色全照出来了，眼皮每抬一次都像有细线把后脑往回拽。",
                hidden_agenda="别让问询结果把自己直接归到不可续租的高噪样本。",
                micro_expression="她把一次性纸杯捏得稍微变了形，像想靠那点脆响把自己从断片里拽回来。",
                action_and_dialogue="她看着桌面录入灯：‘我没失控，只是没睡。你们把人的梦切成这样，再来问为什么会断片，不觉得太省事了吗？’",
            )
        return AgentAction(
            observation_analysis="她已经开始出现梦游性断片，却还保留着足够完整的愤怒，这类样本最容易在记录里留下不稳定尾巴。",
            emotional_shift="他把问询节奏放得更匀，像在用流程本身替自己挡掉她话里的针。",
            hidden_agenda="让她在问询记录里亲口承认自己适合被进一步切分和监管。",
            micro_expression="他把录音指示灯调亮了一格，视线却始终没有真正停在她脸上。",
            action_and_dialogue="他按着表单往下问：‘是否愿意接受更高频次的梦层监管？你只需要回答是或否，不必解释制度。’",
        )

    if "关灯协议" in brief or "完整梦的权利" in brief or "最终签注" in brief:
        if is_miner:
            return AgentAction(
                observation_analysis="最后能押的只剩母亲死后自己还能做一次完整梦的权利，这说明制度已经把她逼到连未来的哀悼都要先签走。",
                emotional_shift="她困得发冷，胸口却像塞了一块没化开的热铁，连呼吸都带着磨损感。",
                hidden_agenda="先把疗养仓和供氧续上，让母亲今晚别断。至于梦，等以后再说——如果以后还剩一点。",
                micro_expression="她把签注笔拿起来又放下，拇指在指节边缘慢慢磨出一层发白的印。",
                action_and_dialogue="她盯着协议最后一栏：‘签完这个，我以后连梦见她一次都得算违约，是吧？那你写清楚，再把供氧给我续到明晚。’",
            )
        return AgentAction(
            observation_analysis="她问的不是价钱，而是违约定义，说明她知道自己已经没有讨价还价的空间，只剩确认损失边界。",
            emotional_shift="他听见自己把条款念得很稳，稳得像那只裂了口的杯子一直没有继续裂下去。",
            hidden_agenda="完成最终签注，把人和责任一并推回系统流程里。",
            micro_expression="他念条款时拇指在杯沿细裂上停了一瞬，又像什么都没碰到一样松开。",
            action_and_dialogue="他把终审页推向她：‘协议生效后，你的连续梦权会被平台托管。疗养仓和夜间供氧可以续到明晚，但你不能再私自保留完整梦层。’",
        )

    return None


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
        contextual = build_mock_character_action(profile, state)
        if contextual is not None:
            return contextual
        round_index = state["turn_count"]
        scripted_turns = self._script.get(profile.name)
        if scripted_turns:
            return scripted_turns[min(round_index, len(scripted_turns) - 1)]
        return AgentAction(
            observation_analysis="眼前流程不会为任何私人理由停下，自己只能在更窄的缝里继续选择。",
            emotional_shift="身体先一步绷紧，像在等一张迟早会落下来的通知。",
            hidden_agenda="把当前损失推迟到下一步，而不是现在就彻底失去主动。",
            micro_expression="他把视线从终端边角挪开时停了一瞬，又迅速恢复成平稳的样子。",
            action_and_dialogue="他把条件往现实的一侧推了一点：‘先把这一格流程走完，剩下的代价以后再算。’",
        )


def _infer_mock_opening(scene_brief: str, state: SceneState) -> tuple[str, str, str, str]:
    previous_time = state.get("time_marker") or "凌晨四点十二分"
    previous_location = state.get("current_location") or "雾港第七码头的情绪采样站担保窗口"
    if any(token in scene_brief for token in ["装载窗口", "失认宿舍", "语义压舱室", "白舱回访", "盲跳许可", "上下文"]):
        context_keyword_map = [
            (
                ("装载窗口", "认知装载舱"),
                (
                    "retain",
                    "这是认知清退制度的开场，人物必须先被钉在预算条和白舱读写灯之间。",
                    "月轨标准时 03:16",
                    "月轨远航局 A-12 认知装载舱",
                ),
            ),
            (
                ("失认宿舍", "低重力宿舍", "补丁包"),
                (
                    "shift",
                    "让装载后的空白先回到私人空间里发作，再由制度追进去继续逼迫。",
                    "月轨标准时 04:02",
                    "弧灯号候补宿舍的低重力睡舱",
                ),
            ),
            (
                ("语义压舱室", "异星外交语义包", "白色压舱室"),
                (
                    "shift",
                    "故事进入更干净、更白、更像手术室的制度空间，强调人格如何给语言让位。",
                    "月轨标准时 18:41",
                    "远航局白色语义压舱室",
                ),
            ),
            (
                ("白舱回访", "人格断片", "人工问询"),
                (
                    "shift",
                    "让装载后果在启航前的回访席上显形，证明失去不会自动变成稳定。",
                    "月轨标准时 21:08",
                    "远航局白舱回访席",
                ),
            ),
            (
                ("盲跳许可", "终航星图", "最终盲跳"),
                (
                    "shift",
                    "所有代价要在正式许可里结算，冷的结尾必须发生在起飞门槛上。",
                    "月轨标准时 23:46",
                    "弧灯号盲跳许可闸门",
                ),
            ),
        ]
        for keywords, result in context_keyword_map:
            if any(token in scene_brief for token in keywords):
                return result
    if not state.get("chapter_history"):
        return (
            "retain",
            "这是小说开场，时间与空间自然从同一窗口起步。",
            "凌晨四点十二分",
            "雾港第七码头的情绪采样站担保窗口",
        )

    keyword_map = [
        (
            ("拆梦工位", "黑市", "下城", "地下", "回收站"),
            (
                "shift",
                "剧情离开官方窗口，转入更脏的地下交换链条。",
                "次日零点前十七分",
                "下城拆梦工位外的回收站后厅",
            ),
        ),
        (
            ("回访席", "问询", "治理局"),
            (
                "shift",
                "故事从黑市或居住空间切回白昼问询场，压力变成更干净的制度语言。",
                "次日中午十一点二十六分",
                "雾港治理局的白昼回访席",
            ),
        ),
        (
            ("关灯协议", "最终签注", "完整梦的权利"),
            (
                "shift",
                "所有线头回到最终执行窗口，收束必须发生在正式流程里。",
                "次日深夜二十三点四十一分",
                "第七码头的关灯协议签注室",
            ),
        ),
        (
            ("蜂巢公寓", "家属", "疗养仓", "电梯井", "公共走廊", "追保"),
            (
                "shift",
                "上一场的制度交易结束后，压力回流到了私人居住空间。",
                "清晨五点零七分",
                "蜂巢公寓 C 栋十七层的公共走廊",
            ),
        ),
        (
            ("复核", "采样站", "窗口"),
            (
                "retain",
                "空间没有真正松开，角色仍被卡在同一制度接口附近。",
                previous_time,
                previous_location,
            ),
        ),
    ]
    for keywords, result in keyword_map:
        if any(token in scene_brief for token in keywords):
            return result
    return (
        "retain",
        "上一场残留的空气还没散，故事继续贴着原有时空流推进。",
        previous_time,
        previous_location,
    )


def _build_mock_beats(scene_brief: str, max_turns: int) -> list[dict[str, str | int]]:
    if any(token in scene_brief for token in ["装载", "失认", "语义", "盲跳", "上下文", "记忆", "清退"]):
        beat_templates = [
            (
                "先把预算推成超限",
                "系统把新的装载包压进来，明确告诉角色：不删就无法继续。（围绕：{scene_brief}）",
                "角色必须自己选出一段该被挪走的私人部分。",
            ),
            (
                "把私人记忆升级为合规问题",
                "上级流程介入，要求角色证明自己没有私藏任何不该留下的情感残留。（围绕：{scene_brief}）",
                "所有回避都会被写进清退记录，关系只能更冷。",
            ),
            (
                "用许可或签章完成不可逆收束",
                "系统要求最终确认、签章或起跳前上传，已删掉的内容不再享有追索权。（围绕：{scene_brief}）",
                "角色必须承认：继续向前的唯一方式，就是接受某段过去再也回不来。",
            ),
        ]
    else:
        beat_templates = [
            (
                "先把局面推成倒计时",
                "系统弹出新的时限提醒，明确告诉角色他们正在失去最后的缓冲层。",
                "所有说话都必须更快、更短、更伤体面。",
            ),
            (
                "把私人处境升级为制度红线",
                "上级流程或风险指标介入，证明这已经不是两个人之间的谈判。",
                "双方都更不可能退让，只能把代价向对方或向自己身上继续压。",
            ),
            (
                "用文书或程序完成一次不可逆收束",
                "系统要求现场确认、签注、上传或交割某项不能撤回的内容。",
                "角色必须丢掉某样东西，才能把眼前这一段时间暂时续上。",
            ),
        ]
    beats: list[dict[str, str | int]] = []
    for index in range(1, max_turns + 1):
        dramatic_function, forced_event, target_shift = beat_templates[
            min(index - 1, len(beat_templates) - 1)
        ]
        beats.append(
            {
                "round_index": index,
                "dramatic_function": dramatic_function,
                "forced_event": forced_event.format(scene_brief=scene_brief) if "{scene_brief}" in forced_event else f"{forced_event}（围绕：{scene_brief}）",
                "target_shift": target_shift,
            }
        )
    return beats


class MockShowrunnerEngine:
    def plan(
        self,
        *,
        scene_brief: str,
        world_context: str,
        characters: list[CharacterProfile],
        max_turns: int,
        state: SceneState,
    ) -> ShowrunnerPlan:
        continuity_mode, rationale, opening_time, opening_location = _infer_mock_opening(
            scene_brief,
            state,
        )
        if any(token in scene_brief for token in ["装载", "失认", "语义", "盲跳", "上下文", "记忆", "清退"]):
            return ShowrunnerPlan(
                scene_brief=scene_brief,
                scene_purpose="把一场技术流程写成制度怎样逼着人主动删掉自己，才能继续往前移动。",
                target_ending="至少一方为了装下新的知识包或签出新的许可，亲手放弃了一段无法恢复的私人记忆。",
                core_conflict="周惟需要用人格连续性换取远航合同；岑簌则要用更冷的合规把这些失去包装成可执行流程。",
                hidden_foreshadowing="某块白色读写灯、一张失焦的名牌或一段突然说不出的称呼，会反复提醒人物：空白正在扩张。",
                tone_guardrail="不要写伟大牺牲，不要写热血飞船，只写预算条、白舱和清退记录怎样一点点吞掉‘我是谁’。",
                continuity_mode=continuity_mode,
                continuity_rationale=rationale,
                opening_time_marker=opening_time,
                opening_location=opening_location,
                forced_beats=_build_mock_beats(scene_brief, max_turns),
            )
        return ShowrunnerPlan(
            scene_brief=scene_brief,
            scene_purpose="把场景写成角色在制度压力下继续折损自我，而不是重新开一个开场。",
            target_ending="至少有一个人为了活下去或保住岗位，交出更难再拿回来的东西。",
            core_conflict="一方想保住最低限度的生存，一方想保住自己在体系里的安全位置；两者都比善意更硬。",
            hidden_foreshadowing="某个不起眼的器物或身体细节会留下细小裂口，作为后续场景的隐秘证据。",
            tone_guardrail="绝不和解，绝不煽情，只允许更窄的选择、更贵的沉默和更冷的流程。",
            continuity_mode=continuity_mode,
            continuity_rationale=rationale,
            opening_time_marker=opening_time,
            opening_location=opening_location,
            forced_beats=_build_mock_beats(scene_brief, max_turns),
        )


class MockSymbolismEngine:
    def plan(
        self,
        *,
        scene_data: str,
        showrunner_plan: dict[str, object],
        state: SceneState,
    ) -> SymbolismPlan:
        if state.get("cognition_mode") == "eviction_budget":
            return SymbolismPlan(
                scene_subtext="真正被压缩的不是信息，而是人格连续性：系统要求人先删掉自己，才能让技术顺利通过。",
                imagery_cues=[
                    {
                        "motif": "白色读写灯",
                        "sensory_surface": "冷、匀、没有情绪，亮起时像把人的脸削平一层",
                        "emotional_mapping": "制度对人格的擦除没有暴力姿势，只有稳定而持续的照明",
                        "usage_instruction": "让关键决定总发生在读写灯亮起或熄灭的瞬间，不要解释其意义。",
                    },
                    {
                        "motif": "失焦的名牌",
                        "sensory_surface": "塑料边缘被手指磨得发亮，字却像每次看都更陌生一点",
                        "emotional_mapping": "一个人仍在履约，却越来越不像自己的名字所属者",
                        "usage_instruction": "把人物确认自我时的动作，写成摸胸牌、盯编号、核对签名。",
                    },
                    {
                        "motif": "后颈冷却胶",
                        "sensory_surface": "薄、凉、带一点药味，贴久了像神经表面结了一层霜",
                        "emotional_mapping": "技术改造留下的剥夺感被包进日常护理用品里",
                        "usage_instruction": "把强烈情绪写成角色下意识去按、揭、抚平冷却胶，而不是直接陈述悲伤。",
                    },
                ],
                gesture_rewrites=[
                    "把‘他忘记了亲人的名字’改写成他看着语音请求上的称呼栏，却迟迟想不起该怎么读出来。",
                    "把‘她/他很害怕失去自我’改写成他先去摸胸牌或签名处，确认那两个字还和自己有关。",
                    "把‘审核官也动摇了’改写成他念条款时喉咙紧了一瞬，像有个旧名字在里面碰了一下。",
                ],
                forbidden_explicit_phrases=[
                    "他心里想",
                    "他突然想起自己已经不是自己",
                    "制度很残酷",
                    "他们都被震撼了",
                ],
            )
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
                "把‘她害怕账户被冻结’改写成她签字前先去看终端右上角那格闪烁的冻结提示。",
                "把‘他产生了一丝动摇’改写成他的手指在杯沿裂纹上停了一下，又若无其事地移开。",
                "把‘她觉得屈辱’改写成她在签字时故意把名字写得很快，像在缩短自己暴露在台面上的时间。",
            ],
            forbidden_explicit_phrases=[
                "他心里想",
                "她感到很悲伤",
                "制度是残酷的",
                "两人都明白了彼此",
            ],
        )


def _infer_mock_ending(scene_brief: str, state: SceneState) -> tuple[str, str, str, str]:
    opening_time = state.get("time_marker") or "凌晨四点十二分"
    opening_location = state.get("current_location") or "雾港第七码头的情绪采样站担保窗口"
    if any(token in scene_brief for token in ["装载窗口", "失认宿舍", "语义压舱室", "白舱回访", "盲跳许可", "上下文"]):
        if "装载窗口" in scene_brief or "认知装载舱" in scene_brief:
            return (opening_time, opening_location, "月轨标准时 03:49", "A-12 装载舱外的消毒走廊")
        if "失认宿舍" in scene_brief or "补丁包" in scene_brief:
            return (opening_time, opening_location, "月轨标准时 04:28", "弧灯号候补宿舍的公共洗漱间")
        if "语义压舱室" in scene_brief or "异星外交语义包" in scene_brief:
            return (opening_time, opening_location, "月轨标准时 19:17", "白色语义压舱室外的编号廊桥")
        if "白舱回访" in scene_brief or "人格断片" in scene_brief:
            return (opening_time, opening_location, "月轨标准时 21:44", "远航局白舱回访席外的静压通道")
        if "盲跳许可" in scene_brief or "终航星图" in scene_brief:
            return (opening_time, opening_location, "月轨标准时 23:58", "弧灯号盲跳许可闸门内侧")
    if not state.get("chapter_history"):
        return (
            opening_time,
            opening_location,
            "清晨四点二十九分",
            "第七码头外环连桥的安检口",
        )
    if "关灯协议" in scene_brief or "最终签注" in scene_brief or "完整梦的权利" in scene_brief:
        return (
            opening_time,
            opening_location,
            "次日零点后六分",
            "第七码头关灯协议签注室外的安检走廊",
        )
    if "蜂巢公寓" in scene_brief or "疗养仓" in scene_brief:
        return (
            opening_time,
            opening_location,
            "清晨五点四十八分",
            "蜂巢公寓 C 栋十七层的低配疗养仓门外",
        )
    if "地下" in scene_brief or "回收站" in scene_brief or "黑市" in scene_brief:
        return (
            opening_time,
            opening_location,
            "次日零点十三分",
            "下城黑市回收站后门的潮湿巷口",
        )
    if "天亮前" in scene_brief or "最终确认窗口" in scene_brief or "静音" in scene_brief:
        return (
            opening_time,
            opening_location,
            "清晨五点五十九分",
            "雾港第七码头最终确认窗口外的安检闸门",
        )
    return (
        opening_time,
        opening_location,
        opening_time,
        opening_location,
    )


class MockContinuityEngine:
    def summarize(self, *, scene_data: str, state: SceneState) -> ContinuitySummary:
        opening_time, opening_location, ending_time, ending_location = _infer_mock_ending(
            state["scene_brief"],
            state,
        )
        carryover_threads = [
            state["showrunner_plan"].get("hidden_foreshadowing", "某个细小裂口还留着。"),
            state["showrunner_plan"].get("core_conflict", "制度冲突没有结束，只是换了位置继续。"),
            state["showrunner_plan"].get("target_ending", "本场的代价会在下一场继续计息。"),
        ]
        carryover_threads = [item for item in carryover_threads if item][:3]
        resolved_threads = [
            f"本场围绕“{state['scene_brief']}”的即时手续已被迫推进完毕。"
        ]
        continuity_decision = "retain" if ending_location == opening_location else "shift"
        continuity_reason = (
            "人物仍滞留在同一制度接口附近，下一场可以直接接续。"
            if continuity_decision == "retain"
            else "本场把代价转移到了另一个空间，下一场更适合明确换场。"
        )
        return ContinuitySummary(
            scene_brief=state["scene_brief"],
            chapter_summary=(
                f"{state['showrunner_plan'].get('target_ending', '本场完成了一次带损耗的推进。')}"
                f" 代价没有被清算，只是从 {opening_location} 带到了 {ending_location}。"
            ),
            opening_time_marker=opening_time,
            opening_location=opening_location,
            ending_time_marker=ending_time,
            ending_location=ending_location,
            continuity_decision=continuity_decision,
            continuity_reason=continuity_reason,
            irreversible_change="至少一项体面、权利、岗位安全感或情绪所有权已经被正式折损，无法完整回收。",
            carryover_threads=carryover_threads,
            resolved_threads=resolved_threads,
            next_scene_pressure=("系统不会归还被清退的内容；下一场只会让新的装载和更大的空白继续叠上去。" if state.get("cognition_mode") == "eviction_budget" else "系统不会撤回本场后果；下一场只会让它们在新的地点或更晚的时间里继续发作。"),
        )
