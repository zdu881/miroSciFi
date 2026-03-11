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
    is_earth = profile.name == "沈竟"
    is_jupiter = profile.name == "韩漠"

    if "四十分钟旧闻" in brief or "主权协调厅" in brief or "旧闻" in brief:
        if is_earth:
            return AgentAction(
                observation_analysis="他眼前这段港务局画面已经晚了四十分钟，这意味着自己现在做的不是指挥现场，而是在替一个已经过去的局面补写法统。",
                emotional_shift="太阳穴被同步厅的冷光顶得发紧，他明明坐着，却像一直悬在一段来不及落地的命令上。",
                hidden_agenda="先把地球的合法性和接管姿态发出去，即使它抵达时现场早已换了主人。",
                micro_expression="他盯着延迟图看了两秒，手指在发送键上方停住，又像怕别人看出迟疑一样压了下去。",
                action_and_dialogue="他把接管草案推上主屏：‘先发紧急维稳令，要求木卫三港务局冻结装卸、保护结算节点，并向地球回传完整伤亡名录。’",
                transmission_target="韩漠",
                transmission_content="地球主权协调局下达紧急维稳令：冻结港口、等待地球接管小组、不得擅自改动结算和安保权限。",
            )
        if is_jupiter:
            return AgentAction(
                observation_analysis="地球那边还没看见现在的码头，他真正拥有的是这一段没人能及时纠正的现时。",
                emotional_shift="舱壁传来的低鸣让他胸口更稳，不是轻松，而是那种终于比中心先活在现实里的冷静。",
                hidden_agenda="在地球回波抵达前把港口调度权和本地治安权收拢成既成事实。",
                micro_expression="他把港口舱图放大时眼皮几乎没动，只在民兵驻点那一格停得稍微久了一点。",
                action_and_dialogue="他对值班组说：‘先封锁外环装卸臂，把结算主机迁进民兵看守区。地球要发话，等它的话飞到这里再说。’",
                transmission_target="沈竟",
                transmission_content="木卫三局势已进入本地紧急状态。我将先行接管港口调度与结算主机，稍后补发说明。",
            )

    if "镇压指令在途" in brief or "接管命令" in brief or "盲区" in brief:
        if is_earth:
            return AgentAction(
                observation_analysis="命令已经在路上，自己能做的却只是继续往一段旧现实上加注，这让主权越来越像一种延迟播出的语音。",
                emotional_shift="他喉咙里有点发干，像刚吞下一口还没来得及发热的金属。",
                hidden_agenda="用更重的条款把地球的权力感补回去，至少让回波抵达时还能像命令。",
                micro_expression="他把第二版文书里的‘临时军事接管’几个字加粗，又迅速把手从屏幕上收开。",
                action_and_dialogue="他追加签注：‘若木卫三未按令冻结港口，则视为对地球法统的拒绝执行，后续一切结算免责由本地主官承担。’",
                transmission_target="韩漠",
                transmission_content="追加指令：若你拒绝冻结港口并等待接管，即构成对地球法统的公开违逆，相关责任由你本人承担。",
            )
        if is_jupiter:
            return AgentAction(
                observation_analysis="来自地球的沉默本身就是窗口：越久没有回波，本地工会和民兵越相信必须先听见自己人的命令。",
                emotional_shift="他不觉得热，只有一种被真空磨薄后的清醒，像每个决定都在替自己提前切断退路。",
                hidden_agenda="趁地球命令仍在路上，把港口、配给和保险切换到本地规程。",
                micro_expression="他把港务章印在三份纸质授权上盖下去时，手腕稳得像练过一样。",
                action_and_dialogue="他对工会代表说：‘今天开始，木卫三的补给优先序列由本地议事厅签字。地球的命令如果晚到，只能算意见。’",
                transmission_target="沈竟",
                transmission_content="木卫三已按本地紧急条例接管港口与补给顺位。等待地球批准会先让人饿死。",
            )

    if "五秒期货" in brief or "中继站" in brief or "保险价格" in brief:
        if is_earth:
            return AgentAction(
                observation_analysis="市场总比公文更早收到未来，说明自己维护的不是现实，而是被交易系统不断提前榨干的合法性尾声。",
                emotional_shift="他盯着保险曲线往上拱时，眼底那点疲惫像被突然擦亮，变成更难看的冷意。",
                hidden_agenda="先冻结木星圈相关结算，别让价格走势替木卫三宣布独立已成事实。",
                micro_expression="他把手背压在桌沿，像在用那点硬度把自己钉回仍可命令的位置。",
                action_and_dialogue="他对金融联络官说：‘先暂停木星航道保险的地球清算通道，别让价格比法令更早给出政治答案。’",
                transmission_target="韩漠",
                transmission_content="地球将暂停木星圈部分清算通道。任何试图利用价格波动固化本地接管的行为，都将被视作叛离法统。",
            )
        if is_jupiter:
            return AgentAction(
                observation_analysis="当氧气、保险和航道保证金先于法令涨跌时，真正宣布现实的人就不再是部长，而是更靠近信号源的报价器。",
                emotional_shift="他看着中继站提前回来的价格，胃里反而轻了一点，像终于有东西比地球的回波更愿意站在自己这边。",
                hidden_agenda="把市场对既成事实的押注继续做大，让地球即使想撤回也买不起。",
                micro_expression="他把一份本地结算备忘录折起一角，塞进工会代表手里时动作很小，像只是在递一张过时的收据。",
                action_and_dialogue="他低声说：‘把本地结算折扣放出去。谁先按木卫三规程成交，谁就先收到明天。’",
                transmission_target="沈竟",
                transmission_content="你们的法令还在路上，但市场已经开始按木卫三的明天定价。",
            )

    if "回波戒严" in brief or "戒严" in brief:
        if is_earth:
            return AgentAction(
                observation_analysis="当镇压令终于抵达，却只成为现场众多声音里最晚的一条，法统被证明首先输给了传播速度。",
                emotional_shift="他肩背没有垮，只是那种长期训练出的官样平稳里忽然掺进一丝被迫承认迟到的钝痛。",
                hidden_agenda="即使抢不回现场，也要把木卫三的既成事实写成程序违规，为地球保住追认或追责的空间。",
                micro_expression="他把屏幕上的‘拒不执行’标红，眼睛却短暂地离开了那行字，像不想承认它此刻毫无重量。",
                action_and_dialogue="他冷声记录：‘木卫三已在未经地球批准的情况下实施本地戒严与港口接管，相关状态暂记为非法既成事实。’",
                transmission_target="韩漠",
                transmission_content="地球确认你已构成非法既成事实。你现在拥有的不是合法权力，只是一段尚未被追究的延迟。",
            )
        if is_jupiter:
            return AgentAction(
                observation_analysis="迟到的镇压令只要无法反转码头上的签字和枪口，就只能替自己证明中心确实已经晚了。",
                emotional_shift="他没有兴奋，只觉得舱内空气比刚才更薄，像一切胜利都需要立刻拿更多现实去压住。",
                hidden_agenda="把地球的迟到命令公开成反面证据，让更多本地人相信他们已经不必等。",
                micro_expression="他把收到的镇压令投在公共墙屏上，嘴角没有起伏，只是让那串时间戳自己说话。",
                action_and_dialogue="他对议事厅说：‘看清楚，地球现在才抵达四十分钟前的港口。我们要是还等它，就只配统治自己的过去。’",
                transmission_target="沈竟",
                transmission_content="你的命令已公开展示给议事厅。它证明的不是统治，而是地球在木卫三只能收到自己的过去。",
            )

    if "光锥盲区" in brief or "谈判" in brief or "第五章" in brief:
        if is_earth:
            return AgentAction(
                observation_analysis="他永远收不到韩漠此刻的表情，因此所谓谈判不过是在两段过去之间挑一个损失更小的版本存档。",
                emotional_shift="他说话仍然平，可胸口像压着一张不断失效的主权地图，每一秒都在变旧。",
                hidden_agenda="保住地球对木星圈的最低名义与结算接口，别让分裂被一次性说死。",
                micro_expression="他在停顿里把指尖从桌面边缘挪开，像怕自己碰碎那点还剩的程序体面。",
                action_and_dialogue="他对迟到的通讯链路说：‘地球可以暂不派接管舰队，但木卫三必须承认航道法统仍由地球备案。名义先留下，事实以后再谈。’",
                transmission_target="韩漠",
                transmission_content="地球愿意接受木卫三的临时自治现实，但要求保留航道法统备案权与部分清算名义。",
            )
        if is_jupiter:
            return AgentAction(
                observation_analysis="沈竟真正想保住的不是现场，而是名义；这说明中心已经接受自己拿不到此刻，只能要求在文件上活下去。",
                emotional_shift="他听完那段旧了四十分钟的声音，后槽牙才很轻地咬紧一下，像在给已经到手的现实再加一道锁。",
                hidden_agenda="让木卫三的自治先被地球语言被动承认，再把名义上的让步压到最低。",
                micro_expression="他没有立刻回话，只抬手把公共时延图上的地球坐标关掉，屏幕顿时暗了一格。",
                action_and_dialogue="他平声回复：‘木卫三可以把部分备案名称留给地球，但港口、民兵和结算顺位不会再等地球批准。我们只接受已经发生的法。’",
                transmission_target="沈竟",
                transmission_content="木卫三接受最低限度的名义备案交换，但港口、民兵与结算现实不再等待地球批准。",
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
            observation_analysis="她问的不是价钱，而是违约定义，说明她已经没有讨价还价的空间，只剩确认损失边界。",
            emotional_shift="他听见自己把条款念得很稳，稳得像那只裂了口的杯子一直没有继续裂下去。",
            hidden_agenda="完成最终签注，把人和责任一并推回系统流程里。",
            micro_expression="他念条款时拇指在杯沿细裂上停了一瞬，又像什么都没碰到一样松开。",
            action_and_dialogue="他把终审页推向她：‘协议生效后，你的连续梦权会被平台托管。疗养仓和夜间供氧可以续到明晚，但你不能再私自保留完整梦层。’",
        )

    return None


class MockCharacterEngine:
    def invoke(self, *, profile: CharacterProfile, state: SceneState) -> AgentAction:
        action = build_mock_character_action(profile, state)
        if action is not None:
            return action
        return AgentAction(
            observation_analysis="他从对方的停顿和流程提示里确认，眼前这一轮谈话不会带来善意，只会继续压缩可选择的余地。",
            emotional_shift="身体已经先于语言进入紧绷状态，像在等一张迟早要落下来的罚单。",
            hidden_agenda="在不彻底失去主动权的前提下，尽量把当前损失推迟到下一轮。",
            micro_expression="他把视线从屏幕角落挪开时停了一瞬，又很快恢复成一副没事的样子。",
            action_and_dialogue="他压低声音，把条件往现实的一侧推了一寸：‘先把能确认的部分落下来，剩下的账以后再算。’",
        )


def _infer_mock_opening(scene_brief: str, state: SceneState) -> tuple[str, str, str, str]:
    previous_time = state.get("time_marker") or "凌晨四点十二分"
    previous_location = state.get("current_location") or "雾港第七码头的情绪采样站担保窗口"
    if "四十分钟旧闻" in scene_brief or "木卫三" in scene_brief or "主权协调厅" in scene_brief:
        return (
            "retain" if not state.get("chapter_history") else "shift",
            "这是信息延迟政治的开端，故事必须明确把人物钉在彼此错开的现在里。",
            "地球标准时 07:40",
            "地球同步轨道主权协调厅",
        )
    keyword_map = [
        (
            ("镇压指令在途",),
            (
                "shift",
                "视角切到木卫三，让地球命令在飞行途中继续变旧。",
                "木卫三地方时 08:20",
                "木卫三外环港务调度室",
            ),
        ),
        (
            ("五秒期货", "中继站", "保险价格"),
            (
                "shift",
                "故事进入‘谁更早收到未来’的市场维度。",
                "木星结算时 08:27",
                "木星信号中继站的结算舱",
            ),
        ),
        (
            ("回波戒严", "戒严"),
            (
                "shift",
                "让迟到的法令和已经落地的事实在同一地点相撞。",
                "木卫三地方时 09:05",
                "木卫三公共议事厅",
            ),
        ),
        (
            ("光锥盲区", "谈判"),
            (
                "shift",
                "结尾要回到跨光程谈判，强调双方永远拿不到同一个现在。",
                "地球标准时 11:10",
                "地球-木卫三延迟谈判链路的录音室",
            ),
        ),
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
    if any(token in scene_brief for token in ["旧闻", "盲区", "期货", "回波", "光锥"]):
        beat_templates = [
            (
                "把一条旧消息变成新的政治现实",
                "一段已经过时的情报被迫充当最新依据。",
                "角色必须根据别处已经过去的事实，替自己当前的处境下注。",
            ),
            (
                "让市场或法令先一步宣布未来",
                "报价、保险、戒严令或接管令在不同地点以不同时间到达。",
                "谁更早收到数据，谁就更先把它写成现实。",
            ),
            (
                "用迟到的回波完成一次不可逆收束",
                "某条命令或回复终于抵达，但现场已经被另一套事实占住。",
                "角色只能在保全名义和保全现实之间选一个更冷的版本。",
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
        dramatic_function, forced_event, target_shift = beat_templates[min(index - 1, len(beat_templates) - 1)]
        beats.append(
            {
                "round_index": index,
                "dramatic_function": dramatic_function,
                "forced_event": f"{forced_event}（围绕：{scene_brief}）",
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
        if any(token in scene_brief for token in ["旧闻", "盲区", "期货", "回波", "光锥", "木卫三"]):
            return ShowrunnerPlan(
                scene_brief=scene_brief,
                scene_purpose="把一场政治决策写成迟到信息怎样逼着人用过时现实下注。",
                target_ending="至少一方把尚未抵达的命令、尚未确认的市场或尚未同步的合法性，提前写成了无法回收的现实。",
                core_conflict="地球试图用迟到的法统覆盖木星圈已经发生的事实；木卫三则要在回波到达前把自治做成既成事实。",
                hidden_foreshadowing="时延图上的闪烁坐标和一条迟到的语音回波，会反复提醒所有人：他们从未活在同一个现在。",
                tone_guardrail="不要写英雄主义，不要写宏大战争快感，只写延迟如何把法律、市场和命令一寸寸拆开。",
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
    if "四十分钟旧闻" in scene_brief or "主权协调厅" in scene_brief:
        return (
            opening_time,
            opening_location,
            "地球标准时 07:48",
            "地球同步轨道主权协调厅的延迟图前",
        )
    if "镇压指令在途" in scene_brief:
        return (
            opening_time,
            opening_location,
            "木卫三地方时 08:34",
            "木卫三外环港务调度室外的低重力走廊",
        )
    if "五秒期货" in scene_brief:
        return (
            opening_time,
            opening_location,
            "木星结算时 08:33",
            "木星信号中继站的结算舱门口",
        )
    if "回波戒严" in scene_brief:
        return (
            opening_time,
            opening_location,
            "木卫三地方时 09:16",
            "木卫三公共议事厅外的戒严通道",
        )
    if "光锥盲区" in scene_brief or "谈判" in scene_brief:
        return (
            opening_time,
            opening_location,
            "地球标准时 11:26",
            "地球-木卫三延迟谈判链路的回放室外",
        )
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
            next_scene_pressure=("下一场只会让这些后果在新的地点或更晚的时间里继续发作。" if state.get("communication_mode") != "delayed_inbox" else "下一场里，仍在路上的旧命令和更早落地的新事实会继续互相错开。"),
        )
