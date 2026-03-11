from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .engine import validate_provider_credentials
from .models import SceneState
from .prompts import (
    WRITER_SYSTEM_PROMPT,
    build_writer_user_prompt,
    format_chapter_history,
    format_carryover_threads,
    format_relationship_snapshot,
    format_resource_snapshot,
    format_scene_log,
    format_showrunner_plan,
    format_symbolism_plan,
    format_transmission_log,
)


class SceneWriter(Protocol):
    def write(
        self,
        *,
        scene_data: str,
        subtext_guide: str,
        state: SceneState,
    ) -> str:
        ...


@dataclass
class LiveSceneWriter:
    model: str = "anthropic:claude-3-5-sonnet-latest"
    temperature: float = 0.9
    timeout: int = 120

    def write(
        self,
        *,
        scene_data: str,
        subtext_guide: str,
        state: SceneState,
    ) -> str:
        validate_provider_credentials(self.model)
        chat_model = init_chat_model(
            self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=2,
        )
        result = chat_model.invoke(
            [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_writer_user_prompt(
                        scene_data,
                        subtext_guide,
                        state.get("chapter_target", "1000 字左右"),
                    )
                ),
            ]
        )
        return getattr(result, "content", str(result)).strip()


class MockSceneWriter:
    def write(
        self,
        *,
        scene_data: str,
        subtext_guide: str,
        state: SceneState,
    ) -> str:
        return (
            "阮宁走到采样台前时，先把袖口往下带了一下，像是下意识去遮那些已经褪成浅色的针眼。"
            "站里的灯坏了几支，灰蓝色的政务字幕贴着终端底部缓慢往前滚，光从玻璃和金属边缘反回来，把人脸都磨得发冷。"
            "凌晨四点十二分，雾港第七码头重新开放情绪税抵扣窗口，广播里那句逾期冻结基础积分的话已经播到第三遍，像低烧一样退不下去。\n\n"
            "她开口先问压价，没问能不能通融。裴崧听得出来，这种人最麻烦的地方不在于会闹，而在于她们总能把快要塌下去的东西再撑一会儿，"
            "撑到连求情都显得多余。他把她昨晚的样本曲线调出来，屏幕冷光掠过那只放在终端边上的保温杯。杯沿上有一道新裂纹，细得像头发丝，"
            "只有手指蹭上去时才会有一点不该存在的阻滞。他说那只是市场波动，系统按规则结算。语气平得过了头，像不是他在说话，是一条处理过的语音模板借了他的嘴。\n\n"
            "阮宁没立刻接话，只把指尖压在台面边缘。后颈接口隔着衣领发热，像有一小块铁片埋在皮下，被灯光和疲惫一点点烤软。"
            "终端忽然弹出催款警报，母亲贴片额度的宽限期只剩三分钟，数字在屏幕角落跳动得很克制，并不刺眼，可她还是盯着看了两秒。"
            "那两秒让她脸上最后一点还能称为讨价还价的东西消失了。她只说，要复核就快一点。\n\n"
            "裴崧切到人工签注页时，右上角的配额红线跟着亮起来。保温杯被他往旁边推开半寸，指腹在裂纹上停了一下，又移开，像什么都没碰到。"
            "走廊尽头传来短促的骚动，像有人被拖着鞋跟走过地面，没人回头。排队号码继续往前跳，空气里是消毒水、潮湿铁锈和旧空调滤网一起发出来的味道。"
            "他告诉她，流程快慢取决于样本稳定性，不取决于她的排班。那句“对你更划算”说得很轻，轻得像一张自动打印出来的票据。\n\n"
            "到最后，确认单还是被推到了阮宁面前。异常标签贴在签字栏上方，像一块已经风干的痂。若拒绝签字，账户将在天亮前进入冻结序列。"
            "她没有再争。只是把名字写得很快，快得像想缩短自己暴露在台面上的时间。签完以后，她拇指在发炎的接口周围压了压，那里有一点隐秘的湿意，"
            "像身体里某个本该属于人的部分正在被标准件慢慢替换掉。\n\n"
            "裴崧补完备注，把单子推回去，动作标准得近乎礼貌，眼睛却始终没有真正落在她脸上。自动门在阮宁身后缓慢闭合，计时屏归零，"
            "杯沿那道裂纹在冷光里白了一瞬，又很快看不见了。外面天快亮了，站里的人都知道这一点，可谁也没有抬头去确认。"
        )


def build_scene_data(state: SceneState) -> str:
    return (
        f"[世界观]\n{state['world_context']}\n\n"
        f"[跨场连续性]\n"
        f"- 上一场摘要：{state.get('last_scene_summary') or '（这是小说开场）'}\n"
        f"- 当前开场坐标：{state.get('time_marker') or '未知时间'} @ {state.get('current_location') or '未知地点'}\n"
        f"- 累计章节历史：\n{format_chapter_history(state.get('chapter_history', []))}\n"
        f"- 仍在发作的线头：\n{format_carryover_threads(state.get('carryover_threads', []))}\n\n"
        f"[场景简报]\n{state['scene_brief']}\n\n"
        f"[Showrunner 节拍表]\n{format_showrunner_plan(state['showrunner_plan'])}\n\n"
        f"[角色关系快照]\n{format_relationship_snapshot(state['dynamic_relationships'])}\n\n"
        f"[资源压力快照]\n{format_resource_snapshot(state['resource_state'])}\n\n"
        f"[完整场景日志]\n{format_scene_log(state['scene_log'])}"
    )


def build_subtext_guide(state: SceneState) -> str:
    return format_symbolism_plan(state["symbolism_plan"])


def write_chapter(*, state: SceneState, writer: SceneWriter) -> dict[str, str]:
    scene_data = state["scene_data"] or build_scene_data(state)
    subtext_guide = state["subtext_guide"] or build_subtext_guide(state)
    chapter_text = writer.write(
        scene_data=scene_data,
        subtext_guide=subtext_guide,
        state=state,
    )
    return {
        "scene_data": scene_data,
        "subtext_guide": subtext_guide,
        "chapter_text": chapter_text,
    }
