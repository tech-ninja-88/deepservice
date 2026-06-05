"""
=============================================================================
DeepService 对话管理 — 对话状态跟踪 (Dialogue State Tracker)
=============================================================================
职责：
  1. 有限状态机（FSM）管理对话流程 — 结构化业务流程
  2. 槽位填充（Slot Filling）— 逐步收集业务所需信息
  3. 多轮信息确认 — 退换货/投诉等需要多步确认的场景

企业级设计原则：
  - FSM 管理需要结构化信息收集的业务流程（退换货、工单提交）
  - 槽位机制确保信息完整（避免遗漏退货原因、照片等关键信息）
  - 状态可持久化（Redis）保证服务重启不丢失进行中的流程
  - 自由对话与结构化流程可切换（大部分时间是自由对话）

对话状态机示例（退换货流程）：
  ┌─────────┐   确认意图   ┌──────────┐   收集订单号  ┌──────────┐
  │  IDLE   │────────────→│CONFIRMING│────────────→│SLOT_FILL │
  └─────────┘             └──────────┘             └─────┬────┘
       ↑                                                  │
       │                                  收集退货原因     │
       │              ┌──────────┐       收集照片         │
       └──────────────│COMPLETED │←───────────────────────┘
                      └──────────┘  槽位已满 / 用户确认

参考：
  [reference:4] — 意图识别-对话管理-回复生成三段式架构
=============================================================================
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum

from loguru import logger

from config import get_config


# ============================================================================
# 状态机核心定义
# ============================================================================
class DialogueState(str, Enum):
    """
    对话状态枚举

    状态分为两层：
      - 全局状态：对话的整体状态（自由对话 / 结构化流程中）
      - 流程状态：具体业务流程的步骤（退换货的第几步）
    """
    # 全局状态
    IDLE = "idle"                       # 空闲，等待用户输入
    FREE_CHAT = "free_chat"             # 自由对话模式（常规问答）
    STRUCTURED_FLOW = "structured"      # 结构化流程中（退换货、工单等）
    CLARIFYING = "clarifying"           # 澄清中（需要用户补充信息）
    CONFIRMING = "confirming"           # 确认中（等待用户确认信息）
    WAITING_USER = "waiting_user"       # 等待用户操作
    HANDOFF = "handoff"                 # 转人工中
    TERMINATED = "terminated"           # 已结束


class FlowType(str, Enum):
    """业务流程类型"""
    RETURN_EXCHANGE = "return_exchange"     # 退换货流程
    COMPLAINT = "complaint"                 # 投诉流程
    ORDER_LOOKUP = "order_lookup"           # 订单查询（需验证身份）
    FEEDBACK = "feedback"                   # 反馈建议收集
    ACCOUNT_RECOVERY = "account_recovery"   # 账号找回


# ============================================================================
# 槽位定义
# ============================================================================
@dataclass
class Slot:
    """
    槽位 — 业务流程中的一个信息字段

    属性：
      - required: 是否必填
      - prompt: 向用户询问该信息的问题模板
      - validate: 验证函数（返回 None=通过，返回 str=错误提示）
      - value: 当前填充值
      - filled: 是否已填充
    """
    name: str                           # 槽位名称
    description: str                    # 描述
    prompt: str                         # 询问用户的提示语
    required: bool = True
    value: Any = None
    filled: bool = False
    validate_func: Optional[Callable[[Any], Optional[str]]] = None
    attempts: int = 0                   # 已尝试询问次数
    max_attempts: int = 3               # 最大询问次数

    def validate(self) -> Optional[str]:
        """验证当前值，返回错误信息或 None（通过）"""
        if self.required and (self.value is None or self.value == ""):
            return f"请提供{self.description}"

        if self.validate_func and self.value is not None:
            return self.validate_func(self.value)

        return None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "required": self.required,
            "value": self.value,
            "filled": self.filled,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }


@dataclass
class SlotCollection:
    """
    槽位集合 — 管理一个业务流程的所有槽位

    支持槽位依赖关系（如：先填订单号才能查订单详情）。
    """
    slots: List[Slot] = field(default_factory=list)
    current_slot_index: int = 0

    @property
    def all_filled(self) -> bool:
        return all(
            slot.filled or not slot.required
            for slot in self.slots
        )

    @property
    def missing_required(self) -> List[Slot]:
        return [
            slot for slot in self.slots
            if slot.required and not slot.filled
        ]

    @property
    def next_unfilled(self) -> Optional[Slot]:
        """获取下一个未填充的必填槽位"""
        for slot in self.slots:
            if slot.required and not slot.filled:
                if slot.attempts < slot.max_attempts:
                    return slot
        return None

    def fill_slot(self, name: str, value: Any) -> bool:
        """填充指定槽位"""
        for slot in self.slots:
            if slot.name == name:
                slot.value = value
                error = slot.validate()
                if error:
                    slot.attempts += 1
                    return False
                slot.filled = True
                logger.info(f"[SlotCollection] 槽位 {name} 已填充: {value}")
                return True
        return False

    def to_dict(self) -> Dict:
        return {
            "slots": [s.to_dict() for s in self.slots],
            "current_slot_index": self.current_slot_index,
            "all_filled": self.all_filled,
            "missing": [s.name for s in self.missing_required],
        }

    def to_summary(self) -> str:
        """生成槽位填充状态摘要"""
        parts = []
        for slot in self.slots:
            status = "✓" if slot.filled else "○"
            value_str = str(slot.value) if slot.value else "(待填写)"
            parts.append(f"  {status} {slot.description}: {value_str}")
        return "\n".join(parts)


# ============================================================================
# 流程定义
# ============================================================================
class FlowDefinition:
    """业务流程定义"""

    @staticmethod
    def return_exchange_flow() -> SlotCollection:
        """退换货流程槽位定义"""
        return SlotCollection(slots=[
            Slot(
                name="order_id",
                description="订单号",
                prompt="请提供您的订单号（可在'我的订单'中查看）",
                required=True,
                validate_func=lambda v: None if (str(v).strip() and len(str(v)) >= 6)
                else "请输入有效的订单号",
            ),
            Slot(
                name="reason",
                description="退换货原因",
                prompt="请选择退换货原因：\n1. 质量问题\n2. 尺寸不合适\n3. 商品与描述不符\n4. 不想要了\n5. 其他",
                required=True,
                validate_func=lambda v: None if str(v).strip() else "请选择退换货原因",
            ),
            Slot(
                name="has_photo",
                description="是否有凭证照片",
                prompt="请上传能说明问题的照片（质量问题必传）",
                required=False,
                validate_func=None,
            ),
            Slot(
                name="description",
                description="问题描述",
                prompt="请简单描述一下遇到的问题",
                required=False,
                validate_func=None,
            ),
            Slot(
                name="confirm",
                description="信息确认",
                prompt="请确认以上信息是否正确？（回复'确认'或'修改'）",
                required=True,
                validate_func=lambda v: None if v in ["确认", "是", "对的", "正确", "yes", "ok"]
                else "请回复'确认'继续，或回复'修改'调整信息",
            ),
        ])

    @staticmethod
    def complaint_flow() -> SlotCollection:
        """投诉流程槽位定义"""
        return SlotCollection(slots=[
            Slot(
                name="complaint_type",
                description="投诉类型",
                prompt="请问您要投诉哪方面的问题？\n1. 商品质量\n2. 物流服务\n3. 客服态度\n4. 售后处理\n5. 其他",
                required=True,
            ),
            Slot(
                name="order_id",
                description="相关订单号",
                prompt="请提供相关订单号（如无订单号，请回复'无'）",
                required=False,
            ),
            Slot(
                name="description",
                description="详细描述",
                prompt="请详细描述您遇到的问题（时间、经过、期望的处理方式）",
                required=True,
                validate_func=lambda v: None if len(str(v)) >= 10
                else "请至少提供10个字的描述",
            ),
            Slot(
                name="contact",
                description="联系方式",
                prompt="请留下您的联系方式（手机号），以便我们处理完毕后通知您",
                required=True,
                validate_func=lambda v: None if len(str(v).strip()) >= 8
                else "请输入有效的手机号码",
            ),
        ])

    @staticmethod
    def get_flow(flow_type: FlowType) -> Optional[SlotCollection]:
        """按类型获取流程定义"""
        flows = {
            FlowType.RETURN_EXCHANGE: FlowDefinition.return_exchange_flow,
            FlowType.COMPLAINT: FlowDefinition.complaint_flow,
        }
        factory = flows.get(flow_type)
        return factory() if factory else None


# ============================================================================
# 有限状态机 (FSM)
# ============================================================================
@dataclass
class StateMachineContext:
    """状态机上下文 — 保存在会话中"""
    current_state: DialogueState = DialogueState.IDLE
    current_flow: Optional[FlowType] = None
    slots: Optional[SlotCollection] = None
    state_history: List[DialogueState] = field(default_factory=list)
    entered_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class DialogueStateMachine:
    """
    有限状态机 — 管理对话流程的状态转移

    状态转移规则：
      IDLE → FREE_CHAT（用户开始对话）
      FREE_CHAT → STRUCTURED_FLOW（触发结构化流程）
      STRUCTURED_FLOW → CONFIRMING（收集完信息，等待确认）
      CONFIRMING → COMPLETED（用户确认）
      任意状态 → HANDOFF（触发转人工）
      任意状态 → TERMINATED（用户主动结束）
    """

    # 状态转移表
    TRANSITIONS: Dict[DialogueState, Set[DialogueState]] = {
        DialogueState.IDLE: {
            DialogueState.FREE_CHAT,
            DialogueState.STRUCTURED_FLOW,
            DialogueState.TERMINATED,
        },
        DialogueState.FREE_CHAT: {
            DialogueState.FREE_CHAT,         # 持续自由对话
            DialogueState.STRUCTURED_FLOW,   # 触发结构化流程
            DialogueState.CLARIFYING,         # 需要澄清
            DialogueState.CONFIRMING,         # 需要确认
            DialogueState.WAITING_USER,       # 等待用户操作
            DialogueState.HANDOFF,            # 转人工
            DialogueState.TERMINATED,         # 结束
        },
        DialogueState.STRUCTURED_FLOW: {
            DialogueState.STRUCTURED_FLOW,   # 持续收集信息
            DialogueState.CLARIFYING,         # 需要澄清槽位
            DialogueState.CONFIRMING,         # 槽位填满，确认
            DialogueState.WAITING_USER,       # 等待用户（如上传照片）
            DialogueState.HANDOFF,            # 转人工
            DialogueState.FREE_CHAT,          # 用户取消流程
        },
        DialogueState.CLARIFYING: {
            DialogueState.STRUCTURED_FLOW,   # 澄清后回到流程
            DialogueState.FREE_CHAT,          # 澄清后自由对话
            DialogueState.CONFIRMING,         # 澄清后确认
        },
        DialogueState.CONFIRMING: {
            DialogueState.FREE_CHAT,          # 确认完成
            DialogueState.STRUCTURED_FLOW,   # 用户要修改信息
            DialogueState.WAITING_USER,       # 等待用户确认
        },
        DialogueState.WAITING_USER: {
            DialogueState.FREE_CHAT,
            DialogueState.STRUCTURED_FLOW,
            DialogueState.HANDOFF,
            DialogueState.TERMINATED,
        },
        DialogueState.HANDOFF: {
            DialogueState.FREE_CHAT,          # 转接中用户继续聊天
            DialogueState.TERMINATED,         # 转接完成
        },
        DialogueState.TERMINATED: set(),      # 终态
    }

    def __init__(self):
        self._contexts: Dict[str, StateMachineContext] = {}
        logger.info("[DialogueStateMachine] FSM 初始化完成")

    def get_context(self, session_id: str) -> StateMachineContext:
        """获取或创建状态机上下文"""
        if session_id not in self._contexts:
            self._contexts[session_id] = StateMachineContext()
        return self._contexts[session_id]

    def transition(
        self,
        session_id: str,
        target_state: DialogueState,
        flow_type: Optional[FlowType] = None,
    ) -> Tuple[bool, str]:
        """
        执行状态转移

        返回: (是否成功, 消息)
        """
        ctx = self.get_context(session_id)
        current = ctx.current_state

        # 验证转移合法性
        allowed = self.TRANSITIONS.get(current, set())
        if target_state not in allowed and current != target_state:
            return False, f"不允许从 {current.value} 转移到 {target_state.value}"

        # 记录历史
        ctx.state_history.append(current)
        ctx.current_state = target_state
        ctx.entered_at = time.time()

        # 进入结构化流程时初始化槽位
        if target_state == DialogueState.STRUCTURED_FLOW and flow_type:
            ctx.current_flow = flow_type
            ctx.slots = FlowDefinition.get_flow(flow_type)

        # 退出结构化流程时清理
        if current == DialogueState.STRUCTURED_FLOW and target_state != DialogueState.STRUCTURED_FLOW:
            pass  # 保留槽位数据用于后续，不在转移时清理

        logger.info(
            f"[FSM] 状态转移: {session_id[:12]}... "
            f"{current.value} → {target_state.value}"
        )
        return True, f"状态转移成功: {current.value} → {target_state.value}"

    def get_current_state(self, session_id: str) -> DialogueState:
        """获取当前状态"""
        return self.get_context(session_id).current_state

    def is_in_structured_flow(self, session_id: str) -> bool:
        """是否在结构化流程中"""
        ctx = self.get_context(session_id)
        return (ctx.current_state == DialogueState.STRUCTURED_FLOW and
                ctx.slots is not None)

    def fill_current_slot(self, session_id: str, value: Any) -> Tuple[bool, str, Optional[str]]:
        """
        尝试填充当前槽位

        返回: (是否成功, 提示消息, 下一个要询问的槽位名称)
        """
        ctx = self.get_context(session_id)

        if not ctx.slots:
            return False, "当前无活跃的槽位集合", None

        next_slot = ctx.slots.next_unfilled
        if not next_slot:
            return True, "所有必填信息已收集完毕", None

        # 尝试填充
        success = ctx.slots.fill_slot(next_slot.name, value)
        if success:
            next_unfilled = ctx.slots.next_unfilled
            if next_unfilled:
                return True, next_unfilled.prompt, next_unfilled.name
            else:
                return True, "信息收集完成，请确认", None
        else:
            next_slot.attempts += 1
            if next_slot.attempts >= next_slot.max_attempts:
                return False, (
                    f"已尝试{next_slot.max_attempts}次仍未获取有效信息。"
                    "将为您转接人工客服。"
                ), None
            return False, next_slot.prompt, next_slot.name

    def reset(self, session_id: str):
        """重置状态机"""
        if session_id in self._contexts:
            del self._contexts[session_id]
        logger.debug(f"[FSM] 状态机已重置: {session_id[:12]}...")


# ============================================================================
# 对话状态跟踪器（统一接口）
# ============================================================================
class DialogueStateTracker:
    """
    对话状态跟踪器 — 对外统一接口

    整合了 FSM + Slot Filling，提供简洁的业务 API。
    """

    def __init__(self):
        self.fsm = DialogueStateMachine()
        logger.info("[DialogueStateTracker] 初始化完成")

    def start_flow(
        self,
        session_id: str,
        flow_type: FlowType,
    ) -> Dict:
        """
        启动一个结构化业务流程

        返回流程的第一个提示。
        示例用途：用户说"我要退货" → 启动退换货流程
        """
        success, msg = self.fsm.transition(
            session_id,
            DialogueState.STRUCTURED_FLOW,
            flow_type=flow_type,
        )

        if not success:
            return {"status": "error", "message": msg}

        ctx = self.fsm.get_context(session_id)
        if ctx.slots and ctx.slots.next_unfilled:
            first_prompt = ctx.slots.next_unfilled.prompt
        else:
            first_prompt = "请提供必要信息以继续。"

        return {
            "status": "flow_started",
            "flow_type": flow_type.value,
            "message": first_prompt,
            "next_slot": ctx.slots.next_unfilled.name if ctx.slots and ctx.slots.next_unfilled else None,
        }

    def process_user_input(
        self,
        session_id: str,
        user_input: str,
    ) -> Dict:
        """
        处理用户输入（在结构化流程中）

        根据当前状态和槽位状态，决定下一步动作。
        """
        ctx = self.fsm.get_context(session_id)
        state = ctx.current_state

        # 非结构化流程中的输入 → 自由对话
        if state != DialogueState.STRUCTURED_FLOW:
            return {
                "status": "free_chat",
                "state": state.value,
                "message": None,
            }

        # 检查是否要取消/退出流程
        cancel_signals = ["取消", "算了", "不用了", "不退了", "不想", "退出", "返回"]
        if any(sig in user_input for sig in cancel_signals):
            self.fsm.transition(session_id, DialogueState.FREE_CHAT)
            return {
                "status": "flow_cancelled",
                "message": "好的，已取消当前流程。请问还有什么可以帮您的？",
            }

        # 尝试填充槽位
        success, prompt, next_slot = self.fsm.fill_current_slot(session_id, user_input)

        # 检查是否所有槽位已填满
        if success and ctx.slots and ctx.slots.all_filled:
            if ctx.current_state != DialogueState.CONFIRMING:
                self.fsm.transition(session_id, DialogueState.CONFIRMING)
            return {
                "status": "ready_to_confirm",
                "message": self._build_confirmation_message(ctx),
                "slots_summary": ctx.slots.to_summary(),
                "all_filled": True,
            }

        # 用户确认
        if ctx.current_state == DialogueState.CONFIRMING:
            confirm_signals = ["确认", "是", "对的", "正确", "yes", "ok", "没问题", "可以"]
            if any(sig in user_input.lower() for sig in confirm_signals):
                # 完成流程
                flow_data = {
                    "flow_type": ctx.current_flow.value if ctx.current_flow else "unknown",
                    "slots": ctx.slots.to_dict() if ctx.slots else {},
                    "completed_at": time.time(),
                }
                self.fsm.transition(session_id, DialogueState.FREE_CHAT)
                return {
                    "status": "flow_completed",
                    "message": "信息已确认。我们将为您处理，请留意后续通知。",
                    "flow_data": flow_data,
                }

            modify_signals = ["修改", "改", "不对", "错了", "no"]
            if any(sig in user_input.lower() for sig in modify_signals):
                # 重置确认槽位，回到收集状态
                self.fsm.transition(session_id, DialogueState.STRUCTURED_FLOW)
                ctx.slots.current_slot_index = 0
                for slot in ctx.slots.slots:
                    if slot.name == "confirm":
                        slot.filled = False
                next_unfilled = ctx.slots.missing_required[0] if ctx.slots.missing_required else None
                return {
                    "status": "modifying",
                    "message": f"请重新输入。{next_unfilled.prompt if next_unfilled else ''}",
                }

            # 未识别的确认回复
            return {
                "status": "awaiting_confirmation",
                "message": "请回复'确认'提交，或回复'修改'调整信息。",
            }

        # 填充结果处理
        if success:
            return {
                "status": "slot_filled",
                "message": prompt,
                "next_slot": next_slot,
                "progress": self._get_progress(ctx),
            }
        else:
            # 填充失败（验证不通过或次数用尽）
            if next_slot is None:
                # 次数用尽，转人工
                self.fsm.transition(session_id, DialogueState.HANDOFF)
                return {
                    "status": "flow_failed",
                    "message": prompt,  # 转人工提示
                    "transfer_human": True,
                }
            return {
                "status": "retry_slot",
                "message": prompt,
                "next_slot": next_slot,
            }

    def get_status(self, session_id: str) -> Dict:
        """获取当前对话状态摘要"""
        ctx = self.fsm.get_context(session_id)
        result = {
            "session_id": session_id,
            "state": ctx.current_state.value,
            "flow_type": ctx.current_flow.value if ctx.current_flow else None,
        }
        if ctx.slots:
            result["slots"] = ctx.slots.to_dict()
        return result

    def reset(self, session_id: str):
        """重置对话状态"""
        self.fsm.reset(session_id)

    def _build_confirmation_message(self, ctx: StateMachineContext) -> str:
        """构建确认信息"""
        if not ctx.slots:
            return "请确认以上信息。"
        summary = ctx.slots.to_summary()
        return f"请确认以下信息：\n\n{summary}\n\n回复'确认'提交，或回复'修改'调整。"

    def _get_progress(self, ctx: StateMachineContext) -> float:
        """计算当前流程进度（0-1）"""
        if not ctx.slots:
            return 0.0
        total_required = sum(1 for s in ctx.slots.slots if s.required)
        filled_required = sum(1 for s in ctx.slots.slots if s.required and s.filled)
        return filled_required / total_required if total_required > 0 else 0.0


# ============================================================================
# 全局单例
# ============================================================================
import threading

_tracker: Optional[DialogueStateTracker] = None
_tracker_lock = threading.Lock()


def get_dialogue_state_tracker() -> DialogueStateTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = DialogueStateTracker()
    return _tracker


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Dialogue State Tracker — 独立测试")
    logger.info("=" * 60)

    tracker = DialogueStateTracker()
    session_id = "test_fsm_001"

    # 测试1：启动退换货流程
    logger.info("\n[测试1] 启动退换货流程")
    result = tracker.start_flow(session_id, FlowType.RETURN_EXCHANGE)
    logger.info(f"  状态: {result['status']}")
    logger.info(f"  提示: {result['message']}")

    # 测试2：逐步填充槽位
    test_inputs = [
        "#20240001",              # 订单号
        "质量问题",                # 退货原因
        "有照片",                  # 是否有照片
        "衣服有明显的色差和线头",    # 问题描述
        "确认",                    # 确认
    ]

    for i, user_input in enumerate(test_inputs):
        logger.info(f"\n[测试2-{i+1}] 用户输入: '{user_input}'")
        result = tracker.process_user_input(session_id, user_input)
        logger.info(f"  状态: {result['status']}")
        logger.info(f"  回复: {result.get('message', 'N/A')[:100]}")

    # 测试3：获取当前状态
    logger.info("\n[测试3] 当前状态")
    status = tracker.get_status(session_id)
    logger.info(f"  {json.dumps(status, ensure_ascii=False, indent=2)}")

    # 测试4：取消流程
    logger.info("\n[测试4] 取消流程")
    tracker.start_flow(session_id, FlowType.RETURN_EXCHANGE)
    result = tracker.process_user_input(session_id, "算了不退了")
    logger.info(f"  状态: {result['status']}")
    logger.info(f"  回复: {result.get('message', 'N/A')}")

    logger.info("=" * 60)
    logger.info("对话状态跟踪测试完成 ✓")
