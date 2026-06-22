"""
Dialogue State Tracker — FSM + slot filling for structured business flows.
FSM switches from free-form chat into guided flows (returns, complaints).
State persists to Redis so in-progress flows survive restarts.
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum

from loguru import logger

from config import get_config


# <<< FSM core definitions
class DialogueState(str, Enum):
    """
    Dialogue states: top-level (free chat vs structured flow) and step-level within a flow.
    """
    IDLE = "idle"                       # waiting for user input
    FREE_CHAT = "free_chat"             # open-ended Q&A
    STRUCTURED_FLOW = "structured"      # inside a guided flow
    CLARIFYING = "clarifying"           # asking user for more info
    CONFIRMING = "confirming"           # waiting for user to confirm
    WAITING_USER = "waiting_user"       # waiting for user action
    HANDOFF = "handoff"                 # transferring to human
    TERMINATED = "terminated"           # ended


class FlowType(str, Enum):
    """Supported business flow types"""
    RETURN_EXCHANGE = "return_exchange"     # return / exchange flow
    COMPLAINT = "complaint"                 # complaint flow
    ORDER_LOOKUP = "order_lookup"           # order lookup (requires identity verification)
    FEEDBACK = "feedback"                   # feedback collection
    ACCOUNT_RECOVERY = "account_recovery"   # account recovery


# <<< Slot definitions
@dataclass
class Slot:
    """
    A single information field within a business flow.
    required, prompt, validate_func, and optional value/filled tracking.
    """
    name: str
    description: str
    prompt: str                         # question to ask the user
    required: bool = True
    value: Any = None
    filled: bool = False
    validate_func: Optional[Callable[[Any], Optional[str]]] = None
    attempts: int = 0                   # times already asked
    max_attempts: int = 3               # max times before escalating

    def validate(self) -> Optional[str]:
        """Validate current value; returns error string or None if OK"""
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
        """Fill a slot by name; returns False if validation fails"""
        for slot in self.slots:
            if slot.name == name:
                slot.value = value
                error = slot.validate()
                if error:
                    slot.attempts += 1
                    return False
                slot.filled = True
                logger.info(f"[SlotCollection] Slot '{name}' filled: {value}")
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
        """Return a human-readable slot-filling progress summary"""
        parts = []
        for slot in self.slots:
            status = "✓" if slot.filled else "○"
            value_str = str(slot.value) if slot.value else "(pending)"
            parts.append(f"  {status} {slot.description}: {value_str}")
        return "\n".join(parts)


# <<< Flow definitions
class FlowDefinition:
    """Pre-built slot collections for each flow type"""

    @staticmethod
    def return_exchange_flow() -> SlotCollection:
        """Return/exchange flow slots"""
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
        """Complaint flow slots"""
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
        """Look up flow definition by type"""
        flows = {
            FlowType.RETURN_EXCHANGE: FlowDefinition.return_exchange_flow,
            FlowType.COMPLAINT: FlowDefinition.complaint_flow,
        }
        factory = flows.get(flow_type)
        return factory() if factory else None


# <<< Finite state machine
@dataclass
class StateMachineContext:
    """FSM context persisted per session"""
    current_state: DialogueState = DialogueState.IDLE
    current_flow: Optional[FlowType] = None
    slots: Optional[SlotCollection] = None
    state_history: List[DialogueState] = field(default_factory=list)
    entered_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class DialogueStateMachine:
    """
    FSM that governs dialogue state transitions.
    Key transitions: IDLE->FREE_CHAT->STRUCTURED_FLOW->CONFIRMING->FREE_CHAT.
    Any state can reach HANDOFF or TERMINATED.
    """

    # Transition table
    TRANSITIONS: Dict[DialogueState, Set[DialogueState]] = {
        DialogueState.IDLE: {
            DialogueState.FREE_CHAT,
            DialogueState.STRUCTURED_FLOW,
            DialogueState.TERMINATED,
        },
        DialogueState.FREE_CHAT: {
            DialogueState.FREE_CHAT,         # continue free chat
            DialogueState.STRUCTURED_FLOW,   # trigger structured flow
            DialogueState.CLARIFYING,         # need clarification
            DialogueState.CONFIRMING,         # need confirmation
            DialogueState.WAITING_USER,       # wait for user action
            DialogueState.HANDOFF,            # transfer to human
            DialogueState.TERMINATED,         # end
        },
        DialogueState.STRUCTURED_FLOW: {
            DialogueState.STRUCTURED_FLOW,   # continue collecting info
            DialogueState.CLARIFYING,         # clarify a slot
            DialogueState.CONFIRMING,         # slots full, confirm
            DialogueState.WAITING_USER,       # wait for user (e.g. photo upload)
            DialogueState.HANDOFF,            # transfer to human
            DialogueState.FREE_CHAT,          # user cancelled flow
        },
        DialogueState.CLARIFYING: {
            DialogueState.STRUCTURED_FLOW,   # back to flow after clarification
            DialogueState.FREE_CHAT,          # back to free chat
            DialogueState.CONFIRMING,         # confirm after clarification
        },
        DialogueState.CONFIRMING: {
            DialogueState.FREE_CHAT,          # confirmation done
            DialogueState.STRUCTURED_FLOW,   # user wants to modify info
            DialogueState.WAITING_USER,       # waiting for user confirmation
        },
        DialogueState.WAITING_USER: {
            DialogueState.FREE_CHAT,
            DialogueState.STRUCTURED_FLOW,
            DialogueState.HANDOFF,
            DialogueState.TERMINATED,
        },
        DialogueState.HANDOFF: {
            DialogueState.FREE_CHAT,          # user continues chatting during transfer
            DialogueState.TERMINATED,         # transfer complete
        },
        DialogueState.TERMINATED: set(),      # terminal state
    }

    def __init__(self):
        self._contexts: Dict[str, StateMachineContext] = {}
        logger.info("[DialogueStateMachine] FSM initialized")

    def get_context(self, session_id: str) -> StateMachineContext:
        """Get or create FSM context for a session"""
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
        Execute a state transition. Returns (success, message).
        """
        ctx = self.get_context(session_id)
        current = ctx.current_state

        # Validate transition is allowed
        allowed = self.TRANSITIONS.get(current, set())
        if target_state not in allowed and current != target_state:
            return False, f"Transition forbidden: {current.value} -> {target_state.value}"

        # Record history
        ctx.state_history.append(current)
        ctx.current_state = target_state
        ctx.entered_at = time.time()

        # Initialize slots when entering structured flow
        if target_state == DialogueState.STRUCTURED_FLOW and flow_type:
            ctx.current_flow = flow_type
            ctx.slots = FlowDefinition.get_flow(flow_type)

        # Keep slot data on exit (don't clear here)
        if current == DialogueState.STRUCTURED_FLOW and target_state != DialogueState.STRUCTURED_FLOW:
            pass

        logger.info(
            f"[FSM] Transition: {session_id[:12]}... "
            f"{current.value} -> {target_state.value}"
        )
        return True, f"Transition ok: {current.value} -> {target_state.value}"

    def get_current_state(self, session_id: str) -> DialogueState:
        """Return current FSM state"""
        return self.get_context(session_id).current_state

    def is_in_structured_flow(self, session_id: str) -> bool:
        """True if the session is inside a structured flow"""
        ctx = self.get_context(session_id)
        return (ctx.current_state == DialogueState.STRUCTURED_FLOW and
                ctx.slots is not None)

    def fill_current_slot(self, session_id: str, value: Any) -> Tuple[bool, str, Optional[str]]:
        """
        Try to fill the current slot. Returns (success, prompt_msg, next_slot_name).
        """
        ctx = self.get_context(session_id)

        if not ctx.slots:
            return False, "No active slot collection", None

        next_slot = ctx.slots.next_unfilled
        if not next_slot:
            return True, "All required fields collected", None

        # Try filling
        success = ctx.slots.fill_slot(next_slot.name, value)
        if success:
            next_unfilled = ctx.slots.next_unfilled
            if next_unfilled:
                return True, next_unfilled.prompt, next_unfilled.name
            else:
                return True, "All info collected, please confirm", None
        else:
            next_slot.attempts += 1
            if next_slot.attempts >= next_slot.max_attempts:
                return False, (
                    f"{next_slot.max_attempts} attempts without valid input. "
                    "Transferring to human agent."
                ), None
            return False, next_slot.prompt, next_slot.name

    def reset(self, session_id: str):
        """Reset FSM context for a session"""
        if session_id in self._contexts:
            del self._contexts[session_id]
        logger.debug(f"[FSM] Reset: {session_id[:12]}...")


# <<< Dialogue state tracker (public API)
class DialogueStateTracker:
    """
    Public API that integrates FSM + slot filling.
    """

    def __init__(self):
        self.fsm = DialogueStateMachine()
        logger.info("[DialogueStateTracker] Initialized")

    def start_flow(
        self,
        session_id: str,
        flow_type: FlowType,
    ) -> Dict:
        """
        Start a structured flow. Returns the first prompt, e.g. "我要退货" -> return flow.
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
        Handle user input during a structured flow (slot filling, confirmation, etc.).
        """
        ctx = self.fsm.get_context(session_id)
        state = ctx.current_state

        # Input outside structured flow -> free chat
        if state != DialogueState.STRUCTURED_FLOW:
            return {
                "status": "free_chat",
                "state": state.value,
                "message": None,
            }

        # Check for cancel/exit signals
        cancel_signals = ["取消", "算了", "不用了", "不退了", "不想", "退出", "返回"]
        if any(sig in user_input for sig in cancel_signals):
            self.fsm.transition(session_id, DialogueState.FREE_CHAT)
            return {
                "status": "flow_cancelled",
                "message": "好的，已取消当前流程。请问还有什么可以帮您的？",
            }

        # Try filling the current slot
        success, prompt, next_slot = self.fsm.fill_current_slot(session_id, user_input)

        # Check if all required slots are filled
        if success and ctx.slots and ctx.slots.all_filled:
            if ctx.current_state != DialogueState.CONFIRMING:
                self.fsm.transition(session_id, DialogueState.CONFIRMING)
            return {
                "status": "ready_to_confirm",
                "message": self._build_confirmation_message(ctx),
                "slots_summary": ctx.slots.to_summary(),
                "all_filled": True,
            }

        # User confirmation phase
        if ctx.current_state == DialogueState.CONFIRMING:
            confirm_signals = ["确认", "是", "对的", "正确", "yes", "ok", "没问题", "可以"]
            if any(sig in user_input.lower() for sig in confirm_signals):
                # Flow complete
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
                # Reset confirmation slot, go back to collecting
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

            # Unrecognized confirmation reply
            return {
                "status": "awaiting_confirmation",
                "message": "请回复'确认'提交，或回复'修改'调整信息。",
            }

        # Handle fill result
        if success:
            return {
                "status": "slot_filled",
                "message": prompt,
                "next_slot": next_slot,
                "progress": self._get_progress(ctx),
            }
        else:
            # Fill failed (validation error or max attempts)
            if next_slot is None:
                # Max attempts exceeded -> transfer to human
                self.fsm.transition(session_id, DialogueState.HANDOFF)
                return {
                    "status": "flow_failed",
                    "message": prompt,
                    "transfer_human": True,
                }
            return {
                "status": "retry_slot",
                "message": prompt,
                "next_slot": next_slot,
            }

    def get_status(self, session_id: str) -> Dict:
        """Return current dialogue state summary"""
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
        """Reset dialogue state for a session"""
        self.fsm.reset(session_id)

    def _build_confirmation_message(self, ctx: StateMachineContext) -> str:
        """Build the confirmation summary for the user"""
        if not ctx.slots:
            return "请确认以上信息。"
        summary = ctx.slots.to_summary()
        return f"请确认以下信息：\n\n{summary}\n\n回复'确认'提交，或回复'修改'调整。"

    def _get_progress(self, ctx: StateMachineContext) -> float:
        """Compute flow completion ratio (0-1)"""
        if not ctx.slots:
            return 0.0
        total_required = sum(1 for s in ctx.slots.slots if s.required)
        filled_required = sum(1 for s in ctx.slots.slots if s.required and s.filled)
        return filled_required / total_required if total_required > 0 else 0.0


_tracker: Optional[DialogueStateTracker] = None


def get_dialogue_state_tracker() -> DialogueStateTracker:
    global _tracker
    if _tracker is None:
        _tracker = DialogueStateTracker()
    return _tracker


# <<< Self-check
if __name__ == "__main__":
    logger.info("Dialogue State self-check")
    tracker = DialogueStateTracker()
    session_id = "test_fsm_001"

    # 1. Start return flow
    result = tracker.start_flow(session_id, FlowType.RETURN_EXCHANGE)
    logger.info(f"  Start flow: {result['status']}")

    # 2. Step through slots
    test_inputs = [
        "#20240001",
        "质量问题",
        "有照片",
        "衣服有明显的色差和线头",
        "确认",
    ]
    for i, user_input in enumerate(test_inputs):
        result = tracker.process_user_input(session_id, user_input)
        logger.info(f"  Step {i+1}: {result['status']}")

    # 3. Get status
    status = tracker.get_status(session_id)
    logger.info(f"  State: {status['state']}")

    # 4. Cancel flow
    tracker.start_flow(session_id, FlowType.RETURN_EXCHANGE)
    result = tracker.process_user_input(session_id, "算了不退了")
    logger.info(f"  Cancel: {result['status']}")

    logger.info("Dialogue state self-check complete.")
