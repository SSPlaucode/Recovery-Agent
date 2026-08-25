"""
Data models for the AI Revenue Recovery prototype (Version 0).

customer_response_probability is intentionally the ONLY field the
agent/policy layer never reads directly. It exists purely as hidden
simulator state used to decide whether a customer-facing action
(schedule_retry, send_recovery_message) succeeds. Exposing it to the
decision layer would let the "agent" read the answer key.
"""

from dataclasses import dataclass, field
from enum import Enum


class PaymentMethod(str, Enum):
    CARD = "CARD"
    UPI = "UPI"
    NETBANKING = "NETBANKING"


class FailureReason(str, Enum):
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    CARD_NETWORK_ERROR = "CARD_NETWORK_ERROR"
    BANK_DECLINE = "BANK_DECLINE"
    EXPIRED_CARD = "EXPIRED_CARD"
    AUTH_FAILURE = "AUTH_FAILURE"
    UNKNOWN = "UNKNOWN"


class PaymentHistory(str, Enum):
    RELIABLE = "RELIABLE"
    USUALLY_PAYS_LATE = "USUALLY_PAYS_LATE"
    UNRELIABLE = "UNRELIABLE"


class TerminalState(str, Enum):
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"
    STOPPED = "STOPPED"


class ActionType(str, Enum):
    RETRY = "RETRY"                    # immediate retry, same instrument
    SCHEDULE_RETRY = "SCHEDULE_RETRY"  # delayed retry (funds may arrive)
    SEND_MESSAGE = "SEND_MESSAGE"      # ask customer to act (update card / pay)
    ESCALATE = "ESCALATE"
    STOP = "STOP"


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    payment_method: PaymentMethod
    failure_reason: FailureReason
    previous_failures: int
    customer_payment_history: PaymentHistory
    time_since_failure_hours: float
    subscription_flag: bool

    # Hidden simulator state. Never read by policy.py or the diagnosis step.
    customer_response_probability: float = field(repr=False)

    # Mutated as the case is processed.
    attempt_count: int = 0
    transaction_status: str = "FAILED"


@dataclass
class ActionResult:
    action: ActionType
    success: bool
    amount_recovered: float


@dataclass
class AuditEvent:
    transaction_id: str
    step: int
    state: str
    detail: str


@dataclass
class CaseResult:
    transaction_id: str
    amount: float
    terminal_state: TerminalState
    amount_recovered: float
    total_attempts: int
    failure_reason: "FailureReason" = None
    had_policy_violation: bool = False
    had_ai_fallback: bool = False
    final_confidence: float = None
    first_recommended_action: "ActionType" = None
    ai_calls: int = 0
    audit_trail: list = field(default_factory=list)
