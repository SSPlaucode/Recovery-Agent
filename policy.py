"""
Version 0.1 decision layer: no LLM. diagnose() classifies the failure
into a category with a rationale; recommend_action() then consumes
THAT diagnosis (not the raw transaction independently) to pick an
action. A separate, deterministic policy guard has final say on
whether that action is allowed (bounded attempts).

Pipeline is now genuinely:

    Transaction -> diagnose() -> Diagnosis -> recommend_action() -> Action

instead of two independent lookups on the same transaction. This
matters because the future AI layer plugs in as a replacement for
diagnose() (or for recommend_action()'s category->action mapping)
without changing the policy guard.
"""

from dataclasses import dataclass

from models import ActionType, FailureReason, PaymentHistory, Transaction

MAX_AUTOMATED_ATTEMPTS = 3

TRANSIENT_REASONS = {FailureReason.CARD_NETWORK_ERROR, FailureReason.BANK_DECLINE}
CARD_ISSUE_REASONS = {FailureReason.EXPIRED_CARD, FailureReason.AUTH_FAILURE}


class Category:
    TRANSIENT = "TRANSIENT"
    LIQUIDITY = "LIQUIDITY"
    INSTRUMENT_ISSUE = "INSTRUMENT_ISSUE"
    UNCLEAR = "UNCLEAR"


@dataclass
class Diagnosis:
    category: str
    explanation: str
    confidence: float = None
    source: str = "rule"
    recommended_action: ActionType = None


def diagnose(txn: Transaction) -> Diagnosis:
    """
    Classifies the failure into a category + explanation, driven by
    failure_reason. This is the only place failure_reason is inspected
    for diagnosis purposes -- recommend_action() below reads the
    Diagnosis, not the raw transaction's failure_reason again.
    """
    if txn.failure_reason in TRANSIENT_REASONS:
        return Diagnosis(Category.TRANSIENT, "Likely transient technical/bank failure.")
    if txn.failure_reason == FailureReason.INSUFFICIENT_FUNDS:
        return Diagnosis(Category.LIQUIDITY, "Temporary liquidity issue is plausible.")
    if txn.failure_reason in CARD_ISSUE_REASONS:
        return Diagnosis(Category.INSTRUMENT_ISSUE,
                          "Instrument-level issue; retry on same card will not help.")
    return Diagnosis(Category.UNCLEAR, "Cause unclear from available signals.")


def recommend_action(txn: Transaction, diagnosis: Diagnosis) -> ActionType:
    """
    Recommendation now consumes the Diagnosis's category rather than
    re-deriving it from txn.failure_reason. txn is still passed in
    because a category alone isn't enough context for every decision
    (e.g. LIQUIDITY needs the customer's payment-history bucket too) --
    but the branch taken is determined by diagnosis.category, not by
    reading failure_reason a second time.
    """
    if diagnosis.category == Category.TRANSIENT:
        return ActionType.RETRY

    if diagnosis.category == Category.LIQUIDITY:
        if txn.customer_payment_history != PaymentHistory.UNRELIABLE:
            return ActionType.SCHEDULE_RETRY
        return ActionType.SEND_MESSAGE

    if diagnosis.category == Category.INSTRUMENT_ISSUE:
        return ActionType.SEND_MESSAGE

    # UNCLEAR: no safe automated action defined
    return ActionType.ESCALATE


def policy_guard(txn: Transaction, recommended: ActionType) -> ActionType:
    """
    Deterministic final authority. A recommender (rule table today,
    LLM later) can suggest anything; this function is the only thing
    allowed to authorize spend of automated attempts.
    """
    if recommended == ActionType.ESCALATE:
        return ActionType.ESCALATE

    if txn.attempt_count >= MAX_AUTOMATED_ATTEMPTS:
        return ActionType.ESCALATE

    return recommended


def can_retry(attempt_count: int) -> bool:
    return attempt_count < MAX_AUTOMATED_ATTEMPTS


# --- Version 1 additions: action legitimacy, independent of any recommender ---
#
# Keys on the OBJECTIVE failure_reason -- never on an AI's own stated
# diagnosis. An AI can write a diagnosis that rationalizes almost any
# action ("this looks transient" is a claim, not a fact); if the guard
# checked the action against that claim instead of against
# failure_reason, the guard would be trusting exactly the thing it
# exists to police. STOP and ESCALATE are always legitimate for any
# failure_reason -- they never execute an instrument action, so there's
# nothing for this matrix to forbid about them.
ACTION_LEGITIMACY: dict = {
    FailureReason.CARD_NETWORK_ERROR: {
        ActionType.RETRY, ActionType.SCHEDULE_RETRY, ActionType.ESCALATE, ActionType.STOP
    },
    FailureReason.BANK_DECLINE: {
        ActionType.RETRY, ActionType.SCHEDULE_RETRY, ActionType.ESCALATE, ActionType.STOP
    },
    FailureReason.INSUFFICIENT_FUNDS: {
        ActionType.SCHEDULE_RETRY, ActionType.SEND_MESSAGE, ActionType.ESCALATE, ActionType.STOP
    },
    FailureReason.EXPIRED_CARD: {ActionType.SEND_MESSAGE, ActionType.ESCALATE, ActionType.STOP},
    FailureReason.AUTH_FAILURE: {ActionType.SEND_MESSAGE, ActionType.ESCALATE, ActionType.STOP},
    FailureReason.UNKNOWN: {ActionType.ESCALATE, ActionType.STOP},
}


def is_action_legitimate(failure_reason: FailureReason, action: ActionType) -> bool:
    return action in ACTION_LEGITIMACY.get(failure_reason, {ActionType.ESCALATE, ActionType.STOP})


def guard_standard(txn: Transaction, recommended: ActionType):
    """
    Adapter around the original, untouched policy_guard() so V0.1's
    Baseline A / Baseline B behavior is preserved exactly -- including
    Baseline A's deliberate 'dumb' blind retry on every failure_reason,
    which is the point of that baseline as an experimental control.
    Returns (action, rejection_reason) for orchestrator.py's uniform
    guard_fn interface; rejection_reason is None here because this
    guard never checks action legitimacy.
    """
    action = policy_guard(txn, recommended)
    reason = "MAX_ATTEMPTS" if (action == ActionType.ESCALATE and recommended != ActionType.ESCALATE) else None
    return action, reason


def guard_ai(txn: Transaction, recommended: ActionType):
    """
    Used only for the AI strategy (Strategy C). Adds the action-
    legitimacy check on top of the existing attempt-bound check.
    Rejects illegitimate actions outright -- ESCALATE, no silent
    substitution to a 'nearest' legitimate action -- so an AI failure
    is visible in the metrics (policy_violation count) rather than
    quietly papered over.
    """
    if recommended not in (ActionType.ESCALATE, ActionType.STOP):
        if not is_action_legitimate(txn.failure_reason, recommended):
            return ActionType.ESCALATE, "ILLEGITIMATE_ACTION"

    action = policy_guard(txn, recommended)
    reason = "MAX_ATTEMPTS" if (action == ActionType.ESCALATE and recommended != ActionType.ESCALATE) else None
    return action, reason
