"""
Baseline A -- Fixed retry.

No diagnosis, no failure-reason awareness: every failed transaction
just gets retry_payment() called on it, up to the same attempt bound
as the rule engine (so the comparison isn't rigged by giving one
strategy more tries than the other).

Signatures match policy.diagnose / policy.recommend_action so both
strategies can run through the same orchestrator.run_case loop.
"""

from models import ActionType, Transaction
from policy import Diagnosis


def fixed_retry_diagnose(txn: Transaction) -> Diagnosis:
    return Diagnosis(category="N/A", explanation="No diagnosis performed (fixed-retry baseline).")


def fixed_retry_recommend(txn: Transaction, diagnosis: Diagnosis) -> ActionType:
    return ActionType.RETRY
