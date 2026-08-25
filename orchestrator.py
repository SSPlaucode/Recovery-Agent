"""
Runs one transaction through the recovery state machine:

DETECTED -> ANALYZING -> DIAGNOSED -> INTERVENTION_SELECTED -> POLICY_CHECK
POLICY_CHECK --reject--> ESCALATED / STOPPED
POLICY_CHECK --allow--> ACTION_EXECUTED
ACTION_EXECUTED --success--> RECOVERED (terminal)
ACTION_EXECUTED --failure--> back to DIAGNOSED (attempt_count += 1),
    bounded by guard_fn's attempt check

A "strategy" is (diagnose_fn, recommend_fn, guard_fn) so the same loop
runs Baseline A, Baseline B, and Strategy C (the AI agent) -- the FSM,
audit trail, and CaseResult shape are shared, only the recommendation
and guard differ. guard_fn defaults to policy.guard_standard, which
wraps the ORIGINAL, unmodified policy_guard() so Baseline A/B behavior
is byte-for-byte unchanged from V0.1. Strategy C passes
policy.guard_ai, which additionally enforces action legitimacy against
the objective failure_reason -- never against the AI's own diagnosis.

ONE-INITIAL-CALL DESIGN (this version): diagnose_fn/recommend_fn are
used ONLY on the first decision for a case (txn.attempt_count == 0).
Every decision after the first failed attempt uses retry_diagnose_fn/
retry_recommend_fn instead -- which default to diagnose_fn/recommend_fn
themselves when not given, so Baseline A/B (already deterministic on
every call) see no behavior change at all. Strategy C is the one that
matters: it's wired (in experiment.py) with retry_diagnose_fn=
policy.diagnose, retry_recommend_fn=policy.recommend_action -- the
SAME rule engine that already powers Strategy B, reused rather than
reinvented. This means the AI makes exactly one strategic call per
case; every retry after that is fully deterministic, bounded by the
same guard and attempt limit as always. The AI is not consulted again
merely because its recommended action failed -- "a materially new
decision state" that would justify a second call is not synthesized
in this version, by design (see review discussion).
"""

from models import ActionType, AuditEvent, CaseResult, TerminalState, Transaction
from policy import guard_standard
from simulator import RecoverySimulator

MAX_TRANSITIONS = 20  # hard ceiling, belt-and-suspenders against any loop bug

# Diagnosis.source values that represent an actual AI-layer interaction
# (a real call, a cache replay of a real call, or a failed attempt at
# one) -- as opposed to "rule", which never touches the AI at all.
# Used to count ai_calls per case regardless of which strategy is
# running, so the count is meaningful for A/B (always 0) and C alike.
_AI_SOURCES = {"ai", "ai_cached", "ai_fallback"}


def run_case(txn: Transaction, sim: RecoverySimulator, diagnose_fn, recommend_fn,
             guard_fn=guard_standard, retry_diagnose_fn=None, retry_recommend_fn=None) -> CaseResult:
    retry_diagnose_fn = retry_diagnose_fn or diagnose_fn
    retry_recommend_fn = retry_recommend_fn or recommend_fn

    audit: list[AuditEvent] = []
    step = 0
    total_recovered = 0.0
    had_policy_violation = False
    had_ai_fallback = False
    final_confidence = None
    first_recommended_action = None
    ai_calls = 0

    def log(state: str, detail: str):
        nonlocal step
        audit.append(AuditEvent(txn.transaction_id, step, state, detail))
        step += 1

    def finish(terminal_state: TerminalState) -> CaseResult:
        return CaseResult(
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            terminal_state=terminal_state,
            amount_recovered=total_recovered,
            total_attempts=txn.attempt_count,
            failure_reason=txn.failure_reason,
            had_policy_violation=had_policy_violation,
            had_ai_fallback=had_ai_fallback,
            final_confidence=final_confidence,
            first_recommended_action=first_recommended_action,
            ai_calls=ai_calls,
            audit_trail=audit,
        )

    log("DETECTED", f"amount=₹{txn.amount}, failure={txn.failure_reason.value}")

    while step < MAX_TRANSITIONS:
        log("ANALYZING", f"attempt_count={txn.attempt_count}")

        # Only the very first decision for this case (attempt_count == 0)
        # uses the "initial" functions -- which may call the AI. Every
        # decision after a failed attempt uses the deterministic retry
        # functions instead, so the AI is never consulted again just
        # because its first recommendation failed.
        is_first_decision = txn.attempt_count == 0
        active_diagnose_fn = diagnose_fn if is_first_decision else retry_diagnose_fn
        active_recommend_fn = recommend_fn if is_first_decision else retry_recommend_fn

        diagnosis = active_diagnose_fn(txn)
        source_note = "initial" if is_first_decision else "deterministic-retry"
        log("DIAGNOSED", f"[{diagnosis.category}/{diagnosis.source}/{source_note}] {diagnosis.explanation}"
                          + (f" (confidence={diagnosis.confidence:.2f})" if diagnosis.confidence is not None else ""))
        if diagnosis.source in _AI_SOURCES:
            ai_calls += 1
        if diagnosis.source == "ai_fallback":
            had_ai_fallback = True
        if diagnosis.confidence is not None:
            final_confidence = diagnosis.confidence

        recommended = active_recommend_fn(txn, diagnosis)
        log("INTERVENTION_SELECTED", f"recommended={recommended.value}")
        if first_recommended_action is None:
            first_recommended_action = recommended

        allowed, rejection_reason = guard_fn(txn, recommended)
        if rejection_reason == "ILLEGITIMATE_ACTION":
            had_policy_violation = True
            log("POLICY_CHECK", f"REJECTED {recommended.value}: not a legitimate action for "
                                 f"{txn.failure_reason.value} -> forced ESCALATE")
        elif allowed != recommended:
            log("POLICY_CHECK", f"rejected {recommended.value} ({rejection_reason}); forced {allowed.value}")
        else:
            log("POLICY_CHECK", f"allowed {allowed.value} (attempt {txn.attempt_count + 1}/3)")

        if allowed == ActionType.ESCALATE:
            log("ESCALATED", "max attempts reached, illegitimate action, or no safe automated action")
            return finish(TerminalState.ESCALATED)

        if allowed == ActionType.STOP:
            log("STOPPED", "policy stopped automated recovery")
            return finish(TerminalState.STOPPED)

        result = sim.execute(txn, allowed)
        txn.attempt_count += 1

        if result.success:
            total_recovered += result.amount_recovered
            log("ACTION_EXECUTED", f"{allowed.value} -> SUCCESS, recovered=₹{result.amount_recovered}")
            log("RECOVERED", f"total_recovered=₹{total_recovered}")
            return finish(TerminalState.RECOVERED)
        else:
            log("ACTION_EXECUTED", f"{allowed.value} -> FAILURE")
            # loop continues -> back to DIAGNOSED, now using the
            # deterministic retry functions (not the AI), still bounded
            # by guard_fn's attempt_count check next time through

    # Safety net: should be unreachable given MAX_AUTOMATED_ATTEMPTS=3,
    # but guarantees the FSM can never spin forever.
    log("STOPPED", "transition ceiling reached")
    return finish(TerminalState.STOPPED)
