from dataclasses import dataclass

from models import CaseResult, TerminalState


@dataclass
class BatchMetrics:
    strategy_name: str
    total_transactions: int
    total_revenue_at_risk: float
    total_recovered: float
    recovery_rate: float
    recovered_count: int
    escalated_count: int
    stopped_count: int
    avg_attempts_to_recovery: float
    policy_violation_count: int = 0
    ai_fallback_count: int = 0
    avg_confidence: float = None
    total_ai_calls: int = 0
    avg_ai_calls_per_case: float = 0.0
    unrecovered_revenue: float = 0.0
    escalation_rate: float = 0.0
    stop_rate: float = 0.0

    def print_table(self):
        print(f"\n--- {self.strategy_name} ---")
        print(f"  Transactions:            {self.total_transactions}")
        print(f"  Revenue at risk:         ₹{self.total_revenue_at_risk:,.2f}")
        print(f"  Recovered:               ₹{self.total_recovered:,.2f}")
        print(f"  Unrecovered:             ₹{self.unrecovered_revenue:,.2f}")
        print(f"  Recovery rate:           {self.recovery_rate * 100:.2f}%")
        print(f"  Cases recovered:         {self.recovered_count}")
        print(f"  Escalated:               {self.escalated_count} ({self.escalation_rate * 100:.2f}%)")
        print(f"  Stopped:                 {self.stopped_count} ({self.stop_rate * 100:.2f}%)")
        print(f"  Avg attempts (recovered):{self.avg_attempts_to_recovery:.2f}")
        if self.total_ai_calls or self.ai_fallback_count or self.policy_violation_count or self.avg_confidence is not None:
            print(f"  Total AI calls:          {self.total_ai_calls} "
                  f"({self.avg_ai_calls_per_case:.2f} per case)")
            print(f"  AI fallbacks:            {self.ai_fallback_count} / {self.total_transactions}")
            print(f"  AI policy violations:    {self.policy_violation_count} / {self.total_transactions}")
            if self.avg_confidence is not None:
                print(f"  Avg AI confidence:       {self.avg_confidence:.2f}")


def compute_agreement(results_a: list[CaseResult], results_b: list[CaseResult]) -> dict:
    """
    First-action agreement/divergence between two strategies run on the
    IDENTICAL transaction batch (same transaction_id at each index --
    experiment.py guarantees this via verify_identical_starting_transactions).
    Used for B vs C: does the AI's first recommended action match what
    the rule engine would have done? Near-100% agreement is itself a
    finding (the AI may not be adding anything); this makes that
    checkable from the benchmark output directly, not just from
    run_pilot.py's ad hoc printout.
    """
    assert len(results_a) == len(results_b), "compute_agreement requires equal-length, index-aligned results"
    agree = 0
    disagree = 0
    for ra, rb in zip(results_a, results_b):
        assert ra.transaction_id == rb.transaction_id, "results are not index-aligned by transaction_id"
        if ra.first_recommended_action == rb.first_recommended_action:
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": (agree / total) if total else 0.0,
        "divergence_rate": (disagree / total) if total else 0.0,
    }


def compute_metrics(strategy_name: str, results: list[CaseResult]) -> BatchMetrics:
    total_at_risk = sum(r.amount for r in results)
    total_recovered = sum(r.amount_recovered for r in results)
    recovered = [r for r in results if r.terminal_state == TerminalState.RECOVERED]
    escalated = [r for r in results if r.terminal_state == TerminalState.ESCALATED]
    stopped = [r for r in results if r.terminal_state == TerminalState.STOPPED]

    avg_attempts = (
        sum(r.total_attempts for r in recovered) / len(recovered) if recovered else 0.0
    )

    policy_violations = sum(1 for r in results if r.had_policy_violation)
    fallbacks = sum(1 for r in results if r.had_ai_fallback)
    confidences = [r.final_confidence for r in results if r.final_confidence is not None]
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None
    total_ai_calls = sum(r.ai_calls for r in results)
    avg_ai_calls = (total_ai_calls / len(results)) if results else 0.0
    unrecovered = total_at_risk - total_recovered
    n = len(results)
    escalation_rate = (len(escalated) / n) if n else 0.0
    stop_rate = (len(stopped) / n) if n else 0.0

    return BatchMetrics(
        strategy_name=strategy_name,
        total_transactions=len(results),
        total_revenue_at_risk=total_at_risk,
        total_recovered=total_recovered,
        recovery_rate=(total_recovered / total_at_risk) if total_at_risk else 0.0,
        recovered_count=len(recovered),
        escalated_count=len(escalated),
        stopped_count=len(stopped),
        avg_attempts_to_recovery=avg_attempts,
        policy_violation_count=policy_violations,
        ai_fallback_count=fallbacks,
        avg_confidence=avg_confidence,
        total_ai_calls=total_ai_calls,
        avg_ai_calls_per_case=avg_ai_calls,
        unrecovered_revenue=unrecovered,
        escalation_rate=escalation_rate,
        stop_rate=stop_rate,
    )


def compute_category_breakdown(results: list[CaseResult]) -> dict[str, dict]:
    """
    Recovery rate by failure_reason, so wins/losses can be attributed
    to a specific category rather than only seen in an aggregate
    number. Keys are the failure_reason enum's .value string.
    """
    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        key = r.failure_reason.value if r.failure_reason else "UNKNOWN"
        by_category.setdefault(key, []).append(r)

    breakdown = {}
    for category, cat_results in by_category.items():
        at_risk = sum(r.amount for r in cat_results)
        recovered = sum(r.amount_recovered for r in cat_results)
        breakdown[category] = {
            "count": len(cat_results),
            "at_risk": at_risk,
            "recovered": recovered,
            "recovery_rate": (recovered / at_risk) if at_risk else 0.0,
        }
    return breakdown


def print_category_comparison(named_breakdowns: list):
    """
    named_breakdowns: list of (label, breakdown_dict) pairs, 2 or more.
    Prints one row per failure category, one column per strategy.
    """
    all_categories = sorted(set().union(*[set(b) for _, b in named_breakdowns]))
    print(f"\n--- Recovery rate by failure category ---")
    header = f"  {'Category':<22}" + "".join(f"{label:>16}" for label, _ in named_breakdowns)
    print(header)
    for cat in all_categories:
        row = f"  {cat:<22}"
        for _, breakdown in named_breakdowns:
            rate = breakdown.get(cat, {}).get("recovery_rate", 0.0) * 100
            row += f"{rate:>15.2f}%"
        print(row)
