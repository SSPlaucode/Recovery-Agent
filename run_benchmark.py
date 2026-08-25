"""
The single reproducible benchmark command for Track 3 submission.

    python run_benchmark.py --n 1000 --seed 42
    python run_benchmark.py --n 500 --seeds 20        # adds multi-seed robustness

Wraps experiment.py -- does not duplicate any business logic. Produces
both a machine-readable JSON file and a human-readable printed report
covering every field asked for:

  total transactions, total revenue at risk, recovered revenue,
  recovery rate, unrecovered revenue, escalation rate, stop rate,
  average automated attempts, AI calls per case, AI fallbacks,
  AI/rule first-action agreement, AI/rule first-action divergence,
  policy violations, breakdown by failure category

...for all three strategies (A: Fixed Retry, B: Rule Engine,
C: AI Agent), plus mean +/- stdev across seeds when --seeds > 1.

Methodology is unchanged from experiment.py: identical transactions
across strategies, strategy-independent simulator randomness,
deterministic seeds, bounded attempts, no hidden simulator state
exposed to the AI. This script only reports; it does not run anything
experiment.py doesn't already run.
"""

import argparse
import json
import statistics

from experiment import run_experiment
from metrics import print_category_comparison
from simulator import print_assumptions


def _strategy_summary(m, label):
    return {
        "strategy": label,
        "total_transactions": m.total_transactions,
        "total_revenue_at_risk": m.total_revenue_at_risk,
        "recovered_revenue": m.total_recovered,
        "unrecovered_revenue": m.unrecovered_revenue,
        "recovery_rate": m.recovery_rate,
        "escalated_count": m.escalated_count,
        "escalation_rate": m.escalation_rate,
        "stopped_count": m.stopped_count,
        "stop_rate": m.stop_rate,
        "avg_automated_attempts_to_recovery": m.avg_attempts_to_recovery,
        "ai_calls_total": m.total_ai_calls,
        "ai_calls_per_case": m.avg_ai_calls_per_case,
        "ai_fallback_count": m.ai_fallback_count,
        "policy_violation_count": m.policy_violation_count,
        "avg_ai_confidence": m.avg_confidence,
    }


def run_single_seed(n: int, seed: int, llm_client_kind: str, llm_mode: str, cache_file: str):
    print_assumptions()
    result = run_experiment(
        n=n, seed=seed, llm_client_kind=llm_client_kind, llm_mode=llm_mode,
        cache_file=cache_file, verbose=True,
    )
    print(f"\n{result.ai_stats.summary()}")

    print(f"\n=== Benchmark: {n} synthetic at-risk transactions (seed={seed}) ===")
    for m in (result.metrics_a, result.metrics_b, result.metrics_c):
        m.print_table()

    agr = result.agreement_bc
    print(f"\n--- AI vs Rule Engine: first-action agreement ---")
    print(f"  Agree:      {agr['agree']}/{agr['total']} ({agr['agreement_rate']*100:.2f}%)")
    print(f"  Diverge:    {agr['disagree']}/{agr['total']} ({agr['divergence_rate']*100:.2f}%)")

    print_category_comparison([
        (result.metrics_a.strategy_name, result.breakdown_a),
        (result.metrics_b.strategy_name, result.breakdown_b),
        (result.metrics_c.strategy_name, result.breakdown_c),
    ])

    def serialize_case(r):
        if r is None:
            return None
        return {
            "transaction_id": r.transaction_id,
            "amount": r.amount,
            "failure_reason": r.failure_reason.value if r.failure_reason else None,
            "terminal_state": r.terminal_state.value,
            "amount_recovered": r.amount_recovered,
            "total_attempts": r.total_attempts,
            "ai_calls": r.ai_calls,
            "had_ai_fallback": r.had_ai_fallback,
            "had_policy_violation": r.had_policy_violation,
            "final_confidence": r.final_confidence,
            "audit_trail": [{"step": e.step, "state": e.state, "detail": e.detail} for e in r.audit_trail],
        }

    from models import TerminalState
    recovered_sample = next((r for r in result.results_c if r.terminal_state == TerminalState.RECOVERED), None)
    escalated_sample = next((r for r in result.results_c if r.terminal_state == TerminalState.ESCALATED), None)
    violation_sample = next((r for r in result.results_c if r.had_policy_violation), None)

    report = {
        "methodology": {
            "n": n, "seed": seed, "llm_client": llm_client_kind, "llm_mode": llm_mode,
            "identical_transactions_across_strategies": True,
            "strategy_independent_simulator_randomness": True,
            "hidden_simulator_state_exposed_to_ai": False,
        },
        "strategies": [
            _strategy_summary(result.metrics_a, "A: Fixed Retry"),
            _strategy_summary(result.metrics_b, "B: Rule-Engine Agent"),
            _strategy_summary(result.metrics_c, "C: AI Agent"),
        ],
        "ai_vs_rule_first_action_agreement": agr,
        "category_breakdown": {
            "a_fixed_retry": result.breakdown_a,
            "b_rule_engine": result.breakdown_b,
            "c_ai_agent": result.breakdown_c,
        },
        "sample_cases": {
            "recovered": serialize_case(recovered_sample),
            "escalated": serialize_case(escalated_sample),
            "policy_violation": serialize_case(violation_sample),
        },
    }
    return report


def run_multi_seed(n: int, seeds: int, llm_client_kind: str, cache_file_prefix: str):
    seed_list = list(range(1, seeds + 1))
    print(f"\nRunning {len(seed_list)} seeds x {n} transactions each for robustness...")
    per_strategy = {"A": [], "B": [], "C": []}
    for seed in seed_list:
        result = run_experiment(
            n=n, seed=seed, llm_client_kind=llm_client_kind, llm_mode="live",
            cache_file=f"{cache_file_prefix}_seed{seed}.json", verbose=False,
        )
        for label, m in [("A", result.metrics_a), ("B", result.metrics_b), ("C", result.metrics_c)]:
            per_strategy[label].append((m.recovery_rate, m.total_recovered))
        print(f"  seed {seed}: A={result.metrics_a.recovery_rate*100:.1f}%  "
              f"B={result.metrics_b.recovery_rate*100:.1f}%  C={result.metrics_c.recovery_rate*100:.1f}%")

    summary = {}
    print(f"\n=== Multi-seed robustness ({len(seed_list)} seeds, n={n} each) ===")
    names = {"A": "Fixed Retry", "B": "Rule-Engine Agent", "C": "AI Agent"}
    for label in ("A", "B", "C"):
        rates = [r for r, _ in per_strategy[label]]
        totals = [t for _, t in per_strategy[label]]
        mean_rate, stdev_rate = statistics.mean(rates), (statistics.pstdev(rates) if len(rates) > 1 else 0.0)
        mean_total, stdev_total = statistics.mean(totals), (statistics.pstdev(totals) if len(totals) > 1 else 0.0)
        summary[label] = {
            "mean_recovery_rate": mean_rate, "stdev_recovery_rate": stdev_rate,
            "mean_total_recovered": mean_total, "stdev_total_recovered": stdev_total,
        }
        print(f"  {label}: {names[label]:<20} {mean_rate*100:.2f}% +/- {stdev_rate*100:.2f}pp  "
              f"(₹{mean_total:,.2f} +/- ₹{stdev_total:,.2f})")

    gap = summary["C"]["mean_recovery_rate"] - summary["B"]["mean_recovery_rate"]
    noise = summary["C"]["stdev_recovery_rate"]
    print(f"\n  C vs B mean gap: {gap*100:+.2f}pp (C's stdev: {noise*100:.2f}pp)")
    if noise > 0 and abs(gap) < noise:
        print("  Gap is smaller than C's own seed-to-seed variability -- do not claim")
        print("  statistical significance from this sample size.")

    return {"seeds": seed_list, "n": n, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the single-seed detailed run")
    parser.add_argument("--seeds", type=int, default=1,
                         help="If > 1, also run a multi-seed robustness pass (1..N)")
    parser.add_argument("--llm-client", choices=["demo", "anthropic", "gemini"], default="demo")
    parser.add_argument("--cache-file", type=str, default="ai_cache.json")
    parser.add_argument("--out", type=str, default="benchmark_report.json")
    args = parser.parse_args()

    single = run_single_seed(args.n, args.seed, args.llm_client, "live", args.cache_file)

    report = {"single_seed": single}
    if args.seeds > 1:
        report["multi_seed"] = run_multi_seed(args.n, args.seeds, args.llm_client, "ai_cache_bench")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report written to {args.out}")


if __name__ == "__main__":
    main()
