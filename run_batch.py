"""
Version 1.1 single-seed batch runner. No FastAPI, no React.

    python run_batch.py --n 1000 --seed 42
    python run_batch.py --n 1000 --seed 42 --llm-mode cached --cache-file ai_cache.json

Runs the SAME synthetic batch through three strategies on IDENTICAL
starting transactions and identical simulator seeds:

  Strategy A - Fixed retry (no diagnosis)
  Strategy B - Rule-engine agent (deterministic diagnosis + policy)
  Strategy C - AI agent (LLM diagnosis + recommendation, gated by a
               policy guard that also enforces action legitimacy)

By default Strategy C uses DemoLLMClient (no network, no API key --
see llm_client.py's module docstring for why). Pass --llm-client
gemini once you have GEMINI_API_KEY and GEMINI_MODEL set (₹0-budget
default -- Gemini's Flash-family models currently have a free tier),
or --llm-client anthropic with ANTHROPIC_API_KEY/ANTHROPIC_MODEL if
you have a paid Anthropic key instead.

--llm-mode live (default): call the client (or read cache if this
    exact context was already seen), writing new responses into the
    cache file as they're produced.
--llm-mode cached: never calls the client. A miss falls back to the
    rule engine and is counted, not silently re-called live. Run once
    in live mode to warm the cache, then use cached mode for the
    numbers you report -- see experiment.py.

For a robustness check across many seeds (mean +/- stdev instead of a
single run), use run_multiseed.py instead.
"""

import argparse
import csv
import json

from experiment import run_experiment
from metrics import print_category_comparison
from models import TerminalState
from simulator import print_assumptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="batch_results.json")
    parser.add_argument("--llm-client", choices=["demo", "anthropic", "gemini"], default="demo")
    parser.add_argument("--llm-mode", choices=["live", "cached"], default="live")
    parser.add_argument("--cache-file", type=str, default="ai_cache.json")
    parser.add_argument("--invalid-output-rate", type=float, default=0.0,
                         help="DemoLLMClient only: fraction of calls that deliberately "
                              "return invalid output, to exercise the fallback path.")
    args = parser.parse_args()

    print_assumptions()

    result = run_experiment(
        n=args.n, seed=args.seed,
        llm_client_kind=args.llm_client, llm_mode=args.llm_mode,
        cache_file=args.cache_file, invalid_output_rate=args.invalid_output_rate,
        verbose=True,
    )
    metrics_a, metrics_b, metrics_c = result.metrics_a, result.metrics_b, result.metrics_c
    results_a, results_b, results_c = result.results_a, result.results_b, result.results_c
    breakdown_a, breakdown_b, breakdown_c = result.breakdown_a, result.breakdown_b, result.breakdown_c
    stats = result.ai_stats

    print(f"\n{stats.summary()}")
    total_decisions = stats.live_calls + stats.cache_hits + stats.fallbacks
    print(f"  LLM calls per transaction: {total_decisions / args.n:.2f} "
          f"({total_decisions} decisions requested across {args.n} transactions -- "
          f"a transaction can trigger more than one decision if it retries)")

    print(f"\n=== Batch: {args.n} synthetic at-risk transactions (seed={args.seed}) ===")
    for m in (metrics_a, metrics_b, metrics_c):
        m.print_table()

    improvement_b = (
        (metrics_b.total_recovered - metrics_a.total_recovered) / metrics_a.total_recovered * 100
        if metrics_a.total_recovered else float("inf")
    )
    improvement_c_vs_b = (
        (metrics_c.total_recovered - metrics_b.total_recovered) / metrics_b.total_recovered * 100
        if metrics_b.total_recovered else float("inf")
    )
    improvement_c_vs_a = (
        (metrics_c.total_recovered - metrics_a.total_recovered) / metrics_a.total_recovered * 100
        if metrics_a.total_recovered else float("inf")
    )
    print(f"\n  Improvement B vs A (rule engine vs fixed retry):  {improvement_b:+.2f}%")
    print(f"  Improvement C vs B (AI agent vs rule engine):     {improvement_c_vs_b:+.2f}%")
    print(f"  Improvement C vs A (AI agent vs fixed retry):     {improvement_c_vs_a:+.2f}%")
    print("\n  Reporting note: 'AI recovers X% more revenue' on its own is misleading --")
    print("  report C vs B (does AI beat a competent deterministic system) separately")
    print("  from B vs A (does context-aware routing beat blind retry at all). A single")
    print("  seed is also not a robustness claim -- see run_multiseed.py for mean +/- stdev")
    print("  across many seeds before treating this number as stable.")

    print_category_comparison([
        (metrics_a.strategy_name, breakdown_a),
        (metrics_b.strategy_name, breakdown_b),
        (metrics_c.strategy_name, breakdown_c),
    ])

    # --- Detailed sample cases for the demo ---
    recovered_sample = next((r for r in results_c if r.terminal_state == TerminalState.RECOVERED), None)
    escalated_sample = next((r for r in results_c if r.terminal_state == TerminalState.ESCALATED), None)
    violation_sample = next((r for r in results_c if r.had_policy_violation), None)

    for label, sample in [("C: RECOVERED case", recovered_sample),
                           ("C: ESCALATED case", escalated_sample),
                           ("C: POLICY VIOLATION case", violation_sample)]:
        if not sample:
            continue
        print(f"\n--- Sample {label}: {sample.transaction_id} ---")
        print(f"  Amount: ₹{sample.amount:,.2f}  Terminal: {sample.terminal_state.value}  "
              f"Recovered: ₹{sample.amount_recovered:,.2f}  Attempts: {sample.total_attempts}")
        for ev in sample.audit_trail:
            print(f"    [{ev.step:02d}] {ev.state:<22} {ev.detail}")

    # --- Persist results for inspection ---
    def serialize(results):
        return [
            {
                "transaction_id": r.transaction_id,
                "amount": r.amount,
                "failure_reason": r.failure_reason.value if r.failure_reason else None,
                "terminal_state": r.terminal_state.value,
                "amount_recovered": r.amount_recovered,
                "total_attempts": r.total_attempts,
                "had_policy_violation": r.had_policy_violation,
                "had_ai_fallback": r.had_ai_fallback,
                "ai_calls": r.ai_calls,
                "final_confidence": r.final_confidence,
                "audit_trail": [
                    {"step": e.step, "state": e.state, "detail": e.detail} for e in r.audit_trail
                ],
            }
            for r in results
        ]

    with open(args.out, "w") as f:
        json.dump(
            {
                "a_fixed_retry": serialize(results_a),
                "b_rule_engine": serialize(results_b),
                "c_ai_agent": serialize(results_c),
                "category_breakdown": {
                    "a_fixed_retry": breakdown_a,
                    "b_rule_engine": breakdown_b,
                    "c_ai_agent": breakdown_c,
                },
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {args.out}")

    summary_csv = args.out.replace(".json", "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "transaction_id", "amount", "failure_reason", "terminal_state",
                          "amount_recovered", "total_attempts", "had_policy_violation", "had_ai_fallback",
                          "ai_calls"])
        for name, results in [("fixed_retry", results_a), ("rule_engine", results_b), ("ai_agent", results_c)]:
            for r in results:
                writer.writerow([name, r.transaction_id, r.amount,
                                  r.failure_reason.value if r.failure_reason else "",
                                  r.terminal_state.value, r.amount_recovered, r.total_attempts,
                                  r.had_policy_violation, r.had_ai_fallback, r.ai_calls])
    print(f"Summary CSV written to {summary_csv}")


if __name__ == "__main__":
    main()
