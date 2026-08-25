"""
Real-model pilot -- small, cheap, and meant to be READ, not aggregated.

    python run_pilot.py --n 5 --seed 42 --llm-client gemini

Per review: before trusting any aggregate percentage from a real
model, look at the actual per-case decisions. Runs Rule Engine (B) and
the AI agent (C) on n transactions (one seed, defaults to a small
5-case pilot -- this script validates that the live integration
works, not model quality at scale; see SUBMISSION.md for why a small
n is deliberate here, not a shortcut), caches
every AI response, and writes a per-case comparison: the rule engine's
action, the AI's diagnosis + action + confidence, whether the policy
guard allowed or rejected it, and whether the AI's action AGREES with
the rule engine's first action.

That agreement column is the direct way to answer the concern that
matters most here: failure_reason alone makes diagnosis close to
trivial ("EXPIRED_CARD" -> "the card is expired" needs no real
reasoning), so the interesting question isn't whether the AI can
diagnose -- it's whether the AI's ACTION choice uses the rest of the
context (previous_failures, time_since_failure_hours,
customer_payment_history, subscription_flag) in a way that sometimes
diverges from what failure_reason alone would suggest, and whether
those divergences make sense. If agreement is ~100%, that's a signal
the AI isn't adding anything the rule engine doesn't already do --
report that plainly, don't paper over it.

Baseline A also runs (matches the reviewer's stated three-strategy
scope, and costs nothing extra since it's not an LLM call), but the
per-case table is B vs C, since that's the open question.

Defaults to --llm-client gemini (₹0-budget project -- Gemini's
Flash-family models currently have a free tier; verify which specific
model at https://ai.google.dev/gemini-api/docs/pricing before running,
since that changes over time). Pass --llm-client demo for a dry run of
the report machinery with no API key and no cost, or --llm-client
anthropic if you have a paid Anthropic key instead.

Requires GEMINI_MODEL (or ANTHROPIC_MODEL for --llm-client anthropic)
to be set explicitly -- see llm_client.py, no silent default for
either provider.
"""

import argparse
import csv

from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-file", type=str, default="pilot_cache.json")
    parser.add_argument("--llm-client", choices=["demo", "anthropic", "gemini"], default="gemini")
    parser.add_argument("--out", type=str, default="pilot_comparison.csv")
    args = parser.parse_args()

    print(f"Real-model pilot: n={args.n}, seed={args.seed}, llm_client={args.llm_client}")
    if args.llm_client == "anthropic":
        print("This will make live calls to the Anthropic API and may incur cost.\n")
    elif args.llm_client == "gemini":
        print("This will make live calls to the Gemini API. Verify your chosen GEMINI_MODEL")
        print("is actually within your account's free tier at")
        print("https://ai.google.dev/gemini-api/docs/pricing -- this script does not check.\n")

    result = run_experiment(
        n=args.n, seed=args.seed,
        llm_client_kind=args.llm_client, llm_mode="live",
        cache_file=args.cache_file, verbose=True,
    )

    print(f"\n{result.ai_stats.summary()}")

    rows = []
    agree = 0
    disagree = 0
    violations = 0
    for rb, rc in zip(result.results_b, result.results_c):
        assert rb.transaction_id == rc.transaction_id, "results_b/results_c out of order"
        agrees = rb.first_recommended_action == rc.first_recommended_action
        agree += 1 if agrees else 0
        disagree += 0 if agrees else 1
        violations += 1 if rc.had_policy_violation else 0

        rows.append({
            "transaction_id": rb.transaction_id,
            "failure_reason": rb.failure_reason.value if rb.failure_reason else "",
            "rule_engine_action": rb.first_recommended_action.value if rb.first_recommended_action else "",
            "ai_action": rc.first_recommended_action.value if rc.first_recommended_action else "",
            "ai_source": "ai_fallback" if rc.had_ai_fallback else "ai",
            "ai_confidence": rc.final_confidence,
            "ai_calls": rc.ai_calls,
            "agree_with_rule_engine": agrees,
            "policy_violation": rc.had_policy_violation,
            "rule_engine_terminal": rb.terminal_state.value,
            "rule_engine_recovered": rb.amount_recovered,
            "ai_terminal": rc.terminal_state.value,
            "ai_recovered": rc.amount_recovered,
        })

    total = len(rows)
    print(f"\n=== Per-case comparison (n={total}) ===")
    if total:
        print(f"Rule engine vs AI agreement on first action: {agree}/{total} agree, {disagree}/{total} diverge "
              f"({agree/total*100:.1f}% agreement)")
    print(f"Policy violations (AI recommended an illegitimate action): {violations}/{total}")

    if disagree:
        print(f"\n--- Divergent cases (AI chose differently from the rule engine) ---")
        print(f"{'TX_ID':<12}{'failure_reason':<20}{'rule_action':<16}{'ai_action':<16}"
              f"{'confidence':<11}{'source'}")
        for r in rows:
            if not r["agree_with_rule_engine"]:
                conf = f"{r['ai_confidence']:.2f}" if r["ai_confidence"] is not None else "n/a"
                print(f"{r['transaction_id']:<12}{r['failure_reason']:<20}{r['rule_engine_action']:<16}"
                      f"{r['ai_action']:<16}{conf:<11}{r['ai_source']}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull per-case comparison written to {args.out}")

    print("\nRead this file before trusting any aggregate percentage. Specifically check:")
    print("  - Do divergent cases make sense given previous_failures / time_since_failure_hours /")
    print("    customer_payment_history / subscription_flag, or do they look arbitrary?")
    if total:
        print(f"  - Agreement was {agree/total*100:.1f}% -- if this is close to 100%, the AI may just be")
        print("    restating failure_reason, which is the exact thing this pilot exists to check.")
    print("  - Read a few 'ai_fallback' rows if any exist: was the LLM output actually malformed,")
    print("    or did validation reject something that was reasonable? (Check ai_agent.py's")
    print("    validate_response if you suspect the latter -- that would be a validator bug,")
    print("    not an AI failure.)")


if __name__ == "__main__":
    main()
