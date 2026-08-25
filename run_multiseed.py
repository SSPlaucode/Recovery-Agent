"""
Multi-seed robustness check.

Two-phase workflow, matching the approved cached-benchmark
architecture (this was NOT actually wired through before -- caught by
review; the first version always used llm_mode="live", which is
reproducible only because DemoLLMClient happens to be deterministic,
not because the workflow itself enforced replay):

    # Phase 1: live calls, populates each seed's cache file
    python run_multiseed.py --n 500 --seeds 20 --llm-mode live

    # Phase 2: replay from cache, ZERO live LLM calls
    python run_multiseed.py --n 500 --seeds 20 --llm-mode cached

Explicit --seed-list variant:
    python run_multiseed.py --n 500 --seed-list 1,2,3,4,5 --llm-mode cached

A single seed=42 batch (run_batch.py) can't distinguish "this strategy
is genuinely better" from "this particular random draw happened to
favor it." This runs the full A/B/C comparison across many
independent seeds and reports mean +/- population stdev for recovery
rate and total recovered, per strategy -- so a claim like "AI beats
the rule engine" can be checked against seed-to-seed variance instead
of resting on one number.

Each seed gets its OWN AI response cache file (ai_cache_seed<N>.json)
since the transaction batch differs per seed -- there's no shared
cache to warm across seeds the way there is for repeated runs of the
same seed. In cached mode, a seed whose cache file doesn't exist yet
falls back to the rule engine for every decision in that seed (counted
via cache_misses) rather than silently going live -- run phase 1 first.
"""

import argparse
import csv
import json
import statistics

from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Transactions per seed")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run: 1..N")
    parser.add_argument("--seed-list", type=str, default=None,
                         help="Comma-separated explicit seed list, overrides --seeds")
    parser.add_argument("--llm-client", choices=["demo", "anthropic", "gemini"], default="demo")
    parser.add_argument("--llm-mode", choices=["live", "cached"], default="live",
                         help="live: call the client, populate each seed's cache. "
                              "cached: replay only, zero live calls, requires phase-1 "
                              "cache files to already exist.")
    parser.add_argument("--out", type=str, default="multiseed_results.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seed_list.split(",")] if args.seed_list else list(range(1, args.seeds + 1))

    print(f"Running {len(seeds)} seeds x {args.n} transactions each "
          f"({len(seeds) * args.n} total simulated cases), llm-mode={args.llm_mode}...")

    per_seed = {"A": [], "B": [], "C": []}  # each entry: (recovery_rate, total_recovered)
    rows = []
    total_live_calls = total_cache_hits = total_cache_misses = total_fallbacks = 0

    for seed in seeds:
        result = run_experiment(
            n=args.n, seed=seed,
            llm_client_kind=args.llm_client, llm_mode=args.llm_mode,
            cache_file=f"ai_cache_seed{seed}.json",
            verbose=False,
        )
        for label, m in [("A", result.metrics_a), ("B", result.metrics_b), ("C", result.metrics_c)]:
            per_seed[label].append((m.recovery_rate, m.total_recovered))
            rows.append({
                "seed": seed, "strategy": label,
                "recovery_rate": m.recovery_rate, "total_recovered": m.total_recovered,
                "escalated_count": m.escalated_count,
            })
        total_live_calls += result.ai_stats.live_calls
        total_cache_hits += result.ai_stats.cache_hits
        total_cache_misses += result.ai_stats.cache_misses
        total_fallbacks += result.ai_stats.fallbacks
        print(f"  seed {seed}: A={result.metrics_a.recovery_rate*100:.1f}%  "
              f"B={result.metrics_b.recovery_rate*100:.1f}%  "
              f"C={result.metrics_c.recovery_rate*100:.1f}%  "
              f"(live={result.ai_stats.live_calls} cache_hit={result.ai_stats.cache_hits} "
              f"fallback={result.ai_stats.fallbacks})")

    print(f"\nAI calls across all {len(seeds)} seeds: {total_live_calls} live, "
          f"{total_cache_hits} cache hits, {total_cache_misses} cache misses, "
          f"{total_fallbacks} fallbacks.")
    if args.llm_mode == "cached":
        if total_live_calls == 0:
            print("  Confirmed: cached mode made ZERO live LLM calls.")
        else:
            print("  WARNING: cached mode made live calls -- this should never happen; "
                  "investigate before trusting these numbers.")

    print(f"\n=== Multi-seed summary ({len(seeds)} seeds, n={args.n} each) ===")
    names = {"A": "Fixed Retry", "B": "Rule-Engine Agent", "C": "AI Agent"}
    summary = {}
    for label in ("A", "B", "C"):
        rates = [r for r, _ in per_seed[label]]
        totals = [t for _, t in per_seed[label]]
        mean_rate = statistics.mean(rates)
        stdev_rate = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        mean_total = statistics.mean(totals)
        stdev_total = statistics.pstdev(totals) if len(totals) > 1 else 0.0
        summary[label] = {
            "mean_recovery_rate": mean_rate, "stdev_recovery_rate": stdev_rate,
            "mean_total_recovered": mean_total, "stdev_total_recovered": stdev_total,
        }
        print(f"\n  {label}: {names[label]}")
        print(f"    Recovery rate:    {mean_rate*100:.2f}% +/- {stdev_rate*100:.2f}%")
        print(f"    Total recovered:  ₹{mean_total:,.2f} +/- ₹{stdev_total:,.2f}")

    # A quick, honest signal: does C's mean beat B's mean by more than
    # roughly one stdev of C's own variability? This is NOT a proper
    # significance test -- just a sanity check on whether the gap is
    # bigger than the noise before claiming it in a pitch.
    gap = summary["C"]["mean_recovery_rate"] - summary["B"]["mean_recovery_rate"]
    noise = summary["C"]["stdev_recovery_rate"]
    print(f"\n  C vs B mean recovery-rate gap: {gap*100:+.2f} percentage points "
          f"(C's stdev across seeds: {noise*100:.2f} points)")
    if noise > 0 and abs(gap) < noise:
        print("  NOTE: this gap is smaller than C's own seed-to-seed variability --")
        print("  treat 'AI beats rule engine' as unproven at this sample size, not confirmed.")

    with open(args.out, "w") as f:
        json.dump({"per_seed": rows, "summary": summary, "seeds": seeds, "n": args.n}, f, indent=2)
    print(f"\nFull multi-seed results written to {args.out}")

    csv_path = args.out.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "strategy", "recovery_rate",
                                                "total_recovered", "escalated_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-seed CSV written to {csv_path}")


if __name__ == "__main__":
    main()
