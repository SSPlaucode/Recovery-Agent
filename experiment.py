"""
Core "run all three strategies on one seed" logic, factored out of
run_batch.py so run_multiseed.py can reuse it without duplicating the
strategy wiring. run_batch.py is the detailed single-seed CLI (prints
everything, writes JSON/CSV); run_multiseed.py calls this quietly
across many seeds and aggregates.
"""

import copy
from dataclasses import dataclass

import ai_agent
import baseline
import cache as cache_mod
import llm_client
import policy
from metrics import BatchMetrics, compute_agreement, compute_category_breakdown, compute_metrics
from models import CaseResult
from orchestrator import run_case
from simulator import RecoverySimulator, generate_transactions


@dataclass
class ExperimentResult:
    seed: int
    metrics_a: BatchMetrics
    metrics_b: BatchMetrics
    metrics_c: BatchMetrics
    results_a: list
    results_b: list
    results_c: list
    breakdown_a: dict
    breakdown_b: dict
    breakdown_c: dict
    ai_stats: "ai_agent.AgentStats"
    agreement_bc: dict = None


def _run_batch(transactions, sim, diagnose_fn, recommend_fn, strategy_name, guard_fn=None,
                retry_diagnose_fn=None, retry_recommend_fn=None):
    results: list[CaseResult] = []
    kwargs = {}
    if guard_fn is not None:
        kwargs["guard_fn"] = guard_fn
    if retry_diagnose_fn is not None:
        kwargs["retry_diagnose_fn"] = retry_diagnose_fn
    if retry_recommend_fn is not None:
        kwargs["retry_recommend_fn"] = retry_recommend_fn
    for txn in transactions:
        results.append(run_case(txn, sim, diagnose_fn, recommend_fn, **kwargs))
    return compute_metrics(strategy_name, results), results


def verify_identical_starting_transactions(txn_lists: list, verbose: bool = True):
    lengths = {len(t) for t in txn_lists}
    assert len(lengths) == 1, "Batch size mismatch between strategies"
    for group in zip(*txn_lists):
        first = group[0]
        for other in group[1:]:
            assert first.transaction_id == other.transaction_id
            assert first.amount == other.amount
            assert first.failure_reason == other.failure_reason
            assert first.customer_payment_history == other.customer_payment_history
            assert first.customer_response_probability == other.customer_response_probability
            assert other.attempt_count == 0
    if verbose:
        print(f"\nMethodology check: {len(txn_lists[0])}/{len(txn_lists[0])} transactions identical "
              f"across all {len(txn_lists)} strategies before any strategy ran -- passed.")


def run_experiment(n: int, seed: int, llm_client_kind: str = "demo", llm_mode: str = "live",
                    cache_file: str = "ai_cache.json", invalid_output_rate: float = 0.0,
                    verbose: bool = True) -> ExperimentResult:
    base_transactions = generate_transactions(n, seed=seed)

    txns_a = copy.deepcopy(base_transactions)
    txns_b = copy.deepcopy(base_transactions)
    txns_c = copy.deepcopy(base_transactions)
    verify_identical_starting_transactions([txns_a, txns_b, txns_c], verbose=verbose)

    # ONE shared simulator instance across all three strategies. This is
    # now safe (and is the whole point) because RecoverySimulator is a
    # pure function of (seed, transaction_id, attempt_count, action) --
    # it holds no sequential RNG state that a strategy's call pattern
    # could perturb. Sharing one instance makes "same simulated world"
    # a structural guarantee, not something that merely happens to be
    # true because each strategy's simulator was seeded identically.
    sim = RecoverySimulator(seed=seed + 1)

    metrics_a, results_a = _run_batch(
        txns_a, sim,
        diagnose_fn=baseline.fixed_retry_diagnose, recommend_fn=baseline.fixed_retry_recommend,
        strategy_name="A: Fixed Retry",
    )
    metrics_b, results_b = _run_batch(
        txns_b, sim,
        diagnose_fn=policy.diagnose, recommend_fn=policy.recommend_action,
        strategy_name="B: Rule-Engine Agent",
    )

    if llm_client_kind == "anthropic":
        client = llm_client.AnthropicLLMClient()
    elif llm_client_kind == "gemini":
        client = llm_client.GeminiLLMClient()
    else:
        client = llm_client.DemoLLMClient(seed=seed, invalid_output_rate=invalid_output_rate)
    response_cache = cache_mod.ResponseCache(path=cache_file)
    agent = ai_agent.AIAgent(client=client, cache=response_cache, mode=llm_mode)

    metrics_c, results_c = _run_batch(
        txns_c, sim,
        diagnose_fn=agent.diagnose_fn, recommend_fn=agent.recommend_fn,
        strategy_name="C: AI Agent", guard_fn=policy.guard_ai,
        # Deterministic retries after the first (AI) decision fails --
        # reuses the existing rule engine, doesn't invent a new
        # subsystem. See orchestrator.py's module docstring.
        retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
    )
    response_cache.save(cache_file)

    return ExperimentResult(
        seed=seed,
        metrics_a=metrics_a, metrics_b=metrics_b, metrics_c=metrics_c,
        results_a=results_a, results_b=results_b, results_c=results_c,
        breakdown_a=compute_category_breakdown(results_a),
        breakdown_b=compute_category_breakdown(results_b),
        breakdown_c=compute_category_breakdown(results_c),
        ai_stats=agent.stats,
        agreement_bc=compute_agreement(results_b, results_c),
    )
