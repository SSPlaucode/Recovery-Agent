import copy
import unittest

from baseline import fixed_retry_diagnose, fixed_retry_recommend
from metrics import compute_agreement, compute_category_breakdown, compute_metrics
from models import (
    ActionType,
    CaseResult,
    FailureReason,
    PaymentHistory,
    PaymentMethod,
    TerminalState,
    Transaction,
)
from orchestrator import run_case
from policy import (
    MAX_AUTOMATED_ATTEMPTS,
    Category,
    Diagnosis,
    can_retry,
    diagnose,
    policy_guard,
    recommend_action,
)
from simulator import RECOVERY_MODEL, RecoverySimulator, generate_transactions


def make_txn(**overrides):
    defaults = dict(
        transaction_id="TX_TEST",
        customer_id="CUST_1",
        amount=1000.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        failure_reason=FailureReason.CARD_NETWORK_ERROR,
        previous_failures=0,
        customer_payment_history=PaymentHistory.RELIABLE,
        time_since_failure_hours=1.0,
        subscription_flag=False,
        customer_response_probability=0.9,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


class PolicyTests(unittest.TestCase):
    def test_can_retry_within_bound(self):
        self.assertTrue(can_retry(0))
        self.assertTrue(can_retry(1))
        self.assertTrue(can_retry(2))

    def test_can_retry_at_bound_rejected(self):
        self.assertFalse(can_retry(MAX_AUTOMATED_ATTEMPTS))
        self.assertFalse(can_retry(MAX_AUTOMATED_ATTEMPTS + 1))

    def test_policy_guard_forces_escalate_at_max_attempts(self):
        txn = make_txn(attempt_count=MAX_AUTOMATED_ATTEMPTS)
        result = policy_guard(txn, ActionType.RETRY)
        self.assertEqual(result, ActionType.ESCALATE)

    def test_policy_guard_allows_when_under_limit(self):
        txn = make_txn(attempt_count=1)
        result = policy_guard(txn, ActionType.RETRY)
        self.assertEqual(result, ActionType.RETRY)

    def test_unknown_failure_escalates_directly(self):
        txn = make_txn(failure_reason=FailureReason.UNKNOWN)
        diagnosis = diagnose(txn)
        self.assertEqual(recommend_action(txn, diagnosis), ActionType.ESCALATE)

    def test_insufficient_funds_unreliable_history_sends_message(self):
        txn = make_txn(
            failure_reason=FailureReason.INSUFFICIENT_FUNDS,
            customer_payment_history=PaymentHistory.UNRELIABLE,
        )
        diagnosis = diagnose(txn)
        self.assertEqual(recommend_action(txn, diagnosis), ActionType.SEND_MESSAGE)

    def test_expired_card_never_retried(self):
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        diagnosis = diagnose(txn)
        self.assertEqual(recommend_action(txn, diagnosis), ActionType.SEND_MESSAGE)


class DiagnosisFeedsRecommendationTests(unittest.TestCase):
    """Fix 1: recommendation must actually consume the diagnosis, not
    just independently re-inspect the raw transaction."""

    def test_diagnose_returns_structured_category(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        result = diagnose(txn)
        self.assertIsInstance(result, Diagnosis)
        self.assertEqual(result.category, Category.TRANSIENT)
        self.assertTrue(result.explanation)

    def test_recommendation_follows_the_diagnosis_category_not_raw_reason(self):
        # Construct a transaction whose failure_reason would normally
        # map to LIQUIDITY, but hand recommend_action a fabricated
        # TRANSIENT diagnosis instead -- the recommendation must follow
        # the diagnosis it was given, proving it's actually consumed.
        txn = make_txn(failure_reason=FailureReason.INSUFFICIENT_FUNDS)
        fabricated = Diagnosis(category=Category.TRANSIENT, explanation="forced for test")
        self.assertEqual(recommend_action(txn, fabricated), ActionType.RETRY)

    def test_each_category_maps_to_a_distinct_default_action(self):
        txn = make_txn(customer_payment_history=PaymentHistory.RELIABLE)
        self.assertEqual(
            recommend_action(txn, Diagnosis(Category.TRANSIENT, "")), ActionType.RETRY
        )
        self.assertEqual(
            recommend_action(txn, Diagnosis(Category.LIQUIDITY, "")), ActionType.SCHEDULE_RETRY
        )
        self.assertEqual(
            recommend_action(txn, Diagnosis(Category.INSTRUMENT_ISSUE, "")), ActionType.SEND_MESSAGE
        )
        self.assertEqual(
            recommend_action(txn, Diagnosis(Category.UNCLEAR, "")), ActionType.ESCALATE
        )


class SimulatorAssumptionsTests(unittest.TestCase):
    """Fix 2: assumptions must be explicit and inspectable, not just
    bare numbers buried in execute()."""

    def test_assumptions_list_is_nonempty_and_documented(self):
        self.assertGreater(len(RECOVERY_MODEL), 0)
        for rule in RECOVERY_MODEL:
            self.assertTrue(rule.formula_text)
            self.assertTrue(rule.rationale)

    def test_every_action_type_has_at_least_one_assumption_note(self):
        actions_covered = {rule.action.value for rule in RECOVERY_MODEL}
        for expected in ("RETRY", "SCHEDULE_RETRY", "SEND_MESSAGE"):
            self.assertIn(expected, actions_covered)

    def test_execute_uses_the_same_probability_recovery_model_declares(self):
        """V1.1 fix: RECOVERY_MODEL must be the actual thing execute()
        reads, not documentation that can drift from it. Cross-check a
        few (action, failure_reason) combos by sampling many outcomes
        across many distinct transaction_ids (execute() is now
        deterministic per (seed, txn_id, attempt_count, action), so
        repeated calls on the SAME txn/attempt/action would return the
        same result every time -- varying the id is what produces
        independent samples) and confirming the empirical rate matches
        get_probability()."""
        import simulator as sim_mod
        expected_p = sim_mod.get_probability(
            make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR), ActionType.RETRY
        )
        sim = RecoverySimulator(seed=123)
        n = 4000
        successes = 0
        for i in range(n):
            txn = make_txn(transaction_id=f"TX_SAMPLE_{i}", failure_reason=FailureReason.CARD_NETWORK_ERROR)
            if sim.execute(txn, ActionType.RETRY).success:
                successes += 1
        empirical_p = successes / n
        self.assertAlmostEqual(empirical_p, expected_p, delta=0.03)

    def test_execute_is_independent_of_call_order_and_prior_draws(self):
        """The methodological bug this fixes: a sequential RNG stream's
        position (and therefore every subsequent draw) depended on how
        many actions were executed before it -- which differs by
        strategy. Confirm the SAME (txn_id, attempt_count, action)
        gives the SAME outcome whether it's the first call on a fresh
        simulator or the fourth call after unrelated draws."""
        seed = 999
        target = make_txn(transaction_id="TX_TARGET", failure_reason=FailureReason.BANK_DECLINE)

        fresh_sim = RecoverySimulator(seed=seed)
        baseline_result = fresh_sim.execute(target, ActionType.RETRY)

        busy_sim = RecoverySimulator(seed=seed)
        # Burn several unrelated draws on a different transaction first,
        # simulating a strategy that took more attempts before this one.
        noise_txn = make_txn(transaction_id="TX_NOISE", failure_reason=FailureReason.AUTH_FAILURE)
        for attempt in range(5):
            noise_txn.attempt_count = attempt
            busy_sim.execute(noise_txn, ActionType.SEND_MESSAGE)
        target_again = make_txn(transaction_id="TX_TARGET", failure_reason=FailureReason.BANK_DECLINE)
        busy_result = busy_sim.execute(target_again, ActionType.RETRY)

        self.assertEqual(baseline_result.success, busy_result.success)

    def test_prior_attempts_on_other_transactions_do_not_change_this_outcome(self):
        """Same guarantee, framed as the reviewer's exact scenario:
        strategy A burns many attempts on TX_1; strategy B burns none.
        TX_2's outcome must be identical either way."""
        seed = 555
        heavy_sim = RecoverySimulator(seed=seed)
        tx1 = make_txn(transaction_id="TX_1", failure_reason=FailureReason.BANK_DECLINE)
        for attempt in range(5):
            tx1.attempt_count = attempt
            heavy_sim.execute(tx1, ActionType.RETRY)
        tx2_heavy = make_txn(transaction_id="TX_2", failure_reason=FailureReason.CARD_NETWORK_ERROR)
        result_after_heavy = heavy_sim.execute(tx2_heavy, ActionType.RETRY)

        light_sim = RecoverySimulator(seed=seed)  # no prior calls at all
        tx2_light = make_txn(transaction_id="TX_2", failure_reason=FailureReason.CARD_NETWORK_ERROR)
        result_no_prior = light_sim.execute(tx2_light, ActionType.RETRY)

        self.assertEqual(result_after_heavy.success, result_no_prior.success)


class CategoryBreakdownTests(unittest.TestCase):
    """Fix 3: recovery metrics must be attributable to failure category,
    not just visible as one aggregate number."""

    def test_breakdown_splits_by_failure_reason(self):
        results = [
            CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1,
                       failure_reason=FailureReason.CARD_NETWORK_ERROR),
            CaseResult("TX_2", 200.0, TerminalState.ESCALATED, 0.0, 3,
                       failure_reason=FailureReason.INSUFFICIENT_FUNDS),
            CaseResult("TX_3", 300.0, TerminalState.RECOVERED, 300.0, 1,
                       failure_reason=FailureReason.CARD_NETWORK_ERROR),
        ]
        breakdown = compute_category_breakdown(results)
        self.assertIn("CARD_NETWORK_ERROR", breakdown)
        self.assertIn("INSUFFICIENT_FUNDS", breakdown)
        self.assertEqual(breakdown["CARD_NETWORK_ERROR"]["count"], 2)
        self.assertAlmostEqual(breakdown["CARD_NETWORK_ERROR"]["recovery_rate"], 1.0)
        self.assertAlmostEqual(breakdown["INSUFFICIENT_FUNDS"]["recovery_rate"], 0.0)

    def test_case_result_carries_failure_reason_from_orchestrator(self):
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        sim = RecoverySimulator(seed=1)
        result = run_case(txn, sim, diagnose, recommend_action)
        self.assertEqual(result.failure_reason, FailureReason.EXPIRED_CARD)


class MethodologyTests(unittest.TestCase):
    """Fix 4: both strategies must run on verifiably identical starting
    transactions for the comparison to be meaningful."""

    def test_deepcopy_preserves_identical_starting_state(self):
        base = generate_transactions(20, seed=5)
        txns_a = copy.deepcopy(base)
        txns_b = copy.deepcopy(base)
        for a, b in zip(txns_a, txns_b):
            self.assertEqual(a.transaction_id, b.transaction_id)
            self.assertEqual(a.failure_reason, b.failure_reason)
            self.assertEqual(a.customer_response_probability, b.customer_response_probability)
            self.assertEqual(a.attempt_count, 0)
            self.assertEqual(b.attempt_count, 0)

    def test_mutating_one_copy_does_not_affect_the_other(self):
        base = generate_transactions(5, seed=5)
        txns_a = copy.deepcopy(base)
        txns_b = copy.deepcopy(base)
        txns_a[0].attempt_count = 2
        self.assertEqual(txns_b[0].attempt_count, 0)


class RecoveryFlowTests(unittest.TestCase):
    def test_success_marks_recovered(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(txn, sim, diagnose, recommend_action)
        self.assertEqual(result.terminal_state, TerminalState.RECOVERED)
        self.assertEqual(result.amount_recovered, txn.amount)

    def test_repeated_failure_leads_to_escalation_not_infinite_loop(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, False, 0.0)
        result = run_case(txn, sim, diagnose, recommend_action)
        self.assertEqual(result.terminal_state, TerminalState.ESCALATED)
        self.assertEqual(result.total_attempts, MAX_AUTOMATED_ATTEMPTS)
        self.assertLessEqual(len(result.audit_trail), 25)


class SafetyTests(unittest.TestCase):
    def test_invalid_action_rejected_by_simulator(self):
        txn = make_txn()
        sim = RecoverySimulator(seed=1)
        with self.assertRaises(ValueError):
            sim.execute(txn, "NOT_A_REAL_ACTION")

    def test_batch_is_reproducible_given_same_seed(self):
        batch1 = generate_transactions(50, seed=7)
        batch2 = generate_transactions(50, seed=7)
        self.assertEqual(
            [t.transaction_id for t in batch1], [t.transaction_id for t in batch2]
        )
        self.assertEqual(
            [t.failure_reason for t in batch1], [t.failure_reason for t in batch2]
        )
        self.assertEqual(
            [round(t.amount, 2) for t in batch1], [round(t.amount, 2) for t in batch2]
        )

    def test_no_run_exceeds_transition_ceiling(self):
        txn = make_txn()
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, False, 0.0)
        result = run_case(txn, sim, diagnose, recommend_action)
        self.assertIn(result.terminal_state, (TerminalState.ESCALATED, TerminalState.STOPPED))


class MetricsTests(unittest.TestCase):
    def test_recovery_rate_and_counts(self):
        results = [
            CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1,
                       failure_reason=FailureReason.CARD_NETWORK_ERROR),
            CaseResult("TX_2", 200.0, TerminalState.ESCALATED, 0.0, 3,
                       failure_reason=FailureReason.UNKNOWN),
            CaseResult("TX_3", 300.0, TerminalState.RECOVERED, 300.0, 2,
                       failure_reason=FailureReason.BANK_DECLINE),
        ]
        m = compute_metrics("test", results)
        self.assertEqual(m.total_transactions, 3)
        self.assertEqual(m.total_revenue_at_risk, 600.0)
        self.assertEqual(m.total_recovered, 400.0)
        self.assertAlmostEqual(m.recovery_rate, 400.0 / 600.0)
        self.assertEqual(m.recovered_count, 2)
        self.assertEqual(m.escalated_count, 1)
        self.assertAlmostEqual(m.avg_attempts_to_recovery, 1.5)

    def test_empty_batch_does_not_divide_by_zero(self):
        m = compute_metrics("empty", [])
        self.assertEqual(m.recovery_rate, 0.0)
        self.assertEqual(m.total_transactions, 0)

    def test_unrecovered_revenue_and_rates(self):
        """Phase 2: unrecovered_revenue, escalation_rate, stop_rate."""
        results = [
            CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1,
                       failure_reason=FailureReason.CARD_NETWORK_ERROR),
            CaseResult("TX_2", 200.0, TerminalState.ESCALATED, 0.0, 3,
                       failure_reason=FailureReason.UNKNOWN),
            CaseResult("TX_3", 300.0, TerminalState.STOPPED, 0.0, 1,
                       failure_reason=FailureReason.BANK_DECLINE),
            CaseResult("TX_4", 400.0, TerminalState.RECOVERED, 400.0, 1,
                       failure_reason=FailureReason.BANK_DECLINE),
        ]
        m = compute_metrics("test", results)
        self.assertAlmostEqual(m.unrecovered_revenue, 500.0)  # 1000 at risk - 500 recovered
        self.assertAlmostEqual(m.escalation_rate, 0.25)  # 1/4
        self.assertAlmostEqual(m.stop_rate, 0.25)  # 1/4


class AgreementTests(unittest.TestCase):
    """Phase 2: batch-level AI/rule first-action agreement and
    divergence, computable straight from the benchmark output rather
    than only ad hoc in run_pilot.py."""

    def test_agreement_and_divergence_counts(self):
        results_b = [
            CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1,
                       first_recommended_action=ActionType.RETRY),
            CaseResult("TX_2", 200.0, TerminalState.RECOVERED, 200.0, 1,
                       first_recommended_action=ActionType.SEND_MESSAGE),
        ]
        results_c = [
            CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1,
                       first_recommended_action=ActionType.RETRY),          # agrees
            CaseResult("TX_2", 200.0, TerminalState.RECOVERED, 200.0, 1,
                       first_recommended_action=ActionType.SCHEDULE_RETRY),  # diverges
        ]
        agr = compute_agreement(results_b, results_c)
        self.assertEqual(agr["total"], 2)
        self.assertEqual(agr["agree"], 1)
        self.assertEqual(agr["disagree"], 1)
        self.assertAlmostEqual(agr["agreement_rate"], 0.5)
        self.assertAlmostEqual(agr["divergence_rate"], 0.5)

    def test_agreement_requires_index_aligned_transaction_ids(self):
        results_b = [CaseResult("TX_1", 100.0, TerminalState.RECOVERED, 100.0, 1)]
        results_c = [CaseResult("TX_DIFFERENT", 100.0, TerminalState.RECOVERED, 100.0, 1)]
        with self.assertRaises(AssertionError):
            compute_agreement(results_b, results_c)

    def test_agreement_empty_batches(self):
        agr = compute_agreement([], [])
        self.assertEqual(agr["total"], 0)
        self.assertEqual(agr["agreement_rate"], 0.0)


class BaselineTests(unittest.TestCase):
    def test_fixed_retry_ignores_failure_reason_and_diagnosis(self):
        for reason in FailureReason:
            txn = make_txn(failure_reason=reason)
            diagnosis = fixed_retry_diagnose(txn)
            self.assertEqual(fixed_retry_recommend(txn, diagnosis), ActionType.RETRY)
        self.assertTrue(fixed_retry_diagnose(make_txn()).explanation)


class ReportRenderingTests(unittest.TestCase):
    """Phase 3: report.py must correctly distinguish ALLOWED vs
    REJECTED policy-check stamps regardless of which of the two casing
    variants the orchestrator's audit log uses ('rejected ... MAX_ATTEMPTS'
    lowercase vs 'REJECTED ...' uppercase for illegitimate actions) --
    a real bug this test would have caught."""

    def test_stamp_detects_lowercase_rejected(self):
        import report
        case = {
            "transaction_id": "TX_1", "amount": 100.0, "failure_reason": "BANK_DECLINE",
            "terminal_state": "ESCALATED", "amount_recovered": 0.0, "total_attempts": 3,
            "ai_calls": 1, "audit_trail": [
                {"step": 0, "state": "POLICY_CHECK", "detail": "rejected RETRY (MAX_ATTEMPTS); forced ESCALATE"},
            ],
        }
        html_out = report.render_case_trail(case, "test")
        self.assertIn('stamp-reject', html_out)
        self.assertIn('REJECTED', html_out)
        self.assertNotIn('stamp-allow', html_out)

    def test_stamp_detects_uppercase_rejected(self):
        import report
        case = {
            "transaction_id": "TX_1", "amount": 100.0, "failure_reason": "EXPIRED_CARD",
            "terminal_state": "ESCALATED", "amount_recovered": 0.0, "total_attempts": 0,
            "ai_calls": 1, "audit_trail": [
                {"step": 0, "state": "POLICY_CHECK", "detail": "REJECTED RETRY: not a legitimate action"},
            ],
        }
        html_out = report.render_case_trail(case, "test")
        self.assertIn('stamp-reject', html_out)

    def test_stamp_shows_allowed_when_not_rejected(self):
        import report
        case = {
            "transaction_id": "TX_1", "amount": 100.0, "failure_reason": "BANK_DECLINE",
            "terminal_state": "RECOVERED", "amount_recovered": 100.0, "total_attempts": 1,
            "ai_calls": 1, "audit_trail": [
                {"step": 0, "state": "POLICY_CHECK", "detail": "allowed RETRY (attempt 1/3)"},
            ],
        }
        html_out = report.render_case_trail(case, "test")
        self.assertIn('stamp-allow', html_out)
        self.assertIn('ALLOWED', html_out)
        self.assertNotIn('stamp-reject', html_out)

    def test_none_case_renders_empty(self):
        import report
        self.assertEqual(report.render_case_trail(None, "test"), "")


if __name__ == "__main__":
    unittest.main()
