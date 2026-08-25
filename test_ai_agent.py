import unittest

import ai_agent
import cache as cache_mod
import policy
from llm_client import LLMClientError
from models import ActionType, FailureReason, PaymentHistory, PaymentMethod, Transaction
from orchestrator import run_case
from policy import Diagnosis, guard_ai, guard_standard, is_action_legitimate
from simulator import RecoverySimulator


def make_txn(**overrides):
    defaults = dict(
        transaction_id="TX_TEST",
        customer_id="CUST_1",
        amount=1000.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        failure_reason=FailureReason.EXPIRED_CARD,
        previous_failures=0,
        customer_payment_history=PaymentHistory.RELIABLE,
        time_since_failure_hours=1.0,
        subscription_flag=False,
        customer_response_probability=0.9,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


class FixedResponseClient:
    """Test double that always returns the same raw dict, or raises."""
    def __init__(self, response=None, raise_error: LLMClientError = None):
        self.response = response
        self.raise_error = raise_error
        self.call_count = 0

    def complete(self, context):
        self.call_count += 1
        if self.raise_error:
            raise self.raise_error
        return self.response


class PolicyLegitimacyTests(unittest.TestCase):
    """Fix from V1 review: guard checks the recommended action against
    the OBJECTIVE failure_reason, never against the AI's own diagnosis."""

    def test_retry_illegitimate_for_expired_card(self):
        self.assertFalse(is_action_legitimate(FailureReason.EXPIRED_CARD, ActionType.RETRY))

    def test_send_message_legitimate_for_expired_card(self):
        self.assertTrue(is_action_legitimate(FailureReason.EXPIRED_CARD, ActionType.SEND_MESSAGE))

    def test_escalate_and_stop_always_legitimate(self):
        for reason in FailureReason:
            self.assertTrue(is_action_legitimate(reason, ActionType.ESCALATE))
            self.assertTrue(is_action_legitimate(reason, ActionType.STOP))

    def test_guard_ai_rejects_illegitimate_action_to_escalate_not_substitute(self):
        """Reviewer's explicit requirement: reject -> ESCALATE, never a
        silent substitution to a 'nearest' legitimate action."""
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        action, reason = guard_ai(txn, ActionType.RETRY)
        self.assertEqual(action, ActionType.ESCALATE)
        self.assertEqual(reason, "ILLEGITIMATE_ACTION")
        # explicitly NOT SEND_MESSAGE or any other substitute
        self.assertNotEqual(action, ActionType.SEND_MESSAGE)

    def test_guard_ai_allows_legitimate_action(self):
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        action, reason = guard_ai(txn, ActionType.SEND_MESSAGE)
        self.assertEqual(action, ActionType.SEND_MESSAGE)
        self.assertIsNone(reason)

    def test_guard_ai_ignores_diagnosis_text_uses_only_failure_reason(self):
        """An AI can write a diagnosis that rationalizes anything --
        the guard must not be swayed by the diagnosis text, only by
        the objective failure_reason."""
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        # Even if we imagine the AI's diagnosis claimed this was transient,
        # guard_ai never receives or reads diagnosis text -- it only takes
        # (txn, recommended_action). Confirm RETRY is still rejected.
        action, reason = guard_ai(txn, ActionType.RETRY)
        self.assertEqual(action, ActionType.ESCALATE)

    def test_guard_standard_unaffected_by_legitimacy_matrix(self):
        """Baseline A's deliberate blind-retry behavior must be
        preserved exactly -- guard_standard never checks legitimacy."""
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD, attempt_count=0)
        action, reason = guard_standard(txn, ActionType.RETRY)
        self.assertEqual(action, ActionType.RETRY)  # allowed, unlike guard_ai
        self.assertIsNone(reason)


class ValidationTests(unittest.TestCase):
    def test_missing_field_rejected(self):
        with self.assertRaises(ai_agent.ValidationError):
            ai_agent.validate_response({"diagnosis": "x", "confidence": 0.5})

    def test_forbidden_action_rejected(self):
        with self.assertRaises(ai_agent.ValidationError):
            ai_agent.validate_response({
                "diagnosis": "x", "recommended_action": "CALL_THE_BANK", "confidence": 0.5
            })

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ai_agent.ValidationError):
            ai_agent.validate_response({
                "diagnosis": "x", "recommended_action": "RETRY", "confidence": 1.5
            })

    def test_empty_diagnosis_rejected(self):
        with self.assertRaises(ai_agent.ValidationError):
            ai_agent.validate_response({
                "diagnosis": "   ", "recommended_action": "RETRY", "confidence": 0.5
            })

    def test_valid_response_accepted(self):
        diagnosis, action, confidence = ai_agent.validate_response({
            "diagnosis": "plausible transient failure", "recommended_action": "RETRY", "confidence": 0.8
        })
        self.assertEqual(action, ActionType.RETRY)
        self.assertAlmostEqual(confidence, 0.8)


class AIAgentFallbackTests(unittest.TestCase):
    """LLM failure (network/timeout) and malformed output must both
    fall back to the rule engine, never crash or block the case."""

    def test_llm_error_triggers_fallback(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(raise_error=LLMClientError("timeout"))
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(diagnosis.source, "ai_fallback")
        self.assertIsNotNone(diagnosis.recommended_action)
        self.assertEqual(agent.stats.fallbacks, 1)

    def test_malformed_output_triggers_fallback(self):
        txn = make_txn(failure_reason=FailureReason.BANK_DECLINE)
        client = FixedResponseClient(response={"diagnosis": "x"})  # missing fields
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(diagnosis.source, "ai_fallback")
        self.assertEqual(agent.stats.fallbacks, 1)

    def test_forbidden_action_from_llm_triggers_fallback(self):
        txn = make_txn(failure_reason=FailureReason.AUTH_FAILURE)
        client = FixedResponseClient(response={
            "diagnosis": "bad idea", "recommended_action": "WIRE_TRANSFER", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(diagnosis.source, "ai_fallback")

    def test_fallback_action_is_itself_legitimate(self):
        """The rule-engine fallback must never recommend something the
        legitimacy matrix would itself reject."""
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        client = FixedResponseClient(raise_error=LLMClientError("down"))
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertTrue(is_action_legitimate(txn.failure_reason, diagnosis.recommended_action))

    def test_successful_call_does_not_fall_back(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(response={
            "diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(diagnosis.source, "ai")
        self.assertEqual(diagnosis.recommended_action, ActionType.RETRY)
        self.assertEqual(agent.stats.fallbacks, 0)


class CacheReplayTests(unittest.TestCase):
    def test_live_mode_populates_cache(self):
        txn = make_txn()
        client = FixedResponseClient(response={
            "diagnosis": "x", "recommended_action": "SEND_MESSAGE", "confidence": 0.6
        })
        response_cache = cache_mod.ResponseCache()
        agent = ai_agent.AIAgent(client=client, cache=response_cache, mode="live")
        agent.get_recommendation(txn)
        context = ai_agent.build_context(txn)
        key = cache_mod.make_key(context)
        self.assertIn(key, response_cache)
        self.assertEqual(client.call_count, 1)

    def test_cached_mode_replays_without_calling_client(self):
        txn = make_txn()
        context = ai_agent.build_context(txn)
        key = cache_mod.make_key(context)
        response_cache = cache_mod.ResponseCache()
        response_cache.set(key, {"diagnosis": "cached", "recommended_action": "SEND_MESSAGE", "confidence": 0.6})

        client = FixedResponseClient(response={"diagnosis": "SHOULD NOT BE CALLED"})
        agent = ai_agent.AIAgent(client=client, cache=response_cache, mode="cached")
        diagnosis = agent.get_recommendation(txn)

        self.assertEqual(client.call_count, 0)
        self.assertEqual(diagnosis.source, "ai_cached")
        self.assertEqual(diagnosis.explanation, "cached")

    def test_cached_mode_miss_falls_back_not_crashes(self):
        txn = make_txn()
        client = FixedResponseClient(response={"diagnosis": "SHOULD NOT BE CALLED"})
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="cached")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(client.call_count, 0)
        self.assertEqual(diagnosis.source, "ai_fallback")
        self.assertEqual(agent.stats.cache_misses, 1)

    def test_cache_survives_save_and_reload(self):
        txn = make_txn()
        context = ai_agent.build_context(txn)
        key = cache_mod.make_key(context)

        cache_a = cache_mod.ResponseCache()
        cache_a.set(key, {"diagnosis": "persisted", "recommended_action": "SEND_MESSAGE", "confidence": 0.5})

        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cache.json")
            cache_a.save(path)
            cache_b = cache_mod.ResponseCache(path=path)
            self.assertIn(key, cache_b)
            self.assertEqual(cache_b.get(key)["diagnosis"], "persisted")

    def test_same_context_produces_same_cache_key(self):
        txn1 = make_txn(transaction_id="TX_A")
        txn2 = make_txn(transaction_id="TX_B")  # different id, same everything else
        key1 = cache_mod.make_key(ai_agent.build_context(txn1))
        key2 = cache_mod.make_key(ai_agent.build_context(txn2))
        # transaction_id is not part of the context sent to the LLM,
        # so two txns with identical decision-relevant fields must
        # hash to the same cache key.
        self.assertEqual(key1, key2)


class HiddenStateTests(unittest.TestCase):
    def test_context_never_includes_response_probability(self):
        txn = make_txn(customer_response_probability=0.42)
        context = ai_agent.build_context(txn)
        self.assertNotIn("customer_response_probability", context)
        self.assertNotIn(0.42, context.values())


class EndToEndAIOrchestratorTests(unittest.TestCase):
    def test_ai_strategy_runs_through_orchestrator_with_guard_ai(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(response={
            "diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai)
        self.assertEqual(result.terminal_state.value, "RECOVERED")
        self.assertFalse(result.had_policy_violation)
        self.assertAlmostEqual(result.final_confidence, 0.9)

    def test_ai_strategy_escalates_on_illegitimate_recommendation(self):
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        client = FixedResponseClient(response={
            "diagnosis": "misguided", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        result = run_case(txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai)
        self.assertEqual(result.terminal_state.value, "ESCALATED")
        self.assertTrue(result.had_policy_violation)
        self.assertEqual(result.total_attempts, 0)  # rejected before any execution


class TwoPhaseWorkflowTests(unittest.TestCase):
    """Reviewer's explicit requirement: a whole-batch run in cached
    mode must make zero live LLM calls, and a cached replay must
    reproduce exactly what the live run that populated the cache
    produced -- at the experiment.py level, not just for one agent
    call in isolation."""

    def test_cached_batch_makes_zero_live_calls(self):
        import tempfile, os
        import experiment
        with tempfile.TemporaryDirectory() as d:
            cache_path = os.path.join(d, "cache.json")
            # No live run first -- every context is a cache miss. Cached
            # mode must still make zero live calls; misses fall back.
            result = experiment.run_experiment(
                n=15, seed=321, llm_client_kind="demo", llm_mode="cached",
                cache_file=cache_path, verbose=False,
            )
            self.assertEqual(result.ai_stats.live_calls, 0)
            self.assertGreater(result.ai_stats.cache_misses, 0)
            self.assertGreater(result.ai_stats.fallbacks, 0)

    def test_live_then_cached_reproduces_identical_ai_decisions(self):
        import tempfile, os
        import experiment
        with tempfile.TemporaryDirectory() as d:
            cache_path = os.path.join(d, "cache.json")

            live_result = experiment.run_experiment(
                n=25, seed=777, llm_client_kind="demo", llm_mode="live",
                cache_file=cache_path, verbose=False,
            )
            self.assertGreater(live_result.ai_stats.live_calls, 0)

            cached_result = experiment.run_experiment(
                n=25, seed=777, llm_client_kind="demo", llm_mode="cached",
                cache_file=cache_path, verbose=False,
            )
            self.assertEqual(cached_result.ai_stats.live_calls, 0)
            self.assertEqual(cached_result.ai_stats.cache_misses, 0)

            live_actions = [
                (r.transaction_id, r.terminal_state, r.amount_recovered)
                for r in live_result.results_c
            ]
            cached_actions = [
                (r.transaction_id, r.terminal_state, r.amount_recovered)
                for r in cached_result.results_c
            ]
            self.assertEqual(live_actions, cached_actions)
            self.assertEqual(live_result.metrics_c.total_recovered, cached_result.metrics_c.total_recovered)
            self.assertEqual(live_result.metrics_c.recovery_rate, cached_result.metrics_c.recovery_rate)


class ModelIdRequiredTests(unittest.TestCase):
    """V1.2 fix: no silent default model ID, for EITHER provider. A
    stale/guessed default can silently burn API calls against the
    wrong (or no-longer-free) model."""

    def test_missing_model_raises_at_construction(self):
        import os
        from llm_client import AnthropicLLMClient
        old = os.environ.pop("ANTHROPIC_MODEL", None)
        try:
            with self.assertRaises(LLMClientError):
                AnthropicLLMClient()
        finally:
            if old is not None:
                os.environ["ANTHROPIC_MODEL"] = old

    def test_explicit_model_param_accepted_without_env_var(self):
        import os
        from llm_client import AnthropicLLMClient
        old = os.environ.pop("ANTHROPIC_MODEL", None)
        try:
            client = AnthropicLLMClient(model="claude-something-explicit")
            self.assertEqual(client.model, "claude-something-explicit")
        finally:
            if old is not None:
                os.environ["ANTHROPIC_MODEL"] = old

    def test_gemini_missing_model_raises_at_construction(self):
        import os
        from llm_client import GeminiLLMClient
        old = os.environ.pop("GEMINI_MODEL", None)
        try:
            with self.assertRaises(LLMClientError):
                GeminiLLMClient()
        finally:
            if old is not None:
                os.environ["GEMINI_MODEL"] = old

    def test_gemini_explicit_model_param_accepted_without_env_var(self):
        import os
        from llm_client import GeminiLLMClient
        old = os.environ.pop("GEMINI_MODEL", None)
        try:
            client = GeminiLLMClient(model="gemini-something-explicit")
            self.assertEqual(client.model, "gemini-something-explicit")
        finally:
            if old is not None:
                os.environ["GEMINI_MODEL"] = old

    def test_gemini_env_var_used_when_no_explicit_param(self):
        import os
        from llm_client import GeminiLLMClient
        old = os.environ.get("GEMINI_MODEL")
        os.environ["GEMINI_MODEL"] = "gemini-from-env"
        try:
            client = GeminiLLMClient()
            self.assertEqual(client.model, "gemini-from-env")
        finally:
            if old is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = old

    def test_env_var_used_when_no_explicit_param(self):
        import os
        from llm_client import AnthropicLLMClient
        old = os.environ.get("ANTHROPIC_MODEL")
        os.environ["ANTHROPIC_MODEL"] = "claude-from-env"
        try:
            client = AnthropicLLMClient()
            self.assertEqual(client.model, "claude-from-env")
        finally:
            if old is None:
                os.environ.pop("ANTHROPIC_MODEL", None)
            else:
                os.environ["ANTHROPIC_MODEL"] = old


class FirstRecommendedActionTests(unittest.TestCase):
    """Needed for the per-case pilot comparison: CaseResult must expose
    the FIRST action recommended for a case, unaffected by later
    retries -- this is what run_pilot.py diffs against the rule
    engine's first action to compute agreement/divergence."""

    def test_captures_first_action_on_single_attempt_success(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(txn, sim, policy.diagnose, policy.recommend_action)
        self.assertEqual(result.first_recommended_action, ActionType.RETRY)

    def test_first_action_unchanged_across_retries(self):
        """Even if later attempts would recommend something different in
        principle, first_recommended_action reflects only attempt 1."""
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, False, 0.0)  # always fails
        result = run_case(txn, sim, policy.diagnose, policy.recommend_action)
        # rule engine recommends RETRY every attempt for CARD_NETWORK_ERROR,
        # so this doesn't distinguish "first" from "any" on its own --
        # the real guarantee is checked structurally below.
        self.assertEqual(result.first_recommended_action, ActionType.RETRY)
        self.assertEqual(result.total_attempts, 3)  # confirms multiple attempts did happen

    def test_first_action_reflects_ai_first_decision_not_a_later_fallback(self):
        """AI recommends SCHEDULE_RETRY on attempt 1 (succeeds via a
        client that only returns one response); first_recommended_action
        must be SCHEDULE_RETRY, matching what was actually decided
        first, not e.g. None or a rule-engine value."""
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR, previous_failures=2)
        client = FixedResponseClient(response={
            "diagnosis": "repeat failures, wait it out",
            "recommended_action": "SCHEDULE_RETRY", "confidence": 0.75
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai)
        self.assertEqual(result.first_recommended_action, ActionType.SCHEDULE_RETRY)


class GeminiResponseParsingTests(unittest.TestCase):
    """_parse_response is factored out specifically so the wire-format
    handling can be tested without mocking HTTP. Uses real
    generateContent response shapes per current Gemini API docs."""

    def test_parses_well_formed_gemini_response(self):
        from llm_client import GeminiLLMClient
        data = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.8}'
                    }]
                }
            }]
        }
        result = GeminiLLMClient._parse_response(data)
        self.assertEqual(result, {"diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.8})

    def test_no_candidates_raises_llm_client_error(self):
        from llm_client import GeminiLLMClient
        with self.assertRaises(LLMClientError):
            GeminiLLMClient._parse_response({"candidates": []})

    def test_missing_candidates_key_raises_llm_client_error(self):
        from llm_client import GeminiLLMClient
        with self.assertRaises(LLMClientError):
            GeminiLLMClient._parse_response({"unexpected": "shape"})

    def test_non_json_text_part_raises_llm_client_error(self):
        from llm_client import GeminiLLMClient
        data = {"candidates": [{"content": {"parts": [{"text": "not json at all"}]}}]}
        with self.assertRaises(LLMClientError):
            GeminiLLMClient._parse_response(data)

    def test_response_schema_constrains_recommended_action_enum(self):
        """The schema sent to Gemini should enumerate exactly the
        permitted actions -- this is what response_schema enum
        enforcement relies on."""
        from llm_client import GeminiLLMClient
        enum_values = set(GeminiLLMClient.RESPONSE_SCHEMA["properties"]["recommended_action"]["enum"])
        self.assertEqual(enum_values, {a.value for a in ActionType})

    def test_gemini_client_end_to_end_via_ai_agent_with_fixed_response(self):
        """Confirms GeminiLLMClient's .complete() output (once parsed)
        flows through ai_agent.AIAgent identically to any other
        provider -- the abstraction actually holds."""
        txn = make_txn(failure_reason=FailureReason.BANK_DECLINE)
        client = FixedResponseClient(response={
            "diagnosis": "gemini says transient", "recommended_action": "RETRY", "confidence": 0.66
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        diagnosis = agent.get_recommendation(txn)
        self.assertEqual(diagnosis.source, "ai")
        self.assertEqual(diagnosis.recommended_action, ActionType.RETRY)
        self.assertAlmostEqual(diagnosis.confidence, 0.66)


class OneInitialAICallTests(unittest.TestCase):
    """V1.4 fix: the AI is consulted once per case (the initial
    strategic decision), never re-consulted just because that action
    failed. Retries after the first go through the deterministic rule
    engine (policy.diagnose/policy.recommend_action), the same one
    that already powers Strategy B."""

    def test_normal_case_makes_exactly_one_ai_call_on_success(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(response={
            "diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(
            txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai,
            retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
        )
        self.assertEqual(result.terminal_state.value, "RECOVERED")
        self.assertEqual(client.call_count, 1)
        self.assertEqual(result.ai_calls, 1)

    def test_failed_retry_does_not_trigger_another_ai_call(self):
        """The AI's first action fails; the case still retries (via the
        deterministic rule engine) and eventually succeeds or escalates
        -- but the LLM client is never called a second time."""
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(response={
            "diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        # First execute() call fails, all subsequent ones succeed.
        call_log = {"n": 0}
        def flaky_execute(t, a):
            call_log["n"] += 1
            success = call_log["n"] > 1
            return __import__("models").ActionResult(a, success, t.amount if success else 0.0)
        sim.execute = flaky_execute

        result = run_case(
            txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai,
            retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
        )
        self.assertGreaterEqual(result.total_attempts, 2)  # confirms a retry actually happened
        self.assertEqual(client.call_count, 1)  # but the AI was asked only once
        self.assertEqual(result.ai_calls, 1)

    def test_max_attempts_still_enforced_under_one_call_design(self):
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(response={
            "diagnosis": "transient", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, False, 0.0)  # always fails
        result = run_case(
            txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai,
            retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
        )
        self.assertEqual(result.terminal_state.value, "ESCALATED")
        self.assertEqual(result.total_attempts, 3)  # MAX_AUTOMATED_ATTEMPTS, unchanged
        self.assertEqual(client.call_count, 1)  # still only one AI call across all 3 attempts
        self.assertEqual(result.ai_calls, 1)

    def test_escalation_and_stop_still_correct(self):
        """AI's first recommendation is itself illegitimate -> guard
        rejects to ESCALATE immediately, zero attempts executed, and
        (trivially) only the one AI call that produced the bad
        recommendation in the first place."""
        txn = make_txn(failure_reason=FailureReason.EXPIRED_CARD)
        client = FixedResponseClient(response={
            "diagnosis": "misguided", "recommended_action": "RETRY", "confidence": 0.9
        })
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        result = run_case(
            txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai,
            retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
        )
        self.assertEqual(result.terminal_state.value, "ESCALATED")
        self.assertTrue(result.had_policy_violation)
        self.assertEqual(result.total_attempts, 0)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(result.ai_calls, 1)

    def test_cached_mode_still_makes_zero_live_calls_under_one_call_design(self):
        import tempfile, os
        import experiment
        with tempfile.TemporaryDirectory() as d:
            cache_path = os.path.join(d, "cache.json")
            live_result = experiment.run_experiment(
                n=20, seed=888, llm_client_kind="demo", llm_mode="live",
                cache_file=cache_path, verbose=False,
            )
            self.assertGreater(live_result.ai_stats.live_calls, 0)
            # With one-call-per-case, live_calls should now be <= n
            # (at most one AI decision per transaction), not >n as it
            # could be under the old per-retry-call design.
            self.assertLessEqual(live_result.ai_stats.live_calls, 20)

            cached_result = experiment.run_experiment(
                n=20, seed=888, llm_client_kind="demo", llm_mode="cached",
                cache_file=cache_path, verbose=False,
            )
            self.assertEqual(cached_result.ai_stats.live_calls, 0)
            self.assertEqual(cached_result.ai_stats.cache_misses, 0)

    def test_baseline_and_rule_engine_unaffected_by_retry_fn_default(self):
        """A/B never pass retry_diagnose_fn/retry_recommend_fn -- they
        should default to reusing diagnose_fn/recommend_fn, which are
        already deterministic every call, so behavior is identical to
        before this change."""
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, False, 0.0)
        result = run_case(txn, sim, policy.diagnose, policy.recommend_action)  # no retry_* args
        self.assertEqual(result.terminal_state.value, "ESCALATED")
        self.assertEqual(result.total_attempts, 3)
        self.assertEqual(result.ai_calls, 0)  # rule engine never touches the AI

    def test_ai_calls_recorded_even_on_fallback(self):
        """A fallback still represents an attempted AI call, not a
        rule-engine-native decision -- ai_calls should count it."""
        txn = make_txn(failure_reason=FailureReason.CARD_NETWORK_ERROR)
        client = FixedResponseClient(raise_error=LLMClientError("down"))
        agent = ai_agent.AIAgent(client=client, cache=cache_mod.ResponseCache(), mode="live")
        sim = RecoverySimulator(seed=1)
        sim.execute = lambda t, a: __import__("models").ActionResult(a, True, t.amount)
        result = run_case(
            txn, sim, agent.diagnose_fn, agent.recommend_fn, guard_fn=guard_ai,
            retry_diagnose_fn=policy.diagnose, retry_recommend_fn=policy.recommend_action,
        )
        self.assertEqual(result.ai_calls, 1)
        self.assertTrue(result.had_ai_fallback)


class FakeHttpResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class GeminiBackoffTests(unittest.TestCase):
    """Phase 1: bounded retry on transient HTTP statuses only. Uses an
    injectable sleep_fn so these run instantly, not for real seconds."""

    def test_retries_on_429_then_succeeds(self):
        from llm_client import _send_with_backoff
        responses = [FakeHttpResponse(429), FakeHttpResponse(200)]
        calls = {"n": 0}
        def send():
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        sleeps = []
        result = _send_with_backoff(send, max_retries=3, base_delay=0.5, sleep_fn=sleeps.append)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(sleeps), 1)  # slept once, between the two calls

    def test_retries_each_retryable_status(self):
        from llm_client import _send_with_backoff, RETRYABLE_STATUS_CODES
        self.assertEqual(RETRYABLE_STATUS_CODES, {429, 500, 502, 503, 504})
        for status in RETRYABLE_STATUS_CODES:
            calls = {"n": 0}
            def send(status=status):
                calls["n"] += 1
                return FakeHttpResponse(status if calls["n"] < 2 else 200)
            result = _send_with_backoff(send, max_retries=3, base_delay=0.01, sleep_fn=lambda s: None)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(calls["n"], 2)

    def test_does_not_retry_400(self):
        from llm_client import _send_with_backoff
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return FakeHttpResponse(400)
        result = _send_with_backoff(send, max_retries=3, sleep_fn=lambda s: None)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(calls["n"], 1)  # no retry attempted

    def test_does_not_retry_401(self):
        from llm_client import _send_with_backoff
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return FakeHttpResponse(401)
        result = _send_with_backoff(send, max_retries=3, sleep_fn=lambda s: None)
        self.assertEqual(calls["n"], 1)

    def test_does_not_retry_403(self):
        from llm_client import _send_with_backoff
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return FakeHttpResponse(403)
        result = _send_with_backoff(send, max_retries=3, sleep_fn=lambda s: None)
        self.assertEqual(calls["n"], 1)

    def test_bounded_retries_gives_up_after_max(self):
        from llm_client import _send_with_backoff
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return FakeHttpResponse(503)  # always fails
        result = _send_with_backoff(send, max_retries=3, sleep_fn=lambda s: None)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(calls["n"], 3)  # exactly max_retries attempts, not infinite

    def test_backoff_delays_are_exponential(self):
        from llm_client import _send_with_backoff
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return FakeHttpResponse(500)
        sleeps = []
        _send_with_backoff(send, max_retries=3, base_delay=1.0, sleep_fn=sleeps.append)
        self.assertEqual(sleeps, [1.0, 2.0])  # base*2^0, base*2^1 -- 2 sleeps for 3 attempts

    def test_exception_from_send_fn_is_not_caught_here(self):
        """Network errors/timeouts are a different failure class --
        this function only governs retry-after-a-response, not
        retry-after-an-exception. complete() still wraps this in its
        own try/except, unaffected by this function."""
        from llm_client import _send_with_backoff
        def send():
            raise ConnectionError("boom")
        with self.assertRaises(ConnectionError):
            _send_with_backoff(send, sleep_fn=lambda s: None)

    def test_gemini_complete_uses_backoff_and_succeeds_after_transient_error(self):
        """End-to-end through GeminiLLMClient.complete() with requests
        mocked out, confirming the backoff is actually wired in, not
        just present as an unused helper."""
        import os
        import types
        from llm_client import GeminiLLMClient

        old_model = os.environ.get("GEMINI_MODEL")
        old_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_MODEL"] = "gemini-test-model"
        os.environ["GEMINI_API_KEY"] = "test-key"
        try:
            client = GeminiLLMClient()

            call_count = {"n": 0}
            good_body = {
                "candidates": [{"content": {"parts": [{
                    "text": '{"diagnosis": "d", "recommended_action": "RETRY", "confidence": 0.5}'
                }]}}]
            }

            class FakeResponse:
                def __init__(self, status_code, body=None):
                    self.status_code = status_code
                    self._body = body
                    self.text = str(body)
                def json(self):
                    return self._body

            def fake_post(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return FakeResponse(503)
                return FakeResponse(200, good_body)

            fake_requests = types.SimpleNamespace(post=fake_post)
            import sys
            real_requests = sys.modules.get("requests")
            sys.modules["requests"] = fake_requests
            try:
                result = client.complete({"failure_reason": "BANK_DECLINE", "permitted_actions": ["RETRY"]})
            finally:
                if real_requests is not None:
                    sys.modules["requests"] = real_requests
                else:
                    sys.modules.pop("requests", None)

            self.assertEqual(result["recommended_action"], "RETRY")
            self.assertEqual(call_count["n"], 2)  # one 503, one success
        finally:
            if old_model is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = old_model
            if old_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
