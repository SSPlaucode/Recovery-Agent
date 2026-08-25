"""
Version 1 AI decision layer.

    Transaction -> build_context() -> LLM (cached or live) -> validate()
        -> Diagnosis(source="ai"/"ai_cached", recommended_action=...)
        -> [any failure at any step] -> fallback to policy.diagnose()/
           policy.recommend_action(), Diagnosis(source="ai_fallback")

The AI never sees customer_response_probability (hidden simulator
state, unchanged from V0). The AI's diagnosis and action come from a
SINGLE call per FSM step -- ai_diagnose_fn does the work and stashes
recommended_action on the returned Diagnosis; ai_recommend_fn just
reads it back off, so the orchestrator's (diagnose_fn, recommend_fn)
interface doesn't cost a second LLM call per step.

The AI is never the source of truth for which actions are legitimate.
That check lives entirely in policy.guard_ai() / policy.ACTION_LEGITIMACY,
independent of anything this module returns.
"""

from dataclasses import dataclass, field

import cache as cache_mod
import policy
from models import ActionType, Transaction
from policy import Diagnosis

PERMITTED_ACTIONS = [a.value for a in ActionType]


class ValidationError(Exception):
    pass


def build_context(txn: Transaction) -> dict:
    """
    Everything a real merchant system would have at decision time.
    Deliberately excludes customer_response_probability -- see module
    docstring and simulator.py's hidden-state comment.
    """
    return {
        "amount": txn.amount,
        "currency": txn.currency,
        "payment_method": txn.payment_method.value,
        "failure_reason": txn.failure_reason.value,
        "previous_failures": txn.previous_failures,
        "customer_payment_history": txn.customer_payment_history.value,
        "time_since_failure_hours": txn.time_since_failure_hours,
        "subscription_flag": txn.subscription_flag,
        "attempt_count": txn.attempt_count,
        "permitted_actions": PERMITTED_ACTIONS,
    }


def validate_response(raw: dict) -> tuple:
    """
    Schema + enum validation, independent of the policy legitimacy
    check (that one needs failure_reason and lives in policy.py).
    Raises ValidationError with a specific reason on any failure.
    """
    if not isinstance(raw, dict):
        raise ValidationError("response was not a JSON object")

    for field_name in ("diagnosis", "recommended_action", "confidence"):
        if field_name not in raw:
            raise ValidationError(f"missing field '{field_name}'")

    diagnosis_text = raw["diagnosis"]
    if not isinstance(diagnosis_text, str) or not diagnosis_text.strip():
        raise ValidationError("diagnosis was empty or not a string")

    action_str = raw["recommended_action"]
    if action_str not in PERMITTED_ACTIONS:
        raise ValidationError(f"recommended_action '{action_str}' is not in permitted_actions")

    confidence = raw["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValidationError(f"confidence '{confidence}' is not a number in [0, 1]")

    return diagnosis_text, ActionType(action_str), float(confidence)


@dataclass
class AgentStats:
    live_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallbacks: int = 0
    fallback_reasons: dict = field(default_factory=dict)

    def record_fallback(self, reason: str):
        self.fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def summary(self) -> str:
        total = self.live_calls + self.cache_hits + self.fallbacks
        lines = [
            f"AI agent stats: {total} decisions requested "
            f"({self.live_calls} live calls, {self.cache_hits} cache hits, "
            f"{self.cache_misses} cache misses, {self.fallbacks} fallbacks)"
        ]
        for reason, count in sorted(self.fallback_reasons.items()):
            lines.append(f"  fallback[{reason}]: {count}")
        return "\n".join(lines)


class AIAgent:
    """
    mode="live": call the LLM client; cache every successful response
                 as it comes in (so a later mode="cached" run against
                 the same cache file is reproducible).
    mode="cached": never calls the client. A cache miss is treated as
                 an LLM failure and falls back to the rule engine --
                 this is a deliberate design choice (per approved
                 architecture) so a cold/incomplete cache degrades
                 safely instead of silently going live mid-benchmark.
    """

    def __init__(self, client, cache: cache_mod.ResponseCache, mode: str = "live", stats: AgentStats = None):
        assert mode in ("live", "cached")
        self.client = client
        self.cache = cache
        self.mode = mode
        self.stats = stats if stats is not None else AgentStats()

    def _fallback(self, txn: Transaction, reason: str) -> Diagnosis:
        self.stats.record_fallback(reason)
        rule_diagnosis = policy.diagnose(txn)
        rule_action = policy.recommend_action(txn, rule_diagnosis)
        return Diagnosis(
            category=rule_diagnosis.category,
            explanation=f"[AI FALLBACK: {reason}] {rule_diagnosis.explanation}",
            confidence=None,
            source="ai_fallback",
            recommended_action=rule_action,
        )

    def get_recommendation(self, txn: Transaction) -> Diagnosis:
        context = build_context(txn)
        key = cache_mod.make_key(context)

        raw = None
        source = "ai"

        if self.mode == "cached":
            raw = self.cache.get(key)
            if raw is None:
                self.stats.cache_misses += 1
                return self._fallback(txn, "cache_miss")
            self.stats.cache_hits += 1
            source = "ai_cached"
        else:  # live
            cached = self.cache.get(key)
            if cached is not None:
                raw = cached
                self.stats.cache_hits += 1
                source = "ai_cached"
            else:
                try:
                    raw = self.client.complete(context)
                except Exception as e:
                    return self._fallback(txn, f"llm_error: {e}")
                self.stats.live_calls += 1
                self.cache.set(key, raw)
                source = "ai"

        try:
            diagnosis_text, action, confidence = validate_response(raw)
        except ValidationError as e:
            return self._fallback(txn, f"invalid_output: {e}")

        return Diagnosis(
            category="AI",
            explanation=diagnosis_text,
            confidence=confidence,
            source=source,
            recommended_action=action,
        )

    # Adapters matching orchestrator.run_case's (diagnose_fn, recommend_fn)
    # interface. recommend_fn does NOT call the LLM again -- it reads the
    # action already computed and stashed on the Diagnosis by diagnose_fn
    # in the same FSM step.
    def diagnose_fn(self, txn: Transaction) -> Diagnosis:
        return self.get_recommendation(txn)

    def recommend_fn(self, txn: Transaction, diagnosis: Diagnosis) -> ActionType:
        return diagnosis.recommended_action
