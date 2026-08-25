"""
LLM client abstraction for Version 1.

Three implementations behind the same interface (`.complete(context) -> dict`,
raising LLMClientError on any failure):

- AnthropicLLMClient: calls api.anthropic.com, uses strict tool-use
  for schema-constrained structured output. Requires
  ANTHROPIC_API_KEY and an explicit ANTHROPIC_MODEL (no default --
  see its docstring for why).

- GeminiLLMClient: calls generativelanguage.googleapis.com, uses
  generationConfig.responseSchema for schema-constrained structured
  output. Requires GEMINI_API_KEY and an explicit GEMINI_MODEL (same
  no-default reasoning as Anthropic -- Gemini's model lineup moves
  fast enough that a hardcoded "cheap model" name would likely be
  stale within weeks; verify free-tier eligibility and availability
  for your key yourself at
  https://ai.google.dev/gemini-api/docs/pricing before choosing one).
  Added specifically because the Anthropic path costs money and this
  project has a ₹0 budget -- Gemini's Flash-family models currently
  have a free tier (with usage-based-improvement tradeoffs Google
  documents), though which specific model is in the free tier changes
  over time, which is exactly why this client won't guess for you.

- DemoLLMClient: a deterministic, seeded stand-in that does NOT call
  any network. It exists so the full pipeline (context -> "AI" ->
  validation -> policy -> simulator) can be run, demoed, and tested
  without an API key. It is NOT meant to represent real intelligence
  -- it's explicitly a heuristic over the same context fields, with a
  configurable small chance of producing invalid output, so the
  validation/fallback path in ai_agent.py has something real to catch.
  This distinction is deliberate: don't let a demo client's plausible
  output be mistaken for the real experiment.
"""

import hashlib
import json
import os
import random
import time

from models import ActionType

FAILURE_LLM = "LLM call failed"

# Transient HTTP statuses worth a bounded retry -- server-side or
# rate-limit problems that often clear on their own. 400/401/403 (bad
# request, auth, permissions) are NOT in this set on purpose: retrying
# those just repeats the same failure and burns quota for nothing.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _send_with_backoff(send_fn, max_retries: int = 3, base_delay: float = 0.5, sleep_fn=time.sleep):
    """
    Calls send_fn() -> response repeatedly, retrying ONLY when the
    response's status_code is in RETRYABLE_STATUS_CODES, with bounded
    exponential backoff (base_delay * 2**attempt). Every other status
    code is returned immediately on the first try -- this function
    only governs whether to retry AFTER a response comes back; an
    exception raised BY send_fn itself (network error, timeout) is not
    caught here and propagates immediately, same as before this
    function existed -- those aren't "transient HTTP status" failures,
    they're a different failure class the spec didn't ask to change.

    sleep_fn is injectable so tests can verify retry behavior without
    actually sleeping.
    """
    last_response = None
    for attempt in range(max_retries):
        response = send_fn()
        last_response = response
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response
        if attempt < max_retries - 1:
            sleep_fn(base_delay * (2 ** attempt))
    return last_response


class LLMClientError(Exception):
    """Raised for any LLM call failure: timeout, network error, non-2xx,
    missing API key, etc. ai_agent.py treats this uniformly as
    'fall back to the rule engine for this step'."""


class AnthropicLLMClient:
    """
    Real client. Uses the /v1/messages endpoint with a strict tool
    definition to force schema-conforming structured output
    (diagnosis, recommended_action, confidence), rather than asking
    for JSON in prose or trusting an unconstrained tool call.

    strict: true (combined with tool_choice forcing this specific
    tool) guarantees the response matches input_schema exactly --
    per current Anthropic docs, without strict mode a model can
    return a wrong type or omit a required field. See
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use

    Model is REQUIRED and explicit -- no silent default. Model IDs get
    superseded, and a stale hardcoded/default guess (this code shipped
    one earlier: "claude-sonnet-4-6") can silently burn API calls
    against a model your key can't use, or a model that's since been
    replaced. Set ANTHROPIC_MODEL, or pass model=... explicitly. Verify
    what's actually current/available to your key at
    https://platform.claude.com/docs/en/about-claude/models/overview
    before setting it -- this code does not check that for you.

    Requires the `requests` package and ANTHROPIC_API_KEY in the
    environment. Both api.anthropic.com and this module are otherwise
    inert until you actually invoke .complete().
    """

    API_URL = "https://api.anthropic.com/v1/messages"

    TOOL_SCHEMA = {
        "name": "recovery_recommendation",
        "description": "Diagnose a failed payment and recommend one recovery action.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Short explanation of the likely cause, given the context.",
                },
                "recommended_action": {
                    "type": "string",
                    "description": "One of the permitted_actions given in the prompt.",
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0 confidence in this recommendation.",
                },
            },
            "required": ["diagnosis", "recommended_action", "confidence"],
        },
    }

    def __init__(self, timeout_seconds: float = 15.0, model: str = None):
        self.timeout_seconds = timeout_seconds
        self.model = model or os.environ.get("ANTHROPIC_MODEL")
        if not self.model:
            raise LLMClientError(
                f"{FAILURE_LLM}: no model specified. Set the ANTHROPIC_MODEL environment "
                "variable or pass model=... explicitly -- there is no default. Verify a "
                "current model ID for your API key at "
                "https://platform.claude.com/docs/en/about-claude/models/overview first."
            )

    def complete(self, context: dict) -> dict:
        try:
            import requests
        except ImportError as e:
            raise LLMClientError(f"{FAILURE_LLM}: requests package not available ({e})")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMClientError(f"{FAILURE_LLM}: ANTHROPIC_API_KEY not set")

        prompt = (
            "You are diagnosing a single failed payment for a merchant revenue-recovery "
            "system. Given the transaction context below, diagnose the likely cause and "
            "recommend exactly one action from permitted_actions. Do not recommend any "
            "action outside that list.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [self.TOOL_SCHEMA],
                    "tool_choice": {"type": "tool", "name": "recovery_recommendation"},
                },
                timeout=self.timeout_seconds,
            )
        except Exception as e:  # network error, timeout, etc.
            raise LLMClientError(f"{FAILURE_LLM}: {e}")

        if response.status_code != 200:
            raise LLMClientError(f"{FAILURE_LLM}: HTTP {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            tool_use_blocks = [b for b in data["content"] if b.get("type") == "tool_use"]
            if not tool_use_blocks:
                raise LLMClientError(f"{FAILURE_LLM}: no tool_use block in response")
            return tool_use_blocks[0]["input"]
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"{FAILURE_LLM}: could not parse response ({e})")


class GeminiLLMClient:
    """
    Real client, Google Gemini Developer API. Uses generateContent with
    generationConfig.responseSchema to force schema-conforming
    structured output (same reasoning as AnthropicLLMClient's strict
    tool use: don't trust a model to freehand valid JSON, constrain it
    at generation time). The schema's recommended_action field uses an
    "enum" constraint listing the exact permitted actions, which
    Gemini's schema format supports directly -- a slightly stronger
    guarantee than Anthropic's schema gives (type: string there,
    enum-constrained here), though ai_agent.py's validate_response()
    still checks it independently either way; never trust the wire
    format alone.

    Model is REQUIRED and explicit -- no silent default. As of this
    writing, Gemini's model lineup is shipping new Flash-generation
    models every few weeks, and which specific model has free-tier
    access changes over time -- a hardcoded "cheap model" name in this
    file would likely be stale before you read it. Set GEMINI_MODEL,
    or pass model=... explicitly. Verify current model IDs and
    free-tier eligibility for your key at
    https://ai.google.dev/gemini-api/docs/pricing before choosing one
    -- this code does not check that for you, and does not assume any
    particular model is free just because it's cheap-sounding.

    Requires the `requests` package and GEMINI_API_KEY in the
    environment. Both generativelanguage.googleapis.com and this
    module are otherwise inert until you actually invoke .complete().
    Auth uses the current x-goog-api-key header (not the older ?key=
    query-string form, which still works but isn't what current docs
    show first). Transient failures (HTTP 429/500/502/503/504) get a
    bounded exponential-backoff retry (see _send_with_backoff) --
    auth/request errors (400/401/403) do not retry, since waiting
    won't fix a bad key or a malformed request.
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "Short explanation of the likely cause, given the context.",
            },
            "recommended_action": {
                "type": "string",
                "enum": [a.value for a in ActionType],
                "description": "One of the permitted_actions given in the prompt.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence in this recommendation.",
            },
        },
        "required": ["diagnosis", "recommended_action", "confidence"],
    }

    def __init__(self, timeout_seconds: float = 60.0, model: str = None):
        self.timeout_seconds = timeout_seconds
        self.model = model or os.environ.get("GEMINI_MODEL")
        if not self.model:
            raise LLMClientError(
                f"{FAILURE_LLM}: no model specified. Set the GEMINI_MODEL environment "
                "variable or pass model=... explicitly -- there is no default. Verify a "
                "current model ID (and its free-tier status, if that matters to you) at "
                "https://ai.google.dev/gemini-api/docs/pricing first."
            )

    @staticmethod
    def _parse_response(data: dict) -> dict:
        """Extracted for testability without mocking HTTP. Raises
        LLMClientError on any structurally unexpected response."""
        try:
            candidates = data["candidates"]
            if not candidates:
                raise LLMClientError(f"{FAILURE_LLM}: no candidates in Gemini response")
            parts = candidates[0]["content"]["parts"]
            text = parts[0]["text"]
            return json.loads(text)
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"{FAILURE_LLM}: could not parse Gemini response ({e})")

    def complete(self, context: dict) -> dict:
        try:
            import requests
        except ImportError as e:
            raise LLMClientError(f"{FAILURE_LLM}: requests package not available ({e})")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMClientError(f"{FAILURE_LLM}: GEMINI_API_KEY not set")

        prompt = (
            "You are diagnosing a single failed payment for a merchant revenue-recovery "
            "system. Given the transaction context below, diagnose the likely cause and "
            "recommend exactly one action from permitted_actions. Do not recommend any "
            "action outside that list.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )

        url = f"{self.API_BASE}/{self.model}:generateContent"

        def send():
            return requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": self.RESPONSE_SCHEMA,
                    },
                },
                timeout=self.timeout_seconds,
            )

        try:
            response = _send_with_backoff(send)
        except Exception as e:  # network error, timeout, etc. -- not retried, see _send_with_backoff
            raise LLMClientError(f"{FAILURE_LLM}: {e}")

        if response.status_code != 200:
            raise LLMClientError(f"{FAILURE_LLM}: HTTP {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
        except Exception as e:
            raise LLMClientError(f"{FAILURE_LLM}: response was not valid JSON ({e})")

        return self._parse_response(data)


class DemoLLMClient:
    """
    Deterministic, no-network stand-in for the real LLM. Seeded so the
    same context always produces the same output -- this is what makes
    a 'live' demo run still reproducible without needing the cache.

    Uses richer context than the rule engine on purpose (previous_failures,
    time_since_failure_hours, customer_payment_history, subscription_flag)
    so Strategy C has a genuine chance to diverge from Strategy B, not
    just restate it.

    invalid_output_rate: fraction of calls that deliberately return
    something the validator must reject (bad schema or a forbidden
    action) -- exists purely so the fallback path is exercised by
    something other than a hand-built unit test double. 0.0 by default;
    set it >0 to see fallback counts in a batch run.
    """

    def __init__(self, seed: int = 2026, invalid_output_rate: float = 0.0):
        self._seed = seed
        self.invalid_output_rate = invalid_output_rate

    def complete(self, context: dict) -> dict:
        key_material = json.dumps(context, sort_keys=True) + str(self._seed)
        digest = hashlib.sha256(key_material.encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        if rng.random() < self.invalid_output_rate:
            # Deliberately bad output, split between the two failure
            # modes validation must catch.
            if rng.random() < 0.5:
                return {"diagnosis": "malformed", "confidence": 0.5}  # missing recommended_action
            return {
                "diagnosis": "overconfident bad idea",
                "recommended_action": "TRANSFER_FUNDS_MANUALLY",  # not in permitted_actions
                "confidence": 0.9,
            }

        reason = context["failure_reason"]
        history = context["customer_payment_history"]
        prev_failures = context["previous_failures"]
        hours = context["time_since_failure_hours"]
        subscription = context["subscription_flag"]

        if reason in ("CARD_NETWORK_ERROR", "BANK_DECLINE"):
            if prev_failures >= 2 or hours > 24:
                action, diag = "SCHEDULE_RETRY", (
                    f"{prev_failures} prior failures and {hours:.1f}h elapsed suggest an "
                    "immediate retry is less likely to land than a delayed one."
                )
            else:
                action, diag = "RETRY", "Likely a transient network/bank-side issue."
        elif reason == "INSUFFICIENT_FUNDS":
            if history == "UNRELIABLE":
                action, diag = "SEND_MESSAGE", "Liquidity issue with an unreliable payer; prompt them directly."
            elif subscription and history == "USUALLY_PAYS_LATE":
                action, diag = "SEND_MESSAGE", (
                    "Subscription customer who usually pays late -- a nudge is more "
                    "reliable than waiting silently."
                )
            else:
                action, diag = "SCHEDULE_RETRY", "Temporary liquidity issue is plausible; funds may arrive."
        elif reason in ("EXPIRED_CARD", "AUTH_FAILURE"):
            action, diag = "SEND_MESSAGE", "Instrument-level issue; customer needs to act."
        else:
            action, diag = "ESCALATE", "Cause unclear from available signals."

        confidence = round(min(0.95, max(0.4, 0.75 - 0.05 * prev_failures + rng.uniform(-0.05, 0.05))), 2)
        return {"diagnosis": diag, "recommended_action": action, "confidence": confidence}
