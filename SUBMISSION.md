# AI Revenue Recovery Agent — Razorpay Buildathon Track 3

Status: **core engine frozen, submission documentation reflects the real Gemini pilot that has now been executed.** See "What remains" at the end for what's still outside this codebase (GitHub publication, video, form).

## 1. Problem

Payments fail. Not because customers changed their minds — because a
card network hiccupped, a bank declined for a moment, funds hadn't
landed yet, a card expired, or an OTP failed. Every one of those is
revenue sitting one retry, one message, or one waiting period away
from being recovered — or, mishandled, revenue permanently lost plus
a customer annoyed by a badly-timed retry.

## 2. Why payment recovery is a revenue problem, not a support problem

Most systems treat a failed payment as a support ticket or, worse, a
blind retry loop. Neither is a revenue strategy. Different failure
reasons need different interventions — retrying an `EXPIRED_CARD`
recovers nothing (0% success, by construction, not by estimate);
retrying `INSUFFICIENT_FUNDS` immediately is nearly as futile (5%),
while waiting recovers meaningfully more (35-75% depending on the
customer). A system that can't tell these apart is leaving money on
the table it could have kept. That gap is what this project measures
and closes.

## 3. System architecture

```
Transaction (synthetic, seeded)
    |
Context builder (only fields a real merchant system would have)
    |
Diagnosis + recommended action
    (AI, once per case)  <-or->  (Rule engine, every decision)
    |
Policy guard (deterministic, checks action legitimacy against the
              OBJECTIVE failure reason -- never against the AI's own
              stated diagnosis, and never bypassable by the AI)
    |
Execute (mock recovery tool, deterministic-per-tuple simulated outcome)
    |
Success -> RECOVERED          Failure -> deterministic bounded retry
                                          (rule engine, not the AI)
                                          -> success / escalate / stop
    |
Audit trail (every state transition, human-readable)
    |
Batch metrics (recovery rate, escalation/stop rate, category
                breakdown, AI-call accounting, agreement/divergence)
```

Three strategies run on **identical transactions** with
**strategy-independent simulated outcomes** (see §7), so the
comparison is apples-to-apples:

- **A — Fixed Retry**: no diagnosis, retries everything blindly. The
  control group.
- **B — Rule-Engine Agent**: deterministic diagnosis + action,
  bounded by the policy guard.
- **C — AI Agent**: an LLM makes the initial strategic recommendation;
  the same rule engine as B handles retries after that (see §4).

## 4. AI decision layer

The AI is consulted **once per case** — the initial strategic
decision. It receives everything a real merchant system would have at
decision time (`failure_reason`, `previous_failures`,
`customer_payment_history`, `time_since_failure_hours`,
`subscription_flag`, `amount`, `payment_method`) and returns a
structured `{diagnosis, recommended_action, confidence}`, schema-
constrained at generation time (Anthropic: strict tool use; Gemini:
`responseSchema` with an `enum` on `recommended_action`). If that
first action fails, retries run through the deterministic rule engine
— the AI is never re-consulted just because its recommendation didn't
work ("What broke and how we fixed it" §3 below covers why this changed).

Provider is swappable behind one interface (`llm_client.py`):
`AnthropicLLMClient`, `GeminiLLMClient` (added for a ₹0 budget —
Gemini's Flash family currently has a free tier), and `DemoLLMClient`
(a deterministic, no-network stand-in used for all development,
testing, and the large-scale benchmark — **every result in §9a is
from this demo client**, not a real model; the real model pilot is
§9b). Neither real client has a hardcoded model default — both require `ANTHROPIC_MODEL`/`GEMINI_MODEL` explicitly, since both
providers' model lineups move fast enough that a baked-in default
would likely be stale.

## 5. Deterministic policy guard

The AI recommends; it does not authorize. `policy.guard_ai()` checks
every recommended action against `ACTION_LEGITIMACY[failure_reason]`
— a lookup table keyed on the *objective* failure reason, never on
the AI's own diagnosis text (an AI can write a diagnosis that
rationalizes almost anything; a guard that trusted the diagnosis would
be trusting exactly the thing it exists to police). An illegitimate
recommendation is rejected straight to `ESCALATE` — never silently
substituted for a "nearest legitimate" action, so an AI failure stays
visible in the metrics instead of being papered over. Separately, the
guard enforces `MAX_AUTOMATED_ATTEMPTS = 3` regardless of what any
strategy recommends.

## 6. Recovery state machine

```
DETECTED -> ANALYZING -> DIAGNOSED -> INTERVENTION_SELECTED -> POLICY_CHECK
POLICY_CHECK --reject--> ESCALATED / STOPPED   (both terminal, distinct meanings:
POLICY_CHECK --allow--> ACTION_EXECUTED         ESCALATED = handed to a human,
ACTION_EXECUTED --success--> RECOVERED           STOPPED = automation halted, no handoff)
ACTION_EXECUTED --failure--> back to DIAGNOSED (bounded by attempt count)
```

A hard transition ceiling (`MAX_TRANSITIONS = 20`) exists as a second,
independent guarantee the loop cannot spin forever even if the attempt
bound were ever miscoded — belt and suspenders, verified unreachable
in practice by test.

## 7. Simulator and methodology

Synthetic transactions, seeded (`generate_transactions(n, seed)`).
Outcome probabilities are hand-picked, not fit to real data — made
explicit and inspectable via `simulator.RECOVERY_MODEL` (a single
source of truth read by both the simulator and the `print_assumptions()`
output — see "What broke and how we fixed it" below for why that
matters) rather than buried as
bare numbers. A hidden per-customer `customer_response_probability`
exists only inside the simulator; neither the rule engine nor the AI
ever sees it — they only see the same signals a real merchant system
would have. Outcomes are a deterministic hash of
`(seed, transaction_id, attempt_count, action)`, not draws from a
sequential RNG stream — this is deliberate, see "What broke and how
we fixed it" below.

## 8. Benchmark methodology

`run_benchmark.py` is the single reproducible command. Per run: same
transactions across all three strategies (verified in code, not
assumed), one shared simulator instance, deterministic seeds, bounded
attempts, zero hidden-state leakage to the AI. Reports: total
transactions, revenue at risk, recovered, unrecovered, recovery rate,
escalation rate, stop rate, average automated attempts, AI calls per
case, AI fallbacks, AI/rule first-action agreement and divergence,
policy violations, and a per-failure-category breakdown — for all
three strategies. `--seeds N` adds a multi-seed robustness pass
(mean ± population stdev), with a built-in check that flags when a
gap is smaller than its own noise rather than letting a thin result
pass as a finding.

## 9. Results — two separate evidence layers, not one

**Do not blend these.** They answer different questions and neither
substitutes for the other:

- **Layer A (below, §9a):** a large-scale synthetic benchmark
  (n=1000-10000, `DemoLLMClient`). Demonstrates the recovery workflow,
  the economics of context-aware routing vs. blind retry, bounded
  execution, and strategy behavior at scale. It says nothing about a
  real model's decision quality — `DemoLLMClient` is a hand-coded
  heuristic, not an LLM.
- **Layer B (§9b):** a small real-Gemini pilot (n=5). Demonstrates
  that the real LLM decision layer integrates correctly with the
  agent — structured output parses, the policy guard gates it
  correctly, fallback triggers and recovers cleanly on real API
  failures. It is **not** a statistically meaningful comparison of AI
  vs. rule-engine quality at this sample size, and is not presented as
  one.

### 9a. Large-scale synthetic benchmark (DemoLLMClient, not a real model)

**Single seed, n=1000, seed=42:**

| | A: Fixed Retry | B: Rule Engine | C: AI Agent |
|---|---|---|---|
| Recovered | ₹97.6L | ₹184.7L | ₹188.3L |
| Recovery rate | 39.50% | 74.74% | 76.20% |
| Unrecovered | ₹149.5L | ₹62.4L | ₹58.8L |
| Escalation rate | 59.50% | 24.10% | 22.40% |
| AI calls / case | — | — | 1.00 |
| AI fallbacks | — | — | 0/1000 |
| Policy violations | — | — | 0/1000 |
| AI/rule first-action agreement | — | — | 69.00% (310/1000 diverge) |

B vs A: rule-based routing recovers **+89.2%** more than blind retry.
C vs B: **+1.96%** further on top — see the multi-seed result below
for whether this single-seed gap is stable.

**Multi-seed robustness, n=500 per seed, 20 seeds:**

| Strategy | Mean recovery rate | Stdev |
|---|---|---|
| A: Fixed Retry | 40.26% | ± 2.72pp |
| B: Rule Engine | 76.81% | ± 2.12pp |
| C: AI Agent | 78.66% | ± 2.16pp |

**C vs B mean gap: +1.85 percentage points.** This gap is **smaller
than the AI agent's own seed-to-seed standard deviation (2.16pp)** and
must **not** be presented as statistically significant. Report it
exactly as that: a small, not-yet-distinguishable-from-noise
advantage at this sample size — not "the AI beats the rule engine,"
not "the AI is neutral," just an open question this benchmark design
can detect but this many seeds can't yet resolve. `run_benchmark.py`'s
own `--seeds N` output flags this automatically rather than letting a
thin result pass silently.

### 9b. Real Gemini pilot (n=5, small live validation — not evidence of AI superiority)

Two real pilot runs were executed, on two different models, and both
are reported — not just the more favorable one. The first run hit
launch-week instability on a brand-new model; switching models (a
same-day, same-architecture swap, since the provider is abstracted
behind one interface — see §4) produced a full 5/5 live sample with a
genuine policy-guard catch. Both runs are real API calls against a
free-tier key; neither is cherry-picked over the other.

**Run 2 (reported as primary — full live sample, no infrastructure noise):**

```
$env:GEMINI_MODEL="gemini-3.5-flash-lite"
python run_pilot.py --n 5 --seed 42 --llm-client gemini
```

| | Count |
|---|---|
| Live Gemini calls that succeeded | 5 / 5 |
| Fallbacks | 0 |
| Final actions agreeing with the rule engine | 2 / 5 (40%) |
| Policy violations (Gemini recommended an illegitimate action) | **1 / 5** |

**All 5 cases, in full:**

| TX | Failure | Rule engine action | Gemini action | Agree? | Rule engine outcome | Gemini outcome |
|---|---|---|---|---|---|---|
| TX_00000 | BANK_DECLINE | RETRY | SCHEDULE_RETRY | No | Recovered ₹33,898.63 | Recovered ₹33,898.63 |
| TX_00001 | INSUFFICIENT_FUNDS | SCHEDULE_RETRY | SEND_MESSAGE | No | Recovered ₹28,149.00 | Recovered ₹28,149.00 |
| TX_00002 | BANK_DECLINE | RETRY | SEND_MESSAGE | No | Recovered ₹17,143.48 | **REJECTED by policy guard → Escalated, ₹0** |
| TX_00003 | CARD_NETWORK_ERROR | RETRY | RETRY | Yes | Recovered ₹30,264.56 | Recovered ₹30,264.56 |
| TX_00004 | CARD_NETWORK_ERROR | RETRY | RETRY | Yes | Recovered ₹28,951.14 | Recovered ₹28,951.14 |

**TX_00002 is the most important row in this table, not a footnote.**
Gemini recommended `SEND_MESSAGE` for a `BANK_DECLINE` — not a
legitimate action for that failure type under `ACTION_LEGITIMACY`
(there is nothing a customer-facing message accomplishes when their
own bank declined the charge). `guard_ai()` rejected it and forced
`ESCALATE`, exactly as designed — see §5. The honest cost: this
transaction, which the rule engine would have auto-recovered, went to
manual escalation instead of automated recovery under Strategy C. This
is not a case of the AI "winning" or "losing" in the abstract — it is
the safety architecture doing precisely its job: a real model, on a
real call, proposed something operationally invalid, and the
deterministic guard — not the AI's own judgment — is what prevented it
from executing. That is the project's core thesis (§5), demonstrated
against real model output rather than only synthetic tests.

On the two cases where Gemini agreed with the rule engine
(TX_00003, TX_00004 — both `CARD_NETWORK_ERROR` → `RETRY`), and the two
where it proposed a different but still-legitimate action that also
fully recovered (TX_00000, TX_00001) — no claim is made that Gemini's
alternative was *better* than the rule engine's; both paths succeeded
on this simulator, and n=2/n=2 is far too small to say more than "a
legitimate divergence occurred and didn't fail."

**Run 1 (first attempt, kept for transparency — not the headline number):**

```
python run_pilot.py --n 5 --seed 42 --llm-client gemini --cache-file gemini_pilot_cache.json
```
(model: `gemini-3.7-flash`, a preview model that shipped days before
this pilot was run)

| | Count |
|---|---|
| Live Gemini calls that succeeded | 2 |
| Fallbacks (all 3 caused by Gemini HTTP 503 UNAVAILABLE) | 3 |
| Policy violations | 0 |

The 2 genuine decisions in Run 1: BANK_DECLINE → RETRY (0.85
confidence, recovered); CARD_NETWORK_ERROR → RETRY (0.90 confidence,
recovered) — both trivial agreements with the rule engine, no
divergence signal. The 3 fallbacks were real infrastructure failures
(HTTP 503, not simulated) caught cleanly by the fallback path (§10)
with zero crashes.

**Why the model changed between runs:** `gemini-3.7-flash` returned
503 UNAVAILABLE on 5/5 calls in a second attempt at Run 1 as well,
consistent with launch-week overload on a brand-new preview model
rather than anything specific to this project's request shape.
Switching to `gemini-3.5-flash-lite` — an older, more established
model on the same free tier — required a one-line environment
variable change and zero code changes, which is the direct payoff of
keeping the provider behind one interface (§4). This is itself a small
but real demonstration of that design decision paying off under
pressure, not just in theory.

**What this pilot actually demonstrates, combining both runs:** the
real LLM integration works end-to-end on two different models —
structured output parses and validates correctly, the policy guard
gates real model output correctly (including catching a real invalid
recommendation, Run 2/TX_00002), and real infrastructure failures
(Run 1's 503s) are caught and handled without crashing the batch.
**What it does not demonstrate:** a statistically meaningful
comparison of Gemini's decision quality against the rule engine's —
n=5 per run is a validation sample for the plumbing and the guard, not
a dataset for a superiority claim. The 40% agreement rate in Run 2 is
worth noting precisely because it's *not* near 100% (the failure mode
this pilot was built to catch — see run_pilot.py's own printed
guidance) — Gemini is engaging with more than just `failure_reason`,
for better (two legitimate alternative paths) and for worse (one
illegitimate one, safely caught).

## 10. Failure handling / fallback

Every LLM failure mode — timeout, network error, malformed JSON,
missing field, an action outside the permitted set, HTTP 429/500-504
(retried with bounded exponential backoff; 400/401/403 are not, since
waiting doesn't fix a bad key or a malformed request) — falls back to
the deterministic rule engine for that single case. A fallback is
logged (`source="ai_fallback"`) and counted, never silent. This means
Strategy C can never do *worse* than "rule engine, plus occasional AI
upgrades" from infrastructure failure alone.

## 11. Audit trail

Every case produces a full, human-readable step-by-step trail:
`DETECTED -> ANALYZING -> DIAGNOSED -> INTERVENTION_SELECTED ->
POLICY_CHECK -> ACTION_EXECUTED -> (RECOVERED | loop | ESCALATED |
STOPPED)`. Rendered visually in `report.html` (§13) as a stamped
sequence — the policy-guard step literally shows an ALLOWED/REJECTED
stamp, so the "AI recommends, guard decides" architecture is visible
on screen, not just in code.

## 12. Known limitations

- **The large-scale benchmark (§9a) is `DemoLLMClient`, a
  deterministic heuristic stand-in — not a real model call.** This
  remains the single most important caveat for that layer. Two real
  Gemini pilots (§9b) have been executed at n=5 each — one hit
  launch-week model instability (3/5 fallbacks), the other ran clean
  (5/5 live, including a real policy-guard rejection of an invalid
  Gemini recommendation) — which validates that the integration and
  the guard both work correctly, not that a real model outperforms
  the rule engine at scale. Scaling the real pilot to something closer
  to n=500-1000 live Gemini calls would be needed before any quality
  claim about the real model could be made; that has not been done
  and isn't claimed.
- Outcome probabilities are hand-picked, not empirically fit.
- Single loss type (payment failures) by deliberate scope decision —
  see §4 of the earlier assessment; checkout abandonment and overdue
  receivables would need a structurally separate action/legitimacy
  model, not a small addition, and weren't worth destabilizing the
  validated core for this close to the deadline.
- Single-threaded batch loop; fine at n=1000-10000, would need
  batching/async at real production volume.
- `report.html` uses system font stacks, not a webfont — deliberate,
  for offline reliability during recording, at a small cost to visual
  polish.

## 13. How to run the demo

```bash
python3 run_benchmark.py --n 1000 --seed 42 --out benchmark_report.json
python3 report.py --in benchmark_report.json --out report.html
# open report.html in a browser
```

## 14. How to reproduce the benchmark

See `README.md` for fresh-clone setup (`requirements.txt`, venv,
macOS/Linux and Windows PowerShell instructions).

```bash
# Single seed, detailed
python3 run_benchmark.py --n 1000 --seed 42

# Multi-seed robustness (mean +/- stdev, flags thin gaps)
python3 run_benchmark.py --n 500 --seeds 20

# The real model pilot (requires GEMINI_API_KEY + GEMINI_MODEL,
# or ANTHROPIC_API_KEY + ANTHROPIC_MODEL) -- kept small on purpose,
# see SUBMISSION.md Sec 9b: this validates the live integration and
# fallback handling, not model quality at scale
python3 run_pilot.py --n 5 --seed 42 --llm-client gemini

# Full test suite
python3 -m unittest test_recovery_agent.py test_ai_agent.py -v
```

## What broke and how we fixed it

Three real, verified issues found during development — nothing here
is invented for narrative effect.

**1. Strategy-independent randomness (RNG fairness bug).** The
simulator originally used one sequential `random.Random(seed)` stream
shared within a strategy's run. Strategy A (blind retry) burns more
attempts on hopeless cases than B or C, which route away from retry
immediately — so each strategy's RNG stream drifted out of sync after
the very first transaction. "Same seed" was true; "same potential
outcome for transaction N" was not. Verified impact: re-running the
20-seed benchmark before/after the fix flipped a real result — under
the buggy RNG, the AI agent lost to the rule engine on one seed (an
artifact of stream drift, not a real decision quality difference);
after the fix (outcomes are now a deterministic hash of
`(seed, transaction_id, attempt_count, action)`, not stream position),
that seed favored the AI agent like every other. Fixed in
`simulator.py`; regression-tested by two tests that deliberately burn
unrelated draws before checking a target outcome is unaffected.

**2. Action-legitimacy matrix too narrow.** `ACTION_LEGITIMACY`
initially only allowed `RETRY` (not `SCHEDULE_RETRY`) for transient
network/bank failures. The AI, using richer context than the rule
engine (as intended), reasonably proposed `SCHEDULE_RETRY` for a
repeat-failure transient case — got flagged as a policy violation and
forced to `ESCALATE`, collapsing AI recovery on those categories from
~98% to ~35%. Not a finding about the AI; an overly narrow matrix.
`SCHEDULE_RETRY` isn't unsafe for a transient reason the way `RETRY`
is genuinely impossible for an expired card — the matrix now reflects
that distinction.

**3. Excessive repeated LLM calls per transaction.** The AI was
originally re-consulted on every retry attempt within a case, not just
the first — 1.87 AI calls per transaction at n=1000, wasteful against
a free-tier quota and conceptually wrong (the AI should make the
strategic call; deterministic logic should handle bounded execution).
Fixed by splitting `orchestrator.run_case()`'s decision functions into
"initial" (used once, `attempt_count == 0`) and "retry" (deterministic,
reused from the rule engine, used thereafter) — calls per transaction
dropped to exactly 1.00. Honest cost, not hidden: this also shrank the
AI's measured advantage over the rule engine, since a case whose first
AI-recommended action fails becomes functionally identical to Strategy
B for its remaining attempts.

## What remains for the human/team to do

- *(Done: two real Gemini pilots executed — see §9b. Run 2 (primary):
  n=5, 5/5 live calls, 0 fallbacks, including a real policy-guard
  rejection of an illegitimate Gemini recommendation. Run 1: n=5,
  2 genuine decisions + 3 correctly-handled infrastructure fallbacks.
  Optional follow-up, not required for submission: scale the live
  pilot to a larger n if a real-model quality claim is wanted before
  the deadline — see §12.)*
- Publish the repository to GitHub (public).
- Take screenshots / record `report.html` for the pitch video.
- Record the 5-minute pitch.
- Fill out and submit the application form.

## Final statement

**Core engine frozen.** Architecture, policy guard, simulator
methodology, benchmark, and demo layer are complete, tested (93/93),
and stable. The project is **technically submission-ready as a
prototype** against Track 3's stated bar (batch-measured recovery,
compliant escalation, stopping rules, audit trail). The real Gemini
integration has been validated end-to-end at small scale (§9b) — the
remaining work is outside this codebase: publishing the repo, and
recording the pitch. No further code changes are recommended before
submission.
