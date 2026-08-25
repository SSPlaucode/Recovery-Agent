# AI Revenue Recovery Agent — Razorpay Buildathon Track 3

**See [SUBMISSION.md](./SUBMISSION.md) for the full write-up** —
problem, architecture, AI decision layer, policy guard, methodology,
results, limitations, and how to reproduce everything. This file is
setup + quick-start commands.

## Setup (fresh clone)

Requires Python 3.10+. Only one third-party runtime dependency
(`requests`, used by the real Anthropic/Gemini clients — not needed
for the default demo client) plus `pytest` to run the test suite the
documented way; see `requirements.txt`.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Expected: `93 passed`. `python -m unittest test_recovery_agent.py
test_ai_agent.py -v` also works (the suite is plain
`unittest.TestCase`, no pytest-specific features) if you'd rather not
install pytest.

## Quick start

```bash
python run_benchmark.py --n 1000 --seed 42                     # the benchmark
python report.py --in benchmark_report.json --out report.html  # the demo

python run_pilot.py --n 5 --seed 42 --llm-client gemini        # the real pilot,
                                                                 # small on purpose --
                                                                 # validates the live
                                                                 # integration, not scale
                                                                 # (needs GEMINI_API_KEY + GEMINI_MODEL)
```

A pre-generated `report.html` (demo-client data) is included in this
repo so you can open it immediately without running anything.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | The one third-party dependency (`requests`) + `pytest` |
| `models.py` | Core dataclasses: `Transaction`, `CaseResult`, `Diagnosis`, enums |
| `simulator.py` | Seeded synthetic data + outcome model (`RECOVERY_MODEL`, single source of truth) |
| `policy.py` | Rule engine, `MAX_AUTOMATED_ATTEMPTS`, `ACTION_LEGITIMACY`, policy guards |
| `orchestrator.py` | The recovery state machine — one AI call per case, deterministic retries |
| `ai_agent.py` | Context building, validation, fallback |
| `llm_client.py` | `AnthropicLLMClient`, `GeminiLLMClient`, `DemoLLMClient` — one interface |
| `cache.py` | Response cache for reproducible live/cached benchmarking |
| `baseline.py` | Fixed-retry control strategy |
| `experiment.py` | Runs all three strategies on identical transactions |
| `metrics.py` | Batch metrics, category breakdown, agreement/divergence |
| `run_benchmark.py` | **The** reproducible benchmark command |
| `run_batch.py`, `run_multiseed.py` | Earlier single/multi-seed tools (still work, superseded by `run_benchmark.py` for the submission) |
| `run_pilot.py` | The real-model per-case inspection pilot |
| `report.py` | Static HTML report generator (no business logic, reads benchmark JSON) |
| `test_recovery_agent.py`, `test_ai_agent.py` | 93 tests |
| `SUBMISSION.md` | Full submission write-up |

