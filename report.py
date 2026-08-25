"""
Minimum visual demo layer for the pitch video.

    python run_benchmark.py --n 1000 --seed 42 --out benchmark_report.json
    python report.py --in benchmark_report.json --out report.html

Renders a single, self-contained, offline HTML file from an existing
benchmark_report.json (produced by run_benchmark.py). No business
logic lives here -- every number on the page is read directly from
the JSON; this file only formats and lays it out. Zero external
dependencies (no CDN fonts/JS/CSS) so it renders identically and
reliably on camera with no network involved -- uses system font
stacks instead of a webfont, a deliberate tradeoff for demo
reliability over polish.

Design brief (for future reference / re-runs of this file):
  Subject: an audit ledger for a bounded, gated recovery agent --
  money moving through a system that only acts within limits it can
  prove. Not a generic analytics dashboard.
  Palette: cool paper/ink (not the warm-cream+terracotta AI-design
  default), with three FUNCTIONAL accent colors tied to real outcomes:
  recovered (green), escalated (rust), AI-sourced (indigo). A muted
  red is reserved only for the policy guard's REJECTED stamp.
  Type: serif display for the headline number, a plain sans for body,
  and monospace for every money figure, transaction ID, and audit-
  trail line -- because a ledger/terminal actually would set those in
  monospace; it's a functional choice, not decoration.
  Signature element: one real case rendered as a vertical stamped
  sequence (DETECTED -> AI RECOMMENDATION -> POLICY GUARD -> EXECUTE ->
  OUTCOME), with the policy-guard step rendered as a literal
  ALLOWED/REJECTED rubber-stamp badge. This is the one thing worth
  being memorable about, because it IS the project's actual thesis:
  the AI recommends, the deterministic guard decides.
"""

import argparse
import html
import json


def _money(x):
    if x is None:
        return "—"
    return f"₹{x:,.2f}"


def _pct(x):
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def _bar(rate, width_px=220):
    filled = max(0, min(width_px, round(width_px * rate)))
    return (
        f'<span class="bar"><span class="bar-fill" style="width:{filled}px"></span></span>'
    )


STATE_CLASS = {
    "DETECTED": "step-detect",
    "ANALYZING": "step-detect",
    "DIAGNOSED": "step-ai",
    "INTERVENTION_SELECTED": "step-ai",
    "POLICY_CHECK": "step-policy",
    "ACTION_EXECUTED": "step-exec",
    "RECOVERED": "step-outcome-good",
    "ESCALATED": "step-outcome-warn",
    "STOPPED": "step-outcome-warn",
}


def render_case_trail(case, title):
    if not case:
        return ""
    rows = []
    for event in case["audit_trail"]:
        state = event["state"]
        detail = html.escape(event["detail"])
        cls = STATE_CLASS.get(state, "step-default")
        stamp = ""
        if state == "POLICY_CHECK":
            rejected = "rejected" in detail.lower()
            stamp_cls = "stamp-reject" if rejected else "stamp-allow"
            stamp_text = "REJECTED" if rejected else "ALLOWED"
            stamp = f'<span class="stamp {stamp_cls}">{stamp_text}</span>'
        rows.append(
            f'<div class="trail-row {cls}">'
            f'  <span class="trail-state">{state}</span>'
            f'  <span class="trail-detail">{detail}</span>'
            f'  {stamp}'
            f'</div>'
        )
    terminal = case["terminal_state"]
    terminal_cls = "chip-good" if terminal == "RECOVERED" else "chip-warn"
    return f"""
    <div class="case-card">
      <div class="case-head">
        <span class="case-title">{title}</span>
        <span class="case-id">{html.escape(case['transaction_id'])}</span>
        <span class="chip {terminal_cls}">{terminal}</span>
      </div>
      <div class="case-meta">
        {_money(case['amount'])} at risk &middot;
        failure: {html.escape(case['failure_reason'] or '—')} &middot;
        AI calls: {case['ai_calls']} &middot;
        recovered: {_money(case['amount_recovered'])}
      </div>
      <div class="trail">
        {''.join(rows)}
      </div>
    </div>
    """


def render(report: dict) -> str:
    single = report["single_seed"]
    meta = single["methodology"]
    strategies = single["strategies"]
    agr = single["ai_vs_rule_first_action_agreement"]
    breakdown = single["category_breakdown"]
    samples = single.get("sample_cases", {})

    a = strategies[0]
    b = strategies[1]
    c = strategies[2]

    ledger_rows = "".join(
        f"""
        <div class="ledger-row">
          <span class="ledger-label">{html.escape(s['strategy'])}</span>
          {_bar(s['recovery_rate'])}
          <span class="ledger-pct">{_pct(s['recovery_rate'])}</span>
          <span class="ledger-money">{_money(s['recovered_revenue'])}</span>
        </div>"""
        for s in strategies
    )

    categories = sorted(set().union(*[set(v) for v in breakdown.values()]))
    cat_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(cat)}</td>
          <td class="num">{_pct(breakdown['a_fixed_retry'].get(cat, {}).get('recovery_rate'))}</td>
          <td class="num">{_pct(breakdown['b_rule_engine'].get(cat, {}).get('recovery_rate'))}</td>
          <td class="num">{_pct(breakdown['c_ai_agent'].get(cat, {}).get('recovery_rate'))}</td>
        </tr>"""
        for cat in categories
    )

    multi = report.get("multi_seed")
    multi_html = ""
    if multi:
        ms = multi["summary"]
        multi_html = f"""
        <section class="section">
          <h2>Robustness &mdash; {len(multi['seeds'])} seeds &times; {multi['n']} transactions</h2>
          <div class="multiseed-grid">
            {''.join(f'''
            <div class="multiseed-card">
              <div class="multiseed-label">{lbl}</div>
              <div class="multiseed-value">{_pct(ms[k]['mean_recovery_rate'])}
                <span class="multiseed-stdev">&plusmn; {ms[k]["stdev_recovery_rate"]*100:.2f}pp</span></div>
            </div>''' for k, lbl in [("A", "Fixed Retry"), ("B", "Rule Engine"), ("C", "AI Agent")])}
          </div>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Revenue Recovery &mdash; Benchmark Report</title>
<style>
  :root {{
    --paper: #EDEBE4;
    --paper-raised: #F7F6F1;
    --ink: #15181D;
    --ink-muted: #5B5F66;
    --rule: #CFCBBE;
    --recovered: #1F6F54;
    --escalated: #A6491F;
    --ai: #2B4C7E;
    --reject: #8C2F2F;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: -apple-system, "IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 920px; margin: 0 auto; padding: 56px 28px 96px; }}
  .mono {{ font-family: ui-monospace, "IBM Plex Mono", "SF Mono", Consolas, monospace; }}
  .eyebrow {{
    font-family: ui-monospace, "IBM Plex Mono", Consolas, monospace;
    font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-muted); margin-bottom: 14px;
  }}
  h1 {{
    font-family: Georgia, "Iowan Old Style", "Source Serif 4", serif;
    font-size: 54px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em;
  }}
  h1 .of {{ color: var(--ink-muted); font-weight: 400; font-size: 0.55em; }}
  .subhead {{ color: var(--ink-muted); font-size: 15px; margin-bottom: 44px; }}
  .subhead .mono {{ color: var(--ink); }}
  .section {{ margin-top: 52px; }}
  h2 {{
    font-family: Georgia, "Iowan Old Style", "Source Serif 4", serif;
    font-size: 22px; font-weight: 600; margin: 0 0 18px;
    border-bottom: 1px solid var(--rule); padding-bottom: 10px;
  }}
  .ledger-row {{
    display: grid; grid-template-columns: 150px 1fr 64px 150px;
    align-items: center; gap: 14px; padding: 10px 0;
    border-bottom: 1px solid var(--rule);
  }}
  .ledger-label {{ font-size: 14px; font-weight: 600; }}
  .ledger-pct {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 14px; text-align: right; }}
  .ledger-money {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 14px; text-align: right; color: var(--ink-muted); }}
  .bar {{ display: inline-block; width: 220px; height: 10px; background: var(--rule); border-radius: 2px; overflow: hidden; }}
  .bar-fill {{ display: block; height: 100%; background: var(--recovered); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); }}
  th {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--ink-muted); font-weight: 500; }}
  td.num {{ font-family: ui-monospace, "IBM Plex Mono", monospace; text-align: right; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  .stat-card {{ background: var(--paper-raised); border: 1px solid var(--rule); border-radius: 4px; padding: 16px; }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-muted); margin-bottom: 6px; }}
  .stat-value {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 20px; font-weight: 600; }}
  .multiseed-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .multiseed-card {{ background: var(--paper-raised); border: 1px solid var(--rule); border-radius: 4px; padding: 18px; text-align: center; }}
  .multiseed-label {{ font-size: 12px; color: var(--ink-muted); margin-bottom: 8px; }}
  .multiseed-value {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 24px; font-weight: 600; }}
  .multiseed-stdev {{ font-size: 13px; color: var(--ink-muted); font-weight: 400; }}
  .case-card {{ background: var(--paper-raised); border: 1px solid var(--rule); border-radius: 6px; padding: 20px; margin-bottom: 22px; }}
  .case-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }}
  .case-title {{ font-weight: 600; font-size: 15px; }}
  .case-id {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-muted); }}
  .chip {{ margin-left: auto; font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 11px;
           padding: 3px 8px; border-radius: 3px; font-weight: 600; letter-spacing: 0.04em; }}
  .chip-good {{ background: rgba(31,111,84,0.12); color: var(--recovered); }}
  .chip-warn {{ background: rgba(166,73,31,0.12); color: var(--escalated); }}
  .case-meta {{ font-size: 13px; color: var(--ink-muted); margin-bottom: 16px; }}
  .trail {{ border-left: 2px solid var(--rule); padding-left: 16px; }}
  .trail-row {{ display: flex; align-items: baseline; gap: 12px; padding: 7px 0; font-size: 13px; position: relative; }}
  .trail-row::before {{
    content: ""; position: absolute; left: -20px; top: 13px; width: 8px; height: 8px;
    border-radius: 50%; background: var(--rule);
  }}
  .step-ai::before {{ background: var(--ai); }}
  .step-policy::before {{ background: var(--ink); }}
  .step-outcome-good::before {{ background: var(--recovered); }}
  .step-outcome-warn::before {{ background: var(--escalated); }}
  .trail-state {{ font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 11px;
                   letter-spacing: 0.04em; color: var(--ink-muted); min-width: 150px; }}
  .trail-detail {{ flex: 1; }}
  .stamp {{
    font-family: ui-monospace, "IBM Plex Mono", monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; padding: 2px 10px; border: 2px solid; border-radius: 3px;
    transform: rotate(-3deg); white-space: nowrap;
  }}
  .stamp-allow {{ color: var(--recovered); border-color: var(--recovered); }}
  .stamp-reject {{ color: var(--reject); border-color: var(--reject); }}
  footer {{ margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--rule);
            font-size: 12px; color: var(--ink-muted); }}
  footer .mono {{ display: block; margin-top: 6px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Razorpay Buildathon &middot; Track 3 &middot; AI Revenue Recovery</div>
  <h1>{_money(c['recovered_revenue'])} <span class="of">recovered of {_money(a['total_revenue_at_risk'])} at risk</span></h1>
  <div class="subhead">
    <span class="mono">n={meta['n']}</span> synthetic at-risk transactions &middot;
    seed <span class="mono">{meta['seed']}</span> &middot;
    AI client <span class="mono">{meta['llm_client']}</span> &middot;
    identical transactions and strategy-independent randomness across all three strategies
  </div>

  <section class="section">
    <h2>Strategy comparison</h2>
    {ledger_rows}
  </section>

  <section class="section">
    <h2>AI vs. rule engine &mdash; first-action agreement</h2>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Agree</div><div class="stat-value">{agr['agree']}/{agr['total']}</div></div>
      <div class="stat-card"><div class="stat-label">Diverge</div><div class="stat-value">{agr['disagree']}/{agr['total']}</div></div>
      <div class="stat-card"><div class="stat-label">AI calls / case</div><div class="stat-value">{c['ai_calls_per_case']:.2f}</div></div>
      <div class="stat-card"><div class="stat-label">Policy violations</div><div class="stat-value">{c['policy_violation_count']}/{c['total_transactions']}</div></div>
    </div>
  </section>

  <section class="section">
    <h2>Recovery rate by failure category</h2>
    <table>
      <tr><th>Category</th><th>A: Fixed Retry</th><th>B: Rule Engine</th><th>C: AI Agent</th></tr>
      {cat_rows}
    </table>
  </section>

  {multi_html}

  <section class="section">
    <h2>One case, gated</h2>
    {render_case_trail(samples.get('recovered'), 'Recovered case')}
    {render_case_trail(samples.get('escalated'), 'Escalated case (bounded, stopped safely)')}
    {render_case_trail(samples.get('policy_violation'), 'AI recommendation rejected by policy guard') if samples.get('policy_violation') else ''}
  </section>

  <footer>
    Generated from a benchmark JSON produced by run_benchmark.py &mdash; no numbers on this page are computed here.
    <span class="mono">python run_benchmark.py --n {meta['n']} --seed {meta['seed']} --llm-client {meta['llm_client']}</span>
    <span class="mono">python report.py --in benchmark_report.json --out report.html</span>
  </footer>
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=str, default="benchmark_report.json")
    parser.add_argument("--out", type=str, default="report.html")
    args = parser.parse_args()

    with open(args.infile) as f:
        report = json.load(f)

    html_out = render(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
