# AI Finance Controller — Self-Learning Reconciliation Agent

This is A multi-source payment reconciliation agent. It matches Razorpay payments against bank statement
entries — the everyday, tedious job every finance team does by hand — but
instead of one fuzzy-match pass, it escalates through four stages, only
spending AI compute on the pairs that actually need it, and it never trusts
its own answer without a second, adversarial check.

---

## Table of contents

- [What problem this solves](#what-problem-this-solves)
- [How the pipeline works](#how-the-pipeline-works)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running everything](#running-everything)
- [The dashboard](#the-dashboard)
- [Switching to a real LLM (Groq, open-source models)](#switching-to-a-real-llm-groq-open-source-models)
- [Using real Razorpay test-mode data](#using-real-razorpay-test-mode-data)
- [How scoring works](#how-scoring-works)
- [Extending this](#extending-this)

---

## What problem this solves

Reconciliation is: "does this Razorpay payment correspond to this line in my
bank statement?" It sounds trivial until you look at real data:

- Razorpay deducts a fee before settling, so the amount that lands in the
  bank is _not_ the amount the customer paid — the fee has to be inferred.
- Settlement lands 1–2 days later, sometimes longer, so dates don't line up.
- Bank narrations are often garbage — truncated references, generic text,
  no clean order ID.
- Some payments never show up on the bank side at all (real revenue leakage
  that needs to be _flagged_, not silently dropped).
- Two payments can have near-identical amounts on the same day — a classic
  trap for any naive fuzzy-matcher, which will confidently match the wrong
  pair and never know it.

Most reconciliation tools hardcode a rule ("match if amount within 2%, date
within 3 days") and call it done. This agent instead:

1. Matches for free wherever it can (exact references).
2. **Learns** its fee/lag tolerances from that first batch of certain
   matches, instead of a human guessing thresholds up front.
3. Only escalates genuinely ambiguous pairs to an LLM — and bounds that
   cost by only showing it the top-K plausible candidates, never all pairs.
4. Runs every LLM match past a second, adversarial LLM call whose only job
   is to try to prove it wrong.
5. Reports **honest metrics** against labeled ground truth — precision,
   recall, F1 — and an exception list of what it genuinely couldn't
   resolve, instead of quietly forcing matches to inflate its numbers.

---

## How the pipeline works

The four stages run in strict sequence, each one only receiving what the
previous stage failed to resolve (`unresolved_payments` + `remaining_bank`).
This is orchestrated as a small state machine with **LangGraph**
(`src/pipeline_graph.py`):

```
payments + bank_entries
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1 — Deterministic match   (src/stage1_deterministic.py)
│ Exact UTR / reference-number match against bank narration.
│ Free, instant, 100% precision by construction.
└─────────────────────────────────────────────────────────┘
        │ unmatched
        ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2 — Learned rule engine   (src/stage2_rule_engine.py)
│ A RuleLearner statistically DERIVES the fee-rate distribution
│ (mean/std) and the settlement-lag ceiling from Stage 1's
│ zero-ambiguity matches — nothing is hardcoded. It scores every
│ remaining pair against that learned rule and auto-matches high
│ -confidence pairs, while explicitly refusing near-tied
│ candidates (the "two payments, same amount, same day" trap) —
│ those get pushed forward instead of guessed.
└─────────────────────────────────────────────────────────┘
        │ still unresolved
        ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3 — LLM reasoning match   (src/stage3_llm_match.py)
│ Only the genuinely ambiguous pairs reach here. For each
│ unresolved payment, the top-K (default 3) candidate bank
│ entries by amount proximity are sent to an LLM, which reasons
│ about fee plausibility, date lag, and narration overlap, and
│ returns a structured {verdict, confidence, rationale}.
│ Falls back to a transparent heuristic mock if no LLM key is
│ set, so the whole pipeline is testable offline.
└─────────────────────────────────────────────────────────┘
        │ proposed matches (not yet trusted)
        ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 4 — Adversarial verifier   (src/stage4_verifier.py)
│ A second LLM call whose ONLY incentive is to find reasons
│ Stage 3's match is WRONG: a coincidental amount, an
│ implausible fee/lag, or a stronger unclaimed candidate nearby.
│ Only matches that survive this cross-examination are confirmed.
│ This is what keeps the precision numbers honest instead of
│ self-graded.
└─────────────────────────────────────────────────────────┘
        │
        ▼
  RuleLearner refits AGAIN on the newly confirmed Stage 4
  evidence — a live feedback loop, not a one-shot fit.
        │
        ▼
  Final matches + exception list + rule-fit history
  → scored against ground_truth.json → results.json
```

Every single decision — which stage handled it, what rule or model verdict
fired, the confidence, the rationale — is logged with a timestamp to a
SQLite audit trail (`src/audit.py` → `audit_trail.db`), independent of
`results.json`. That's what the dashboard's "raw audit log" panel reads
from.

---

## Project structure

```
reconciliation-agent/
├── data/
│   ├── generate_synthetic_data.py   # builds a labeled 60-payment test set
│   │                                  with injected edge cases (missing
│   │                                  bank entries, near-duplicate amount
│   │                                  traps, messy narrations)
│   ├── fetch_razorpay_testmode.py   # pulls real Razorpay test-mode payments
│   ├── razorpay_payments.json       # generated: source A
│   ├── bank_statement.json          # generated: source B
│   ├── invoices.json                # generated: source C (bonus, GST-style)
│   └── ground_truth.json            # generated: correct payment→bank labels
├── src/
│   ├── audit.py                     # SQLite audit trail (full decision log)
│   ├── stage1_deterministic.py      # Stage 1
│   ├── stage2_rule_engine.py        # Stage 2 — RuleLearner, the core differentiator
│   ├── stage3_llm_match.py          # Stage 3 — LLM reasoning (mock or real)
│   ├── stage4_verifier.py           # Stage 4 — adversarial verifier (mock or real)
│   ├── pipeline_graph.py            # LangGraph orchestration of all 4 stages
│   └── run_pipeline.py              # entry point: loads data, runs pipeline, scores it
├── dashboard/
│   └── app.py                       # Streamlit dashboard (can also trigger a run)
├── requirements.txt
├── results.json                     # written after each run (metrics + matches + exceptions)
├── audit_trail.db                   # written after each run (full decision log)
└── README.md
```

---

## Setup

Requires Python 3.10+.

```bash
git clone <this repo>          # or just unzip it
cd reconciliation-agent
pip install -r requirements.txt          # Windows / macOS
# or, on Linux with an externally-managed Python:
pip install -r requirements.txt --break-system-packages
```

(A virtual environment is recommended but not required — mock mode has no
external dependencies beyond the packages in `requirements.txt`.)

---

## Running everything

**1. Generate the labeled test dataset** (deterministic — same data every
time thanks to a fixed random seed):

```bash
python data/generate_synthetic_data.py
```

This writes `data/razorpay_payments.json`, `data/bank_statement.json`,
`data/invoices.json`, and `data/ground_truth.json`.

**2. Run the pipeline** (works immediately, no API key needed — Stages 3/4
fall back to a transparent heuristic mock):

```bash
cd src
python run_pipeline.py
```

You'll see, printed live and also written to `results.json`:

- a stage-by-stage match breakdown
- honest precision / recall / F1 against `ground_truth.json`
- the exception list (payments that could not be resolved — never hidden)
- the rule learner's fit history (fee-rate and lag window at each refit)

**3. Launch the dashboard:**

```bash
cd ..
streamlit run dashboard/app.py
```

---

## The dashboard

`streamlit run dashboard/app.py` opens a local web UI with:

- **Headline metrics** — match rate, precision, recall, F1, exception count
- **Matches by stage** — bar chart showing how much Stage 1+2 resolve for
  free versus what actually needs the LLM
- **Rule learner evolution** — the fee-rate/lag values at each refit, so
  you can show it isn't hardcoded
- **Audit trail table** — filterable by stage; click any row to expand its
  full reasoning in a card below
- **Exception queue** — what's genuinely unresolved, and why
- **Raw audit log** (expander) — the full SQLite `audit_trail.db` output,
  with timestamps and which rule fired for every single decision
- **Sidebar: "▶ Run reconciliation now"** — re-runs the whole pipeline from
  the browser (toggle mock vs. real LLM, optionally paste a Groq key for
  that run) instead of needing a separate terminal

If `results.json` doesn't exist yet, the dashboard tells you to run the
pipeline first (or just click the sidebar button).

---

## Switching to a real LLM (Groq, open-source models)

Stages 3 and 4 auto-detect an API key and switch from the heuristic mock to
real calls against **`openai/gpt-oss-120b`** — OpenAI's open-weight 120B
model, served free and fast via [Groq](https://console.groq.com/keys) over
an OpenAI-compatible API. No Anthropic key, no paid API required.

```bash
export GROQ_API_KEY=gsk_...          # macOS/Linux
$env:GROQ_API_KEY="gsk_..."          # Windows PowerShell

cd src
python run_pipeline.py
```

Want a different open model? Every model Groq hosts works the same way —
just override the env var:

```bash
export GROQ_MODEL=qwen/qwen3.6-27b
# or: llama-3.3-70b-versatile, moonshotai/kimi-k2-instruct, etc.
```

Prefer running fully offline instead of Groq's hosted API? Point the same
`OpenAI(...)` client in `stage3_llm_match.py` / `stage4_verifier.py` at a
local [Ollama](https://ollama.com) server
(`base_url="http://localhost:11434/v1"`, e.g. model `llama3.1` or
`qwen2.5`) — the rest of the code is unchanged, since Ollama also speaks
the OpenAI-compatible chat completions API.

**Sanity check you're actually hitting the real model** (mock responses
always end their rationale with `[mock LLM]` / `[mock verifier]`):

```bash
grep -c mock results.json     # macOS/Linux — should print 0
Select-String -Path results.json -Pattern "mock"   # Windows PowerShell
```

---

## Using real Razorpay test-mode data

To satisfy a "must use test-mode APIs" requirement, or just to demo against
genuine data instead of the synthetic generator:

```bash
export RAZORPAY_KEY_ID=rzp_test_...
export RAZORPAY_KEY_SECRET=...
python data/fetch_razorpay_testmode.py
```

This pulls your actual captured test-mode payments and converts them to the
same schema the pipeline consumes. Razorpay's API doesn't expose bank
statements, so you'll still need bank-side data — pair the live payments
with the synthetic generator's bank-entry logic for a realistic end-to-end
demo that's still grounded in real Razorpay data. See the docstring in
`fetch_razorpay_testmode.py` for the full setup (creating a test account,
generating test keys, making test payments).

---

## How scoring works

`run_pipeline.py` compares predicted matches against `ground_truth.json`
(which the generator writes alongside the dataset, so it's never
self-graded):

| Metric               | Meaning                                                               |
| -------------------- | --------------------------------------------------------------------- |
| `match_rate`         | fraction of all payments the pipeline produced _any_ match for        |
| `true_positives`     | predicted the exact correct bank entry                                |
| `false_positives`    | matched a payment the ground truth says should have been an exception |
| `false_negatives`    | matched a payment to the wrong bank entry (or missed it)              |
| `precision`          | `tp / (tp + fp)`                                                      |
| `recall`             | `tp / (tp + fn)`                                                      |
| `f1`                 | harmonic mean of precision and recall                                 |
| `exceptions_flagged` | payments honestly reported as unresolved                              |

---

## Extending this

- Swap `stage2_rule_engine.RuleLearner` for a proper decision-tree fit if
  you have more historical data — the interface (`fit`, `score`) is already
  decision-tree-shaped.
- `invoices.json` is already generated but unused — add a third reconciliation
  leg (GST tax-line matching) as a bonus feature beyond 2-way matching.
- Add cost/latency tracking around the Stage 3/4 LLM calls (token count,
  Groq's response time) and surface it in the dashboard — strengthens the
  "cost-bounded by design" story since only ambiguous pairs ever reach the
  LLM.
- The README previously floated wrapping each stage as a `SKILL.md`-based
  Deep Agents skill for dynamic skill discovery instead of a fixed LangGraph
  node. This isn't recommended here: Deep Agents' value is in letting an
  agent _choose_ which tool to invoke at runtime from a growing library.
  This pipeline is the opposite — a fixed, ordered 4-stage escalation where
  the routing logic itself is the point (it's what makes the precision/recall
  numbers honest and auditable). The current LangGraph state machine is the
  right fit; treat Deep Agents as an "extensibility" talking point rather
  than something to actually build here.
