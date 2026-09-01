"""Stage 3 — LLM reasoning match. Only invoked for pairs Stage 1+2 couldn't
resolve confidently. Bounded and gated: the LLM can only emit a structured
verdict (never touch money, never auto-execute). Falls back to a transparent
heuristic mock when no GROQ_API_KEY is set, so the pipeline is fully
testable offline before your demo.

Real calls go to an open-weight model (default: openai/gpt-oss-120b) served
by Groq's free, OpenAI-compatible API — no Anthropic key required.
"""
import json
import os
import re
from datetime import datetime
from audit import log_decision

USE_MOCK = not os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

if not USE_MOCK:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


def _extract_json(text: str) -> dict:
    """gpt-oss and other reasoning models sometimes emit chain-of-thought or
    markdown fences around the JSON verdict even when told not to. Pull out
    the outermost {...} block rather than assuming the whole string is JSON."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))

SYSTEM_PROMPT = """You are a financial reconciliation reasoning agent for an Indian
payments merchant. You will be given ONE Razorpay payment and ONE candidate bank
statement entry. Decide whether they represent the same underlying transaction.

Consider: fee-adjusted amount plausibility (Razorpay deducts ~2-2.5% fees+GST before
settlement), settlement date lag (normally 1-2 days, occasionally longer), and any
overlap between the order ID / merchant reference and the bank narration text.

Respond with ONLY valid JSON, no other text:
{"verdict": "match" | "no_match", "confidence": 0.0-1.0, "rationale": "one sentence"}
"""


def _mock_llm_verdict(payment, bank_entry):
    """Transparent heuristic standing in for the LLM call during offline dev/testing."""
    implied_fee_rate = 1 - (bank_entry["amount"] / payment["amount"])
    d1 = datetime.fromisoformat(payment["captured_at"]).date()
    d2 = datetime.fromisoformat(bank_entry["date"]).date()
    lag = (d2 - d1).days
    ref_hit = payment["order_id"][-6:] in bank_entry["narration"]

    plausible_fee = 0.015 <= implied_fee_rate <= 0.04
    plausible_lag = 0 <= lag <= 8

    if plausible_fee and plausible_lag and (ref_hit or lag <= 3):
        conf = 0.72 if not ref_hit else 0.88
        return {"verdict": "match", "confidence": conf,
                "rationale": f"Fee-adjusted amount plausible ({implied_fee_rate:.3f}), "
                              f"lag={lag}d, ref_overlap={ref_hit} [mock LLM]"}
    return {"verdict": "no_match", "confidence": 0.3,
            "rationale": f"Fee rate {implied_fee_rate:.3f} or lag {lag}d outside plausible range [mock LLM]"}


def _real_llm_verdict(payment, bank_entry):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},  # gpt-oss: skip long chain-of-thought
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"payment": payment, "bank_entry": bank_entry})},
        ],
    )
    text = resp.choices[0].message.content or ""
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        # Fail safe: an unparseable verdict is treated as "no match" rather
        # than crashing the whole pipeline over one bad model response.
        return {"verdict": "no_match", "confidence": 0.0,
                "rationale": f"Model returned unparseable output, treated as no-match: {text[:150]!r}"}


def run_stage3(payments, bank_entries, confidence_threshold=0.6, top_k=3):
    matched_pairs = []
    matched_bank_ids = set()
    exceptions = []

    for pay in payments:
        # Cheap pre-filter: only send the top-K plausible candidates to the LLM
        # (never all-pairs — keeps cost and latency bounded)
        candidates = sorted(
            [b for b in bank_entries if b["bank_entry_id"] not in matched_bank_ids],
            key=lambda b: abs(b["amount"] - pay["net_amount"]),
        )[:top_k]

        best = None
        for bank in candidates:
            verdict = _mock_llm_verdict(pay, bank) if USE_MOCK else _real_llm_verdict(pay, bank)
            if verdict["verdict"] == "match" and verdict["confidence"] >= confidence_threshold:
                if best is None or verdict["confidence"] > best[0]["confidence"]:
                    best = (verdict, bank)

        if best:
            verdict, bank = best
            matched_pairs.append({
                "payment_id": pay["payment_id"],
                "bank_entry_id": bank["bank_entry_id"],
                "stage": "stage3_llm_reasoning",
                "confidence": verdict["confidence"],
                "rationale": verdict["rationale"],
            })
            matched_bank_ids.add(bank["bank_entry_id"])
            log_decision(pay["payment_id"], bank["bank_entry_id"], "stage3_llm_reasoning",
                         "match", verdict["confidence"], verdict["rationale"])
        else:
            exceptions.append({
                "payment_id": pay["payment_id"],
                "reason": "No candidate bank entry cleared the confidence threshold "
                          "after LLM reasoning over top matches by amount proximity.",
            })
            log_decision(pay["payment_id"], None, "stage3_llm_reasoning",
                         "unresolved", 0.0, "No candidate cleared threshold")

    remaining_bank = [b for b in bank_entries if b["bank_entry_id"] not in matched_bank_ids]
    return matched_pairs, exceptions, remaining_bank
