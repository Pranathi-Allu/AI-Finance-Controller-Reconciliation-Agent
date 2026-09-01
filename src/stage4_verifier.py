"""Stage 4 — adversarial verifier. A second agent whose only incentive is to
DISPROVE Stage 3's matches. This is the honesty check the track explicitly
asks for: instead of grading its own homework, the pipeline actively hunts
for its own false positives before they ever reach the audit trail as
confirmed matches.
"""
import json
import os
import re
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
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))

VERIFIER_SYSTEM_PROMPT = """You are an adversarial auditor reviewing a reconciliation
match made by another AI agent. Your ONLY job is to find reasons this match might be
WRONG — a coincidental amount match, an implausible fee rate, an implausible date gap,
or a stronger unclaimed candidate nearby. Be skeptical by default.

Respond with ONLY valid JSON:
{"upheld": true | false, "confidence": 0.0-1.0, "rationale": "one sentence"}
"""


def _mock_verify(payment, bank_entry, all_bank_entries, original_confidence):
    """Heuristic stand-in: specifically hunts for the near-duplicate trap —
    another unclaimed bank entry with an even closer fee-adjusted amount."""
    my_diff = abs(bank_entry["amount"] - payment["net_amount"])
    for other in all_bank_entries:
        if other["bank_entry_id"] == bank_entry["bank_entry_id"]:
            continue
        other_diff = abs(other["amount"] - payment["net_amount"])
        if other_diff < my_diff * 0.5 and other["date"] == bank_entry["date"]:
            return {"upheld": False, "confidence": 0.85,
                    "rationale": f"A closer unclaimed candidate ({other['bank_entry_id']}) exists "
                                  f"on the same date — likely false positive [mock verifier]"}
    if original_confidence < 0.65:
        return {"upheld": False, "confidence": 0.6,
                "rationale": "Original confidence too low to uphold without stronger evidence [mock verifier]"}
    return {"upheld": True, "confidence": 0.9,
            "rationale": "No stronger competing candidate found; fee/lag plausible [mock verifier]"}


def _real_verify(payment, bank_entry, all_bank_entries, original_confidence):
    payload = {
        "payment": payment, "proposed_bank_entry": bank_entry,
        "original_confidence": original_confidence,
        "other_unclaimed_entries_same_window": [
            b for b in all_bank_entries if b["bank_entry_id"] != bank_entry["bank_entry_id"]
        ][:5],
    }
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    text = resp.choices[0].message.content or ""
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        # Fail safe: an unparseable verdict means the verifier couldn't confirm,
        # so err toward rejecting rather than silently upholding a bad match.
        return {"upheld": False, "confidence": 0.0,
                "rationale": f"Verifier returned unparseable output, treated as rejected: {text[:150]!r}"}


def run_stage4(llm_matched_pairs, payments_by_id, bank_by_id, all_remaining_bank):
    confirmed = []
    rejected = []

    for pair in llm_matched_pairs:
        pay = payments_by_id[pair["payment_id"]]
        bank = bank_by_id[pair["bank_entry_id"]]
        verify_fn = _mock_verify if USE_MOCK else _real_verify
        result = verify_fn(pay, bank, all_remaining_bank, pair["confidence"])

        if result["upheld"]:
            confirmed.append(pair)
            log_decision(pair["payment_id"], pair["bank_entry_id"], "stage4_verifier",
                         "upheld", result["confidence"], result["rationale"])
        else:
            rejected.append({
                "payment_id": pair["payment_id"],
                "reason": f"Verifier rejected Stage 3 match: {result['rationale']}",
            })
            log_decision(pair["payment_id"], pair["bank_entry_id"], "stage4_verifier",
                         "rejected", result["confidence"], result["rationale"])

    return confirmed, rejected
