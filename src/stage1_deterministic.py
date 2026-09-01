"""Stage 1 — deterministic match. Free, instant, 100% precision.
If the bank narration or reference number contains the exact Razorpay UTR,
it's a match. No LLM call needed."""
from audit import log_decision


def run_stage1(payments, bank_entries):
    matched_pairs = []
    matched_bank_ids = set()
    unmatched_payments = []

    for pay in payments:
        found = None
        for bank in bank_entries:
            if bank["bank_entry_id"] in matched_bank_ids:
                continue
            if pay["utr"] and (pay["utr"] in bank["narration"] or pay["utr"] == bank["reference_number"]):
                found = bank
                break
        if found:
            matched_pairs.append({
                "payment_id": pay["payment_id"],
                "bank_entry_id": found["bank_entry_id"],
                "stage": "stage1_deterministic",
                "confidence": 1.0,
                "rationale": f"Exact UTR match: {pay['utr']} found in bank narration/reference.",
            })
            matched_bank_ids.add(found["bank_entry_id"])
            log_decision(pay["payment_id"], found["bank_entry_id"], "stage1_deterministic",
                         "match", 1.0, f"Exact UTR match: {pay['utr']}")
        else:
            unmatched_payments.append(pay)

    remaining_bank = [b for b in bank_entries if b["bank_entry_id"] not in matched_bank_ids]
    return matched_pairs, unmatched_payments, remaining_bank
