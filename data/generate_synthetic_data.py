"""
Generates a realistic 3-way reconciliation dataset:
  - Razorpay payments/settlements (source A)
  - Bank statement entries (source B)
  - GST invoices (source C)

Deliberately injects the hard cases real reconciliation agents must handle:
  - exact UTR match (easy)
  - fee-adjusted amount, normal settlement lag (needs learned rule)
  - unusual settlement lag beyond normal window (ambiguous -> LLM stage)
  - narration with no clean reference (ambiguous -> LLM stage)
  - genuinely MISSING bank entry (real revenue leakage -> exception)
  - duplicate / split settlement (trap for false positives -> verifier catches)
  - two payments with near-identical amounts on the same day (matching trap)

Ground truth labels are stored separately so we can score the agent honestly.
"""
import json
import os
import random
import string
from datetime import datetime, timedelta

random.seed(42)

N_PAYMENTS = 60
FEE_RATE = 0.0236  
OUT_DIR = os.path.dirname(__file__)


def rand_id(prefix, n=14):
    return prefix + "".join(random.choices(string.ascii_letters + string.digits, k=n))


def rand_utr():
    return "UTR" + "".join(random.choices(string.digits, k=12))


def make_dataset():
    payments = []
    bank_entries = []
    invoices = []
    ground_truth = {}  # payment_id -> bank_entry_id or None (None = should be an exception)

    base_date = datetime(2026, 8, 1)

    for i in range(N_PAYMENTS):
        pay_id = rand_id("pay_")
        order_id = rand_id("order_")
        amount = round(random.uniform(299, 24999), 2)
        fee = round(amount * FEE_RATE, 2)
        net_amount = round(amount - fee, 2)
        utr = rand_utr()
        captured_at = base_date + timedelta(days=random.randint(0, 20), hours=random.randint(0, 23))
        method = random.choice(["upi", "card", "netbanking", "wallet"])

        payments.append({
            "payment_id": pay_id,
            "order_id": order_id,
            "amount": amount,
            "fee": fee,
            "net_amount": net_amount,
            "currency": "INR",
            "method": method,
            "utr": utr,
            "captured_at": captured_at.isoformat(),
            "status": "captured",
        })

        invoices.append({
            "invoice_id": rand_id("inv_"),
            "order_id": order_id,
            "gst_amount": round(amount * 0.18 / 1.18, 2),
            "total_amount": amount,
            "date": captured_at.date().isoformat(),
        })

        # Assign a scenario for the bank-side entry
        r = random.random()
        settle_lag = timedelta(days=1, hours=random.randint(0, 12))  # normal case

        if r < 0.35:
            scenario = "exact_utr"  # Stage 1 should catch this
        elif r < 0.65:
            scenario = "fee_adjusted_normal"  # Stage 2 (learned rule) should catch this
        elif r < 0.80:
            scenario = "unusual_lag"  # ambiguous -> Stage 3 LLM
            settle_lag = timedelta(days=random.randint(4, 7))
        elif r < 0.90:
            scenario = "messy_narration"  # ambiguous -> Stage 3 LLM
        elif r < 0.96:
            scenario = "missing"  # genuine revenue leakage -> exception, no bank entry created
        else:
            scenario = "near_duplicate_trap"  # two close amounts same day -> verifier must catch

        settled_at = captured_at + settle_lag

        if scenario == "missing":
            ground_truth[pay_id] = None
            continue

        if scenario == "exact_utr":
            narration = f"NEFT CR {utr} RAZORPAY SETL"
        elif scenario == "messy_narration":
            narration = f"RAZRPY SETL {rand_id('', 6).upper()}"
        else:
            narration = f"RZP/{order_id[-6:]}/SETL {rand_id('', 4).upper()}"

        bank_id = rand_id("bank_")
        bank_entries.append({
            "bank_entry_id": bank_id,
            "date": settled_at.date().isoformat(),
            "narration": narration,
            "amount": net_amount,
            "reference_number": utr if scenario == "exact_utr" else rand_id("REF", 8),
        })
        ground_truth[pay_id] = bank_id

        if scenario == "near_duplicate_trap":
            # A decoy bank entry with an almost-identical net amount, same day,
            # unrelated to this payment. Tests whether the agent falsely matches it.
            decoy_id = rand_id("bank_")
            bank_entries.append({
                "bank_entry_id": decoy_id,
                "date": settled_at.date().isoformat(),
                "narration": f"RZP/{rand_id('', 6).upper()}/SETL DECOY",
                "amount": round(net_amount + random.uniform(-1.5, 1.5), 2),
                "reference_number": rand_id("REF", 8),
            })

    with open(f"{OUT_DIR}/razorpay_payments.json", "w") as f:
        json.dump(payments, f, indent=2)
    with open(f"{OUT_DIR}/bank_statement.json", "w") as f:
        json.dump(bank_entries, f, indent=2)
    with open(f"{OUT_DIR}/invoices.json", "w") as f:
        json.dump(invoices, f, indent=2)
    with open(f"{OUT_DIR}/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(payments)} payments, {len(bank_entries)} bank entries, "
          f"{len(invoices)} invoices.")
    print(f"Ground truth: {sum(1 for v in ground_truth.values() if v)} should-match, "
          f"{sum(1 for v in ground_truth.values() if v is None)} genuine exceptions.")


if __name__ == "__main__":
    make_dataset()
