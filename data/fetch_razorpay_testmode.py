"""Pulls REAL test-mode payments from your Razorpay account and converts
them to the same schema the pipeline already consumes. Use this instead of
(or alongside) the synthetic generator for the actual demo, so the judges
see it running against genuine Razorpay test-mode API data.

Setup:
  1. Create a Razorpay account, switch to Test Mode.
  2. Generate test API keys: Settings -> API Keys -> Generate Test Key.
  3. Create a handful of test payments (Razorpay's test cards/UPI work in
     Test Mode: https://razorpay.com/docs/payments/payments/test-card-upi-details/)
  4. export RAZORPAY_KEY_ID=rzp_test_xxx
     export RAZORPAY_KEY_SECRET=xxx
  5. python3 data/fetch_razorpay_testmode.py
"""
import json
import os
import razorpay

OUT_DIR = os.path.dirname(__file__)


def fetch():
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise SystemExit(
            "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET (test-mode keys) before running this."
        )

    client = razorpay.Client(auth=(key_id, key_secret))
    raw_payments = client.payment.all({"count": 100})["items"]

    payments = []
    for p in raw_payments:
        if p["status"] != "captured":
            continue
        amount = p["amount"] / 100  # paise -> rupees
        fee = (p.get("fee") or 0) / 100
        payments.append({
            "payment_id": p["id"],
            "order_id": p.get("order_id") or p["id"],
            "amount": amount,
            "fee": fee,
            "net_amount": round(amount - fee, 2) if fee else round(amount * (1 - 0.0236), 2),
            "currency": p["currency"],
            "method": p["method"],
            # Razorpay test-mode payments don't always carry a bank UTR;
            # fall back to the payment id fragment for the deterministic-match demo path.
            "utr": p.get("acquirer_data", {}).get("utr") or f"UTR{p['id'][-9:]}",
            "captured_at": p["created_at"],
            "status": p["status"],
        })

    with open(f"{OUT_DIR}/razorpay_payments_live.json", "w") as f:
        json.dump(payments, f, indent=2)

    print(f"Fetched {len(payments)} captured test-mode payments -> razorpay_payments_live.json")
    print("Note: you'll still need a bank_statement.json — real bank statements aren't "
          "available via the Razorpay API. Pair this with a few manually-authored bank "
          "entries (or the synthetic generator's bank-entry logic) so the demo shows a "
          "REAL payment source reconciled against a realistic bank-side source.")


if __name__ == "__main__":
    fetch()
