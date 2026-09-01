"""Stage 2 — learned rule engine.

The key differentiator: instead of hardcoding "fee is ~2.36%, lag is ~1 day",
the RuleLearner DERIVES these parameters statistically from Stage 1's
zero-ambiguity exact-UTR matches, and re-derives (refines) them again later
from confirmed Stage 3/4 outcomes. This mirrors the decision-tree-from-evidence
pattern: rules are learned and validated, not authored by hand.
"""
import statistics
from datetime import datetime
from audit import log_decision


class RuleLearner:
    def __init__(self):
        self.fee_rate_mean = 0.0236
        self.fee_rate_std = 0.004
        self.lag_days_max = 2
        self.history = []  # log of (rule params, support size, validation precision)

    def fit(self, confirmed_pairs, payments_by_id, bank_by_id):
        """Derive fee-rate and settlement-lag distribution from confirmed matches."""
        fee_rates, lags = [], []
        for pair in confirmed_pairs:
            pay = payments_by_id[pair["payment_id"]]
            bank = bank_by_id[pair["bank_entry_id"]]
            implied_fee_rate = 1 - (bank["amount"] / pay["amount"])
            fee_rates.append(implied_fee_rate)
            d1 = datetime.fromisoformat(pay["captured_at"]).date()
            d2 = datetime.fromisoformat(bank["date"]).date()
            lags.append((d2 - d1).days)

        if len(fee_rates) >= 5:
            self.fee_rate_mean = statistics.mean(fee_rates)
            self.fee_rate_std = max(statistics.stdev(fee_rates), 0.001)
            self.lag_days_max = max(2, int(statistics.quantiles(lags, n=20)[18]))  # ~95th pct

        self.history.append({
            "support": len(confirmed_pairs),
            "fee_rate_mean": round(self.fee_rate_mean, 5),
            "fee_rate_std": round(self.fee_rate_std, 5),
            "lag_days_max": self.lag_days_max,
        })
        return self.history[-1]

    def score(self, payment, bank_entry):
        """Returns confidence 0-1 that this pair matches, under the current rule."""
        implied_fee_rate = 1 - (bank_entry["amount"] / payment["amount"])
        fee_z = abs(implied_fee_rate - self.fee_rate_mean) / self.fee_rate_std
        if fee_z > 4:
            return 0.0  # amount is nowhere close, reject outright

        d1 = datetime.fromisoformat(payment["captured_at"]).date()
        d2 = datetime.fromisoformat(bank_entry["date"]).date()
        lag = (d2 - d1).days
        if lag < 0 or lag > self.lag_days_max:
            return 0.0

        amount_score = max(0.0, 1 - min(fee_z / 4, 1.0))
        lag_score = max(0.0, 1 - (lag / max(self.lag_days_max, 1)) * 0.3)
        ref_bonus = 0.15 if payment["order_id"][-6:] in bank_entry["narration"] else 0.0

        return min(1.0, 0.55 * amount_score + 0.30 * lag_score + ref_bonus)


def run_stage2(payments, bank_entries, learner, confidence_threshold=0.75):
    payments_by_id = {p["payment_id"]: p for p in payments}
    bank_by_id = {b["bank_entry_id"]: b for b in bank_entries}

    matched_pairs = []
    matched_bank_ids = set()
    still_unmatched = []

    for pay in payments:
        candidates = []
        for bank in bank_entries:
            if bank["bank_entry_id"] in matched_bank_ids:
                continue
            conf = learner.score(pay, bank)
            if conf > 0:
                candidates.append((conf, bank))

        candidates.sort(key=lambda x: -x[0])
        if candidates and candidates[0][0] >= confidence_threshold:
            # Guard against the near-duplicate trap: if two candidates are
            # nearly tied, don't auto-commit — push to LLM reasoning instead.
            top_conf = candidates[0][0]
            second_conf = candidates[1][0] if len(candidates) > 1 else 0.0
            if top_conf - second_conf < 0.08 and second_conf > 0.4:
                still_unmatched.append(pay)
                continue

            bank = candidates[0][1]
            rule_desc = (f"fee_rate~{learner.fee_rate_mean:.4f}±{learner.fee_rate_std:.4f}, "
                         f"lag<={learner.lag_days_max}d")
            matched_pairs.append({
                "payment_id": pay["payment_id"],
                "bank_entry_id": bank["bank_entry_id"],
                "stage": "stage2_learned_rule",
                "confidence": round(top_conf, 3),
                "rationale": f"Learned rule matched on fee-adjusted amount + settlement lag ({rule_desc}).",
            })
            matched_bank_ids.add(bank["bank_entry_id"])
            log_decision(pay["payment_id"], bank["bank_entry_id"], "stage2_learned_rule",
                         "match", top_conf, rule_desc, rule_fired=rule_desc)
        else:
            still_unmatched.append(pay)

    remaining_bank = [b for b in bank_entries if b["bank_entry_id"] not in matched_bank_ids]
    return matched_pairs, still_unmatched, remaining_bank
