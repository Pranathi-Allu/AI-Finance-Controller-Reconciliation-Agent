"""Runs the full reconciliation agent end-to-end and reports HONEST metrics
against labeled ground truth: match rate, precision, recall, and the
unresolved exception list (never hidden or cherry-picked)."""
import json
import os
from rich.console import Console
from rich.table import Table

from audit import init_db
from pipeline_graph import run_pipeline

console = Console()
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def load_data():
    with open(f"{DATA_DIR}/razorpay_payments.json") as f:
        payments = json.load(f)
    with open(f"{DATA_DIR}/bank_statement.json") as f:
        bank_entries = json.load(f)
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    return payments, bank_entries, ground_truth


def score(matches, exceptions, ground_truth):
    predicted = {m["payment_id"]: m["bank_entry_id"] for m in matches}

    tp = fp = fn = 0
    for pay_id, true_bank_id in ground_truth.items():
        pred_bank_id = predicted.get(pay_id)
        if true_bank_id is not None:
            if pred_bank_id == true_bank_id:
                tp += 1
            else:
                fn += 1
        else:
            if pred_bank_id is not None:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    match_rate = len(matches) / len(ground_truth)

    return {
        "total_payments": len(ground_truth),
        "matched": len(matches),
        "match_rate": round(match_rate, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exceptions_flagged": len(exceptions),
    }


def main():
    init_db()
    payments, bank_entries, ground_truth = load_data()

    console.print(f"\n[bold]Loaded[/bold] {len(payments)} payments, {len(bank_entries)} bank entries\n")

    state = run_pipeline(payments, bank_entries)

    matches = state["matches"]
    exceptions = state["exceptions"]
    metrics = score(matches, exceptions, ground_truth)

    stage_counts = {}
    for m in matches:
        stage_counts[m["stage"]] = stage_counts.get(m["stage"], 0) + 1

    t1 = Table(title="Matches by stage")
    t1.add_column("Stage")
    t1.add_column("Matches", justify="right")
    for stage, count in stage_counts.items():
        t1.add_row(stage, str(count))
    console.print(t1)

    t2 = Table(title="Honest evaluation against held-out ground truth")
    t2.add_column("Metric")
    t2.add_column("Value")
    for k, v in metrics.items():
        t2.add_row(k, str(v))
    console.print(t2)

    console.print("\n[bold red]Exception list (not resolved, not hidden):[/bold red]")
    for e in exceptions:
        console.print(f"  - {e['payment_id']}: {e['reason']}")

    console.print(f"\n[bold]Rule learner history[/bold] ({len(state['rule_history'])} refits):")
    for i, r in enumerate(state["rule_history"]):
        console.print(f"  refit {i+1}: support={r['support']}, "
                       f"fee_rate={r['fee_rate_mean']}±{r['fee_rate_std']}, "
                       f"lag_max={r['lag_days_max']}d")

    with open(os.path.join(ROOT_DIR, "results.json"), "w") as f:
        json.dump({"metrics": metrics, "matches": matches, "exceptions": exceptions,
                   "rule_history": state["rule_history"]}, f, indent=2)
    console.print("\n[green]Full results written to results.json[/green]")


if __name__ == "__main__":
    main()
