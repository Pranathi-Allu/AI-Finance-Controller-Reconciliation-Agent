"""Audit trail: every match/exception decision is logged with full provenance
so the dashboard can show WHY the agent decided what it decided."""
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "audit_trail.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS decisions")
    conn.execute("""
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payment_id TEXT,
            bank_entry_id TEXT,
            stage TEXT,
            verdict TEXT,
            confidence REAL,
            rationale TEXT,
            rule_fired TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_decision(payment_id, bank_entry_id, stage, verdict, confidence, rationale, rule_fired=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO decisions (payment_id, bank_entry_id, stage, verdict, confidence, "
        "rationale, rule_fired, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (payment_id, bank_entry_id, stage, verdict, confidence, rationale, rule_fired,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_decisions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]
