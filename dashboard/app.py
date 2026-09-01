"""Reconciliation Agent — live dashboard.
Run with: streamlit run dashboard/app.py
Shows: headline metrics, matches by stage, the rule learner's evolution,
a searchable + clickable audit trail, and a one-click pipeline run.
"""
import json
import subprocess
import sys
import os
import pandas as pd
import streamlit as st

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

st.set_page_config(page_title="Reconciliation Agent", layout="wide")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "audit_trail.db")

st.title("AI Finance Controller — Reconciliation Agent")
st.caption("Multi-source reconciliation with a self-learning rule engine and an adversarial verifier")

# --- Sidebar: run the pipeline without leaving the dashboard ---
with st.sidebar:
    st.header("Run pipeline")
    use_real = st.checkbox("Use real LLM (Groq)", value=bool(os.environ.get("GROQ_API_KEY")))
    groq_key_input = ""
    if use_real:
        has_env_key = bool(os.environ.get("GROQ_API_KEY"))
        if has_env_key:
            st.caption("✅ Using GROQ_API_KEY from your environment.")
        groq_key_input = st.text_input(
            "Override GROQ_API_KEY (optional)",
            value="",  # never pre-fill a real secret into the widget
            type="password",
            placeholder="Leave blank to use the environment variable" if has_env_key else "gsk_...",
        )
        if not has_env_key and not groq_key_input:
            st.warning("No GROQ_API_KEY found. Enter one above, or it will run in mock mode.")
    if st.button("▶ Run reconciliation now", use_container_width=True):
        env = os.environ.copy()
        if use_real:
            if groq_key_input:
                env["GROQ_API_KEY"] = groq_key_input
            elif "GROQ_API_KEY" not in env:
                st.error("Check 'Use real LLM' requires a GROQ_API_KEY.")
                st.stop()
        else:
            env.pop("GROQ_API_KEY", None)  # force mock mode for this run

        with st.spinner("Running 4-stage pipeline..."):
            result = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                cwd=SRC_DIR, env=env, capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("Pipeline finished — results refreshed below.")
            st.rerun()
        else:
            st.error("Pipeline failed. See details below.")
            st.code(result.stderr or result.stdout, language="text")

if not os.path.exists(RESULTS_PATH):
    st.warning("No results yet. Click **Run reconciliation now** in the sidebar, "
               "or run `python run_pipeline.py` from the `src/` folder.")
    st.stop()

with open(RESULTS_PATH) as f:
    results = json.load(f)

metrics = results["metrics"]
matches = results["matches"]
exceptions = results["exceptions"]
rule_history = results["rule_history"]

# --- Headline metrics ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Match rate", f"{metrics['match_rate']*100:.1f}%")
c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
c4.metric("F1", f"{metrics['f1']*100:.1f}%")
c5.metric("Exceptions", metrics["exceptions_flagged"])

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Matches by stage")
    stage_counts = {}
    for m in matches:
        stage_counts[m["stage"]] = stage_counts.get(m["stage"], 0) + 1
    df_stage = pd.DataFrame(
        [{"stage": k.replace("_", " "), "matches": v} for k, v in stage_counts.items()]
    )
    st.bar_chart(df_stage.set_index("stage"))
    st.caption("Stage 1 (deterministic) and 2 (learned rules) resolve the bulk for free. "
               "Only genuinely ambiguous pairs reach the LLM in stage 3.")

with col_right:
    st.subheader("Rule learner evolution")
    df_rules = pd.DataFrame(rule_history)
    df_rules.index = [f"refit {i+1}" for i in range(len(df_rules))]
    st.dataframe(df_rules, use_container_width=True)
    st.caption("The fee-rate and settlement-lag thresholds are DERIVED from confirmed matches, "
               "not hardcoded — and refined again after the verifier confirms stage 3 matches.")

st.divider()
st.subheader("Audit trail — filter by stage, click a row for full reasoning")

df_matches = pd.DataFrame(matches)
all_stages = sorted(df_matches["stage"].unique().tolist())
selected_stage = st.multiselect(
    "Filter by stage", options=all_stages, default=all_stages, key="stage_filter",
)
filtered = df_matches[df_matches["stage"].isin(selected_stage)].reset_index(drop=True)

st.caption(f"Showing **{len(filtered)}** of **{len(df_matches)}** matches")

event = st.dataframe(
    filtered[["payment_id", "bank_entry_id", "stage", "confidence", "rationale"]],
    use_container_width=True, height=300,
    on_select="rerun", selection_mode="single-row", key="match_table",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    row = filtered.iloc[selected_rows[0]]
    with st.container(border=True):
        st.markdown(f"**Payment** `{row['payment_id']}` → **Bank entry** `{row['bank_entry_id']}`")
        st.markdown(f"**Stage:** {row['stage'].replace('_', ' ')}  |  **Confidence:** {row['confidence']:.2f}")
        st.markdown(f"**Full reasoning:** {row['rationale']}")
else:
    st.caption("👆 Click any row above to expand its full reasoning here.")

st.divider()
st.subheader("Exception queue — honest, not hidden")
if exceptions:
    st.dataframe(pd.DataFrame(exceptions), use_container_width=True)
else:
    st.success("No unresolved exceptions in this batch.")

st.divider()
with st.expander("🗄 Raw audit log (SQLite `audit_trail.db`) — every decision, every stage, with timestamps"):
    if os.path.exists(DB_PATH):
        from audit import get_all_decisions
        decisions = get_all_decisions()
        if decisions:
            st.dataframe(pd.DataFrame(decisions), use_container_width=True, height=350)
        else:
            st.info("Audit DB exists but has no rows yet — run the pipeline first.")
    else:
        st.info("No audit_trail.db found yet. It's created the next time you run the pipeline.")