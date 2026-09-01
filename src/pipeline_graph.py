"""LangGraph orchestration: Stage1 -> Stage2 -> Stage3 -> Stage4 -> RuleLearner.
Each node passes forward only what the next stage needs (unmatched payments +
remaining bank entries), which keeps state small and every hop auditable."""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from stage1_deterministic import run_stage1
from stage2_rule_engine import run_stage2, RuleLearner
from stage3_llm_match import run_stage3
from stage4_verifier import run_stage4


class ReconState(TypedDict):
    payments: List[Dict[str, Any]]
    bank_entries: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]
    exceptions: List[Dict[str, Any]]
    unresolved_payments: List[Dict[str, Any]]
    remaining_bank: List[Dict[str, Any]]
    learner: Optional[RuleLearner]
    rule_history: List[Dict[str, Any]]
    stage3_matched: List[Dict[str, Any]]


def node_stage1(state: ReconState) -> ReconState:
    matched, unmatched, remaining_bank = run_stage1(state["payments"], state["bank_entries"])
    state["matches"].extend(matched)
    state["unresolved_payments"] = unmatched
    state["remaining_bank"] = remaining_bank
    return state


def node_stage2(state: ReconState) -> ReconState:
    learner = state["learner"]
    fit_summary = learner.fit(state["matches"],
                               {p["payment_id"]: p for p in state["payments"]},
                               {b["bank_entry_id"]: b for b in state["bank_entries"]})
    state["rule_history"].append(fit_summary)

    matched, unmatched, remaining_bank = run_stage2(
        state["unresolved_payments"], state["remaining_bank"], learner)
    state["matches"].extend(matched)
    state["unresolved_payments"] = unmatched
    state["remaining_bank"] = remaining_bank
    return state


def node_stage3(state: ReconState) -> ReconState:
    matched, exceptions, remaining_bank = run_stage3(
        state["unresolved_payments"], state["remaining_bank"])
    state["stage3_matched"] = matched  # held for verifier, not yet confirmed
    state["exceptions"].extend(exceptions)
    state["remaining_bank"] = remaining_bank
    return state


def node_stage4(state: ReconState) -> ReconState:
    payments_by_id = {p["payment_id"]: p for p in state["payments"]}
    bank_by_id = {b["bank_entry_id"]: b for b in state["bank_entries"]}
    confirmed, rejected = run_stage4(
        state.get("stage3_matched", []), payments_by_id, bank_by_id, state["remaining_bank"])
    state["matches"].extend(confirmed)
    state["exceptions"].extend(rejected)

    # Re-fit the rule learner on the newly confirmed evidence (feedback loop)
    learner = state["learner"]
    fit_summary = learner.fit(state["matches"], payments_by_id, bank_by_id)
    state["rule_history"].append(fit_summary)
    return state


def build_graph():
    graph = StateGraph(ReconState)
    graph.add_node("stage1_deterministic", node_stage1)
    graph.add_node("stage2_learned_rules", node_stage2)
    graph.add_node("stage3_llm_reasoning", node_stage3)
    graph.add_node("stage4_adversarial_verify", node_stage4)

    graph.set_entry_point("stage1_deterministic")
    graph.add_edge("stage1_deterministic", "stage2_learned_rules")
    graph.add_edge("stage2_learned_rules", "stage3_llm_reasoning")
    graph.add_edge("stage3_llm_reasoning", "stage4_adversarial_verify")
    graph.add_edge("stage4_adversarial_verify", END)

    return graph.compile()


def run_pipeline(payments, bank_entries):
    app = build_graph()
    initial_state: ReconState = {
        "payments": payments,
        "bank_entries": bank_entries,
        "matches": [],
        "exceptions": [],
        "unresolved_payments": [],
        "remaining_bank": bank_entries,
        "learner": RuleLearner(),
        "rule_history": [],
        "stage3_matched": [],
    }
    final_state = app.invoke(initial_state)
    return final_state
