"""LangGraph entry point with a deterministic fallback runner."""

from __future__ import annotations

from src.agents.nodes import AgentState, anomaly_node, extract_node, recommend_node, score_node, validate_node


def build_graph():
    """Build the LangGraph state machine when langgraph is installed."""

    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None

    graph = StateGraph(AgentState)
    graph.add_node("extract_node", extract_node)
    graph.add_node("validate_node", validate_node)
    graph.add_node("anomaly_node", anomaly_node)
    graph.add_node("score_node", score_node)
    graph.add_node("recommend_node", recommend_node)
    graph.add_edge(START, "extract_node")
    graph.add_edge("extract_node", "validate_node")
    graph.add_edge("validate_node", "anomaly_node")
    graph.add_edge("anomaly_node", "score_node")
    graph.add_edge("score_node", "recommend_node")
    graph.add_edge("recommend_node", END)
    return graph.compile()


async def run_agent(source_row: dict) -> AgentState:
    """Run the graph, falling back to direct node execution if LangGraph is unavailable."""

    from src.integrations.mlflow_tracing import MLflowTraceLogger

    tracer = MLflowTraceLogger()
    state: AgentState = {
        "raw_text": " ".join(str(source_row.get(key, "") or "") for key in ["description", "procedure", "equipment", "capability"]),
        "source_row_id": str(source_row.get("source_row_id") or source_row.get("id") or "row_unknown"),
        "source_row": source_row,
        "citations": {},
        "retries": 0,
    }
    with tracer.run(f"facility_{state['source_row_id']}") as run_id:
        state["mlflow_run_id"] = run_id
        graph = build_graph()
        if graph is not None:
            final_state = await graph.ainvoke(state)
            tracer.log_step(run_id, "langgraph", {"source_row_id": state["source_row_id"]}, {"state_keys": list(final_state.keys())}, _all_citations(final_state))
            _log_final_metrics(tracer, final_state)
            return final_state
        for node in [extract_node, validate_node, anomaly_node, score_node, recommend_node]:
            prev_state = dict(state)
            state = await node(state)
            step_name = node.__name__.replace("_node", "")
            tracer.log_step(
                run_id=run_id,
                step=step_name,
                inputs={"state_keys": list(prev_state.keys())},
                outputs={"new_keys": [key for key in state if key not in prev_state or state[key] != prev_state[key]]},
                citations=_step_citations(state, step_name),
            )
        _log_final_metrics(tracer, state)
    return state


def _step_citations(state: AgentState, step_name: str) -> list[str]:
    step_numbers = {"extract": "step_1_extraction", "validate": "step_2_validation", "anomaly": "step_3_anomaly", "score": "step_4_scoring", "recommend": "step_5_recommendation"}
    return state.get("citations", {}).get(step_numbers.get(step_name, ""), [])


def _all_citations(state: AgentState) -> list[str]:
    return sorted({source for sources in state.get("citations", {}).values() for source in sources})


def _log_final_metrics(tracer: MLflowTraceLogger, state: AgentState) -> None:
    if tracer.mlflow is None:
        return
    tracer.mlflow.log_metric("anomaly_count", len(state.get("anomalies", [])))
    tracer.mlflow.log_metric("care_index_score", state.get("care_index", {}).get("score", 0))
    tracer.mlflow.log_metric("confidence", state.get("extracted_data", {}).get("confidence_scores", {}).get("overall", 0))
    tracer.mlflow.log_dict(state.get("citations", {}), "citations.json")
