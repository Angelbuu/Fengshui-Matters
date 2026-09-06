from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import (
    execute_navigation,
    handle_destination_decision,
)

from agent_state import AgentState

from llm_agent_destination import (
    DestinationDecision,
    build_resolver,
)


class GraphState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    user_text: str

    destination_decision: DestinationDecision

    agent_state: AgentState

    adapter: Any

    final_message: Optional[str]
    
    retry_count: int
    max_retries: int
    
def resolve_destination_node(
    state: GraphState,
) -> dict:
    """
    LangGraph node:
    understand where the visitor wants to go.
    """

    resolver = build_resolver()

    decision = resolver.resolve_destination(
        state["user_text"]
    )

    agent_state = handle_destination_decision(
        decision
    )
    
    print("\n--- B1 DEBUG ---")
    print("User text:", state["user_text"])
    print("Intent:", decision.intent)
    print("Destination:", decision.destination)
    print("Confidence:", decision.confidence)
    print("Needs clarification:", decision.needs_clarification)
    print("Candidates:", decision.candidates)
    print("Visitor message:", decision.visitor_message)

    return {
        "destination_decision": decision,
        "agent_state": agent_state,
    }
    
def route_after_destination(
    state: GraphState,
) -> str:
    """
    Choose the next graph node after B1.
    """

    agent_state = state["agent_state"]

    if agent_state.status == "READY_TO_NAVIGATE":
        return "navigate"

    if agent_state.status == "WAITING_FOR_CLARIFICATION":
        return "clarify"

    return "unsupported"

def navigate_node(
    state: GraphState,
) -> dict:
    """
    LangGraph node:
    execute closed-loop robot navigation.
    """

    agent_state = state["agent_state"]
    adapter = state["adapter"]

    updated_state = execute_navigation(
        agent_state,
        adapter,
    )

    return {
        "agent_state": updated_state,
    }
    
def clarify_node(
    state: GraphState,
) -> dict:

    decision = state["destination_decision"]

    return {
        "final_message": decision.visitor_message,
    }
    
def unsupported_node(
    state: GraphState,
) -> dict:

    decision = state["destination_decision"]

    return {
        "final_message": decision.visitor_message,
    }
    
def route_after_navigation(
    state: GraphState,
) -> str:

    agent_state = state["agent_state"]

    if agent_state.status == "ARRIVED":
        return "finish"

    if agent_state.human_assistance_required:
        return "human"

    return "failed"

def finish_node(
    state: GraphState,
) -> dict:

    agent_state = state["agent_state"]

    return {
        "final_message": (
            f"We've arrived at "
            f"{agent_state.destination}."
        )
    }
    
def failed_node(
    state: GraphState,
) -> dict:

    return {
        "final_message": (
            "I couldn't complete the navigation."
        )
    }


def human_node(
    state: GraphState,
) -> dict:

    return {
        "final_message": (
            "I need assistance to continue safely."
        )
    }

def recovery_node(state: GraphState) -> dict:
    retry_count = state.get("retry_count", 0) + 1

    return {
        "retry_count": retry_count,
        "final_message": (
            "Navigation failed. Attempting recovery."
        ),
    }
    
def route_after_recovery(state: GraphState) -> str:
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry"

    return "human"

def build_agent_graph():

    builder = StateGraph(GraphState)

    # -------------------------
    # Nodes
    # -------------------------

    builder.add_node(
        "resolve_destination",
        resolve_destination_node,
    )

    builder.add_node(
        "navigate",
        navigate_node,
    )

    builder.add_node(
        "clarify",
        clarify_node,
    )

    builder.add_node(
        "unsupported",
        unsupported_node,
    )

    builder.add_node(
        "finish",
        finish_node,
    )

    builder.add_node(
        "failed",
        failed_node,
    )

    builder.add_node(
        "human",
        human_node,
    )

    # -------------------------
    # Start
    # -------------------------

    builder.add_edge(
        START,
        "resolve_destination",
    )

    # -------------------------
    # B1 routing
    # -------------------------

    builder.add_conditional_edges(
        "resolve_destination",
        route_after_destination,
        {
            "navigate": "navigate",
            "clarify": "clarify",
            "unsupported": "unsupported",
        },
    )

    # -------------------------
    # Navigation routing
    # -------------------------

    builder.add_conditional_edges(
        "navigate",
        route_after_navigation,
        {
            "finish": "finish",
            "failed": "failed",
            "human": "human",
        },
    )

    # -------------------------
    # End states
    # -------------------------

    builder.add_edge(
        "finish",
        END,
    )

    builder.add_edge(
        "clarify",
        END,
    )

    builder.add_edge(
        "unsupported",
        END,
    )

    builder.add_edge(
        "failed",
        END,
    )

    builder.add_edge(
        "human",
        END,
    )

    return builder.compile()

if __name__ == "__main__":

    from control.adapters.sim_adapter import SimAdapter

    graph = build_agent_graph()

    adapter = SimAdapter(
        mode="fake"
    )

    try:

        result = graph.invoke({
            "user_text": (
                "I need to go somewhere for treatment."
            ),
            "adapter": adapter,
        })

        print("\n--- LANGGRAPH RESULT ---")

        print(
            "Destination:",
            result["agent_state"].destination,
        )

        print(
            "Status:",
            result["agent_state"].status,
        )

        print(
            "Message:",
            result["final_message"],
        )

    finally:

        adapter.close()