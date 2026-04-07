"""
MedSimplify Agent Graph — the core orchestration.

This is where LangGraph shines. Instead of a simple sequential chain
(A → B → C), we have a GRAPH with conditional branching:

    parse → router → [lab_analyzer OR radiology_analyzer] → explainer → followup

Then a chat loop for follow-up questions.

Key LangGraph concepts used here:
1. StateGraph: a graph where nodes share a typed state dict
2. add_node: registers a function as a named node
3. add_edge: creates a fixed connection between nodes (A always goes to B)
4. add_conditional_edges: creates a branch (router decides which path)
5. START/END: special nodes marking entry and exit points
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes.chat import chat_node
from agent.nodes.explainer import explainer_node
from agent.nodes.followup import followup_node
from agent.nodes.lab_analyzer import lab_analyzer_node
from agent.nodes.parse import parse_node
from agent.nodes.radiology_analyzer import radiology_analyzer_node
from agent.nodes.router import route_by_type, router_node
from agent.state import MedSimplifyState


def build_analysis_graph() -> StateGraph:
    """Build the report analysis graph (parse → analyze → explain).

    This graph handles the initial report analysis pipeline.
    It does NOT include the chat loop — that's handled separately
    by the API layer, because the chat loop depends on user input
    between iterations (you can't pre-wire "wait for user message"
    in a graph that runs to completion).

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"raw_input": ..., "input_type": ..., ...})
    """
    # Create the graph with our state schema
    graph = StateGraph(MedSimplifyState)

    # --- Register all nodes ---
    graph.add_node("parse", parse_node)
    graph.add_node("router", router_node)
    graph.add_node("lab_analyzer", lab_analyzer_node)
    graph.add_node("radiology_analyzer", radiology_analyzer_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("followup", followup_node)

    # --- Wire the edges ---

    # Entry point: always start with parsing
    graph.add_edge(START, "parse")

    # After parsing, always route
    graph.add_edge("parse", "router")

    # Conditional branch: router decides lab or radiology
    # add_conditional_edges takes:
    #   1. Source node name
    #   2. Routing function (returns a string key)
    #   3. Mapping of {key: destination_node_name}
    graph.add_conditional_edges(
        "router",
        route_by_type,
        {
            "lab": "lab_analyzer",
            "radiology": "radiology_analyzer",
        },
    )

    # Both analyzer branches converge at the explainer
    graph.add_edge("lab_analyzer", "explainer")
    graph.add_edge("radiology_analyzer", "explainer")

    # After explanation, generate follow-up questions
    graph.add_edge("explainer", "followup")

    # End after follow-up questions
    graph.add_edge("followup", END)

    return graph.compile()


def build_chat_graph() -> StateGraph:
    """Build the chat follow-up graph.

    This is a simple single-node graph for handling follow-up questions.
    Why a separate graph instead of adding chat to the analysis graph?

    Because the chat loop is interactive — it requires user input between
    each iteration. LangGraph graphs run to completion once invoked,
    so the "loop" is actually managed by the API layer:

        while user_wants_to_chat:
            user_msg = get_user_input()
            state["chat_history"].append(user_msg)
            state = chat_graph.invoke(state)
            show_response(state)
    """
    graph = StateGraph(MedSimplifyState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    return graph.compile()
