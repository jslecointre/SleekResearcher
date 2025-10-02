"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""
from langgraph.graph import END, START, StateGraph

from deep_research.nodes.research_compress_research import compress_research
from deep_research.nodes.research_llm_call_node import llm_call
from deep_research.nodes.research_should_continue_cond_edge import should_continue
from deep_research.nodes.research_tool_node import tool_node
from deep_research.states.state_research import ResearcherOutputState, ResearcherState

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # Continue research loop
        "compress_research": "compress_research",  # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call")  # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
researcher_agent.get_graph().print_ascii()
