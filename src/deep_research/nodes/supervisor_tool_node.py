import asyncio

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import Literal

from deep_research import logger
from deep_research.consts import max_researcher_iterations
from deep_research.graphs.research_agent import researcher_agent
from deep_research.states.state_multi_agent_supervisor import SupervisorState
from deep_research.tools.think_tools import think_tool


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    logger.info("***[SUPERVISOR] NODE supervisor_tools***")
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # Default next step
    should_end = False

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END

    else:
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "think_tool"]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]))

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                # Launch parallel research agents
                coros = [
                    researcher_agent.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                            "research_topic": tool_call["args"]["research_topic"],
                        }
                    )
                    for tool_call in conduct_research_calls
                ]

                # Wait for all research to complete
                tool_results = await asyncio.gather(*coros)

                # Format research results as tool messages
                # Each sub-agent returns compressed research findings in result["compressed_research"]
                # We write this compressed research as the content of a ToolMessage, which allows
                # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # Aggregate raw notes from all research
                all_raw_notes = ["\n".join(result.get("raw_notes", [])) for result in tool_results]

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        return Command(
            goto=next_step,
            update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")},
        )
    else:
        return Command(goto=next_step, update={"supervisor_messages": tool_messages, "raw_notes": all_raw_notes})
