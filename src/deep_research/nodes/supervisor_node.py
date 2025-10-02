from langchain_core.messages import SystemMessage
from langgraph.types import Command
from typing_extensions import Literal

from deep_research import logger
from deep_research.chains import supervisor_model
from deep_research.consts import max_concurrent_researchers, max_researcher_iterations
from deep_research.prompts import lead_researcher_prompt
from deep_research.states.state_multi_agent_supervisor import (
    ConductResearch,
    ResearchComplete,
    SupervisorState,
)
from deep_research.tools.think_tools import think_tool
from deep_research.utils import get_today_str

supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    logger.info("***[SUPERVISOR] NODE supervisor***")
    supervisor_messages = state.get("supervisor_messages", [])

    # Prepare system message with current date and constraints
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # Make decision about next research steps
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1},
    )
