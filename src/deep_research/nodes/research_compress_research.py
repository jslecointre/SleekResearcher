from langchain_core.messages import HumanMessage, SystemMessage, filter_messages

from deep_research import logger
from deep_research.chains import compress_model
from deep_research.prompts import (
    compress_research_human_message,
    compress_research_system_prompt,
)
from deep_research.states.state_research import ResearcherState
from deep_research.tools.search_tools import tavily_search
from deep_research.tools.think_tools import think_tool
from deep_research.utils import get_today_str

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}


def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    logger.info("***[RESEARCH] NODE compress_research***")
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        + [HumanMessage(content=compress_research_human_message)]
    )
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [str(m.content) for m in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])]

    return {"compressed_research": str(response.content), "raw_notes": ["\n".join(raw_notes)]}
