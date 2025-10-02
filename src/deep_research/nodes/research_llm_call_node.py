from langchain_core.messages import SystemMessage

from deep_research import logger
from deep_research.chains import research_model
from deep_research.prompts import research_agent_prompt, research_agent_prompt_with_mcp
from deep_research.states.state_research import ResearcherState
from deep_research.tools import get_mcp_client
from deep_research.tools.search_tools import tavily_search
from deep_research.tools.think_tools import think_tool
from deep_research.utils import get_today_str

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}


def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    logger.info("***[RESEARCH] NODE llm_call***")
    model_with_tools = research_model.bind_tools(tools)
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))] + state["researcher_messages"]
            )
        ]
    }


async def llm_call_mcp(state: ResearcherState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    logger.info("***[RESEARCH] NODE llm_call_mcp***")
    # Get available tools from MCP server
    client = get_mcp_client()
    mcp_tools = await client.get_tools()

    # Use MCP tools for local document access
    all_tools = mcp_tools + [think_tool]

    # Initialize model with tool binding
    model_with_tools = research_model.bind_tools(all_tools)

    # Process user input with system prompt
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_with_mcp.format(date=get_today_str()))]
                + state["researcher_messages"]
            )
        ]
    }
