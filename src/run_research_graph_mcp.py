import asyncio
import time

from langchain_core.messages import HumanMessage

from deep_research.format_utils import format_messages, show_prompt
from deep_research.graphs.research_agent_mcp import agent_mcp
from deep_research.prompts import research_agent_prompt_with_mcp

show_prompt(research_agent_prompt_with_mcp, "Research Agent Instructions")


async def main():
    from uuid import uuid4

    thread_id = uuid4()
    print(f"\nresearch agent {thread_id}")
    thread = {
        "configurable": {"thread_id": thread_id},
        "run_name": f'research_workflow_mcp_{time.strftime("%m-%d-%Hh%M", time.localtime())}',
    }
    # Example brief
    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically
    on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area,
    using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,
    expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,
    location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary
    sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like
    Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct
    customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
    the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as
    of July 2025."""

    result = await agent_mcp.ainvoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]}, config=thread)
    format_messages(result["researcher_messages"])
    print(result["compressed_research"])


if __name__ == "__main__":
    asyncio.run(main())
