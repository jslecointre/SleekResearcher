import asyncio
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage

from deep_research.format_utils import format_messages
from deep_research.graphs.multi_agent_supervisor import supervisor_agent
from deep_research.utils import get_current_dir

current_dir = get_current_dir()
supervisor_agent.get_graph().print_ascii()
supervisor_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=f"{current_dir}/images/supervisor.png")


async def main():
    thread_id = uuid4()
    print(f"supervisor_agent {thread_id}")
    thread = {
        "configurable": {"thread_id": thread_id},
        "run_name": f'supervisor_workflow_{time.strftime("%m-%d-%Hh%M", time.localtime())}',
    }

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

    result = await supervisor_agent.ainvoke({"supervisor_messages": [HumanMessage(content=f"{research_brief}.")]}, config=thread)
    format_messages(result["supervisor_messages"])


if __name__ == "__main__":
    asyncio.run(main())
