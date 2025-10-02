import time

from langchain_core.messages import HumanMessage

from deep_research.format_utils import format_messages, show_prompt
from deep_research.graphs.research_agent import researcher_agent
from deep_research.prompts import research_agent_prompt
from deep_research.utils import get_current_dir

show_prompt(research_agent_prompt, "Research Agent Instructions")

current_dir = get_current_dir()
researcher_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=f"{current_dir}/images/researcher_agent.png")

if __name__ == "__main__":
    from uuid import uuid4

    thread_id = uuid4()
    print(f"\nresearch agent {thread_id}")
    thread = {
        "configurable": {"thread_id": thread_id},
        "run_name": f'research_workflow_{time.strftime("%m-%d-%Hh%M", time.localtime())}',
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

    result = researcher_agent.invoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]}, config=thread)
    format_messages(result["researcher_messages"])
    print(result["compressed_research"])
