import asyncio
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.format_utils import format_messages
from deep_research.graphs.research_agent_full import deep_researcher_builder
from deep_research.utils import get_current_dir

current_dir = get_current_dir()
checkpointer = InMemorySaver()
full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
full_agent.get_graph().print_ascii()
full_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=f"{current_dir}/images/full_agent.png")


async def main(research_topic: str, follow_up: str):
    thread_id = uuid4()
    print(f"full_agent_agent {thread_id}")
    thread = {
        "configurable": {"thread_id": thread_id},
        "run_name": f'full_agent{time.strftime("%m-%d-%Hh%M", time.localtime())}',
    }

    result = await full_agent.ainvoke({"messages": [HumanMessage(content=research_topic)]}, config=thread)
    format_messages(result["messages"])
    result = await full_agent.ainvoke({"messages": [HumanMessage(content=follow_up)]}, config=thread)
    print(result["final_report"])


if __name__ == "__main__":
    research_topic = "Compare Gemini to OpenAI Deep Research agents."
    follow_up = "Yes the specific Deep Research products."
    asyncio.run(main(research_topic=research_topic, follow_up=follow_up))
