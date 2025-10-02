import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.format_utils import format_messages
from deep_research.graphs.research_agent_scope import deep_researcher_builder

# Run the workflow
from deep_research.utils import get_current_dir

checkpointer = InMemorySaver()
scope = deep_researcher_builder.compile(checkpointer=checkpointer)
scope.get_graph().print_ascii()
current_dir = get_current_dir()
scope.get_graph(xray=True).draw_mermaid_png(output_file_path=f"{current_dir}/images/scoping_agent.png")

if __name__ == "__main__":
    from uuid import uuid4

    thread_id = uuid4()
    print(f"scope research {thread_id}")
    thread = {
        "configurable": {"thread_id": thread_id},
        "run_name": f'scoping_workflow_{time.strftime("%m-%d-%Hh%M", time.localtime())}',
    }
    result = scope.invoke(
        {"messages": [HumanMessage(content="I want to research the best coffee shops in San Francisco.")]}, config=thread
    )
    format_messages(result["messages"])
    result = scope.invoke(
        {"messages": [HumanMessage(content="Let's examine coffee quality to assess the best coffee shops in San Francisco.")]},
        config=thread,
    )
    format_messages(result["messages"])
    print(result["research_brief"])
