from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import Literal

from deep_research import logger
from deep_research.chains import scoping_model
from deep_research.prompts import clarify_with_user_instructions
from deep_research.states.state_scope import AgentState, ClarifyWithUser
from deep_research.utils import get_today_str


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    logger.info("***[SCOPING] NODE clarify_with_user***")
    structured_output_model = scoping_model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=clarify_with_user_instructions.format(
                    messages=get_buffer_string(messages=state["messages"]), date=get_today_str()
                )
            )
        ]
    )

    # Route based on clarification need
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})
