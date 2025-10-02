from langchain_core.messages import HumanMessage, get_buffer_string

from deep_research import logger
from deep_research.chains import scoping_model
from deep_research.prompts import transform_messages_into_research_topic_prompt
from deep_research.states.state_scope import AgentState, ResearchQuestion
from deep_research.utils import get_today_str


def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    logger.info("***[SCOPING] NODE write_research_brief***")
    # Set up structured output model
    structured_output_model = scoping_model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_research_topic_prompt.format(
                    messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
                )
            )
        ]
    )

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")],
    }
