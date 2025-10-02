import os

from langchain_core.messages import HumanMessage
from md2pdf.core import md2pdf

from deep_research import APP_ROOT, PROJECT_ROOT, logger
from deep_research.chains import writer_model
from deep_research.prompts import final_report_generation_prompt
from deep_research.states.state_scope import AgentState
from deep_research.utils import get_today_str


async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    logger.info("***[writer] NODE final_report_generation***")
    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""), findings=findings, date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    final_report_content = final_report.content

    css_path = os.path.join(APP_ROOT, "config", "pdf_styles.css")
    file_path = f"{PROJECT_ROOT}/reports/report_{os.getenv('MODEL_PROVIDER','watsonx')}.pdf"
    md2pdf(file_path, md_content=final_report_content, css_file_path=css_path, base_url=None)
    logger.info(f"***[writer] NODE final_report_generation Report written to {file_path}***")
    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }
