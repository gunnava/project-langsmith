"""
LangGraph node functions and routing logic.
Accesses prompts.SYSTEM_PROMPT at call time so evaluate.py can hot-swap it for experiments.
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langsmith.run_helpers import trace
from langgraph.graph import END

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
import prompts
from schemas import MODEL_NAME

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

from schemas import (
    SupportState,
    QuestionClassification,
    QualityResult,
    _PRIORITY_TO_SEVERITY,
)
from utils_rag import search_langsmith_docs

tools = [search_langsmith_docs]
llm_with_tools = llm.bind_tools(tools)
classify_llm = llm.with_structured_output(QuestionClassification)
quality_llm  = llm.with_structured_output(QualityResult)


# ── Node 1: Classify (tickets only) ──────────────────────────────────────────
@traceable(run_type="chain", name="classify_question")
def classify_node(state: SupportState) -> dict:
    priority = state.get("ticket_fields", {}).get("priority", "")
    priority_hint = (
        f"\nNote: the ticket was filed with priority='{priority}'. "
        "Use this as a hint for severity but override it if the content suggests otherwise."
        if priority else ""
    )
    result = classify_llm.invoke([
        SystemMessage(
            "Classify LangSmith/LangChain developer support questions.\n"
            "Categories: tracing, evaluation, prompt_management, feedback, monitoring, sdk_setup.\n"
            "Severity: P1 (blocked), P2 (feature broken), P3 (how-to), P4 (general)."
            + priority_hint
        ),
        HumanMessage(f"Question:\n{state['question']}"),
    ])
    ticket_severity = _PRIORITY_TO_SEVERITY.get(priority, "")
    severity = ticket_severity if ticket_severity in ("P1", "P2") else result.severity
    return {"category": result.category, "severity": severity}


# ── Node 2: Draft response (tickets — with quality loop) ──────────────────────
def draft_response_node(state: SupportState) -> dict:
    prior_feedback = state.get("quality_feedback", "")
    tf = state.get("ticket_fields", {})
    extra_ctx = ""
    if tf.get("error_message") and tf["error_message"] not in state["question"]:
        extra_ctx += f"Exact error message: {tf['error_message']}\n"
    if tf.get("steps_to_reproduce") and tf["steps_to_reproduce"] not in state["question"]:
        extra_ctx += f"Steps already tried: {tf['steps_to_reproduce']}\n"

    user_msg = (
        f"Category: {state.get('category', 'sdk_setup')}\n\n"
        f"Developer question:\n{state['question']}\n\n"
        + (extra_ctx and f"Additional context:\n{extra_ctx}\n")
        + (f"Previous QA feedback to address:\n{prior_feedback}\n\n" if prior_feedback else "")
        + "Search the LangSmith docs first, then write a precise answer."
    )
    messages = [SystemMessage(prompts.SYSTEM_PROMPT), HumanMessage(user_msg)]

    #Trace as a context manager for a specifc block of code
    with trace("docs_retrieval", run_type="retriever",
               metadata={"category": state.get("category"), "attempt": state.get("attempt", 0) + 1}):
        response = llm_with_tools.invoke(messages)

    used_docs = False
    doc_results = ""
    # if the llm decided to call the tool:
    if response.tool_calls:
        used_docs = True
        messages.append(response)
        for tc in response.tool_calls:
            result = search_langsmith_docs.invoke({
                "query": tc["args"].get("query", state["question"]),
                "category": tc["args"].get("category", state.get("category", "sdk_setup")),
            })
            doc_results = result
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        final = llm.invoke(messages) # second LLM call with doc 
        draft = final.content
    else:
        draft = response.content

    return {
        "draft_response": draft,
        "doc_results": doc_results,
        "used_docs": used_docs,
        "attempt": state.get("attempt", 0) + 1,
        "quality_feedback": "",
    }


# ── Node 3: Quality check (tickets only) ─────────────────────────────────────
@traceable(run_type="chain", name="quality_check")
def quality_check_node(state: SupportState) -> dict:
    result = quality_llm.invoke([
        SystemMessage(
            "You are a QA reviewer for LangSmith technical support. "
            "Score responses 1–5 on: technical accuracy, completeness, correct API usage, and tone."
        ),
        HumanMessage(
            f"Developer question:\n{state['question']}\n\n"
            f"Response to review:\n{state['draft_response']}"
        ),
    ])
    return {"quality_score": result.score, "quality_feedback": result.feedback}


# ── Node 4: Chat response (plain questions — single pass) ─────────────────────
def chat_response_node(state: SupportState) -> dict:
    messages = [SystemMessage(prompts.CHAT_SYSTEM_PROMPT), HumanMessage(state["question"])]

    with trace("docs_retrieval", run_type="retriever", metadata={"attempt": 1}):
        response = llm_with_tools.invoke(messages)

    used_docs = False
    doc_results = ""

    if response.tool_calls:
        used_docs = True
        messages.append(response)
        for tc in response.tool_calls:
            result = search_langsmith_docs.invoke({
                "query": tc["args"].get("query", state["question"]),
                "category": tc["args"].get("category", "sdk_setup"),
            })
            doc_results = result
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        final = llm.invoke(messages)
        draft = final.content
    else:
        draft = response.content

    return {"draft_response": draft, "doc_results": doc_results, "used_docs": used_docs, "attempt": 1}


# ── Routing for ticket vs chat ───────────────────────────────────────────────────────────────────
def route_input(state: SupportState) -> str:
    return "ticket" if state.get("ticket_fields") else "chat"

def route_after_quality(state: SupportState) -> str:
    if state["quality_score"] >= 4 or state["attempt"] >= 3:
        return END
    return "draft_response"
