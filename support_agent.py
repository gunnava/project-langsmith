"""
LangSmith Support Agent 

"""

from typing import Union
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langsmith import Client
from prompts import SYSTEM_PROMPT, push_system_prompt
from schemas import SupportState, TicketInput, _PRIORITY_TO_SEVERITY, MODEL_NAME, PROJECT_NAME
from nodes import (
    classify_node,
    draft_response_node,
    quality_check_node,
    chat_response_node,
    route_input,
    route_after_quality,
)

client = Client()
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# ── Build LangGraph ───────────────────────────────────────────────────────────
graph = StateGraph(SupportState)
graph.add_node("chat_response", chat_response_node)
graph.add_node("classify", classify_node)
graph.add_node("draft_response", draft_response_node)
graph.add_node("quality_check", quality_check_node)

graph.add_conditional_edges(START, route_input, {"chat": "chat_response", "ticket": "classify"})
graph.add_edge("chat_response", END)
graph.add_edge("classify", "draft_response")
graph.add_edge("draft_response", "quality_check")
graph.add_conditional_edges("quality_check", route_after_quality)

agent = graph.compile()



# ── Input normalisation ───────────────────────────────────────────────────────
def normalize_input(raw: Union[str, dict]) -> tuple[str, dict]:
    """Convert a plain prompt string or a TicketInput dict into (question, ticket_fields)."""
    if isinstance(raw, str):
        return raw.strip(), {}

    parts: list[str] = []
    if raw.get("subject"):
        parts.append(f"Subject: {raw['subject']}")
    if raw.get("description"):
        parts.append(f"Description: {raw['description']}")
    if raw.get("error_message"):
        parts.append(f"Error message: {raw['error_message']}")
    if raw.get("steps_to_reproduce"):
        parts.append(f"Steps already tried: {raw['steps_to_reproduce']}")

    question = "\n".join(parts) if parts else str(raw)
    ticket_fields = {k: v for k, v in raw.items() if v}
    return question, ticket_fields


# ── wrap_openai() ───────────────────────────────────────────────────
def summarise_question(question: str) -> str:
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai
    raw_client = wrap_openai(OpenAI())
    resp = raw_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Summarise this support question in 10 words or fewer."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


# ── create_feedback() ──────────────────────────────────────────────
def submit_feedback(run_id: str, satisfied: bool, comment: str = "") -> None:
    client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=1.0 if satisfied else 0.0,
        comment=comment,
    )
    print(f"[Feedback] {'positive' if satisfied else 'negative'} feedback submitted for run {run_id}")


# ── Run agent ─────────────────────────────────────────────────────────────────
def run_input(
    raw: Union[str, dict],
    question_id: str = "Q001",
    thread_id: str = "thread-1",
) -> dict:
    """Invoke the agent on a plain prompt string OR a TicketInput dict."""
    question, ticket_fields = normalize_input(raw)
    config = {
        "metadata": {
            "question_id": question_id,
            "thread_id": thread_id,
            "ls_provider": "openai",
            "ls_model_name": MODEL_NAME,
            "ls_project": PROJECT_NAME, 
            "input_type": "ticket" if ticket_fields else "prompt",
        },
        "run_name": f"support-{question_id}",
        "tags": ["langsmith-support", "demo", "ticket" if ticket_fields else "prompt"],
    }
    initial: SupportState = {
        "question": question,
        "ticket_fields": ticket_fields,
        "category": "",
        "severity": "",
        "doc_results": "",
        "draft_response": "",
        "quality_score": 0,
        "quality_feedback": "",
        "attempt": 0,
        "used_docs": False,
        "thread_id": thread_id,
    }
    return agent.invoke(initial, config=config)


def run_question(question: str, question_id: str = "Q001", thread_id: str = "thread-1") -> dict:
    return run_input(question, question_id=question_id, thread_id=thread_id)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from cli import main
    main()
