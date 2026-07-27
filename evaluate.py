"""
Evaluation suite for the LangSmith Support Agent.

"""

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from support_agent import run_input, client
from schemas import MODEL_NAME, PROJECT_NAME

from langchain_openai import ChatOpenAI
from langsmith import evaluate
from langsmith.evaluation import evaluate_comparative
from pydantic import BaseModel, Field

judge_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


# ──  Golden dataset ──────────────────────────────────────────────────
DATASET_NAME = f"{PROJECT_NAME}-qa-v1"

# expected_keywords: terms that MUST appear in a correct answer (deterministic check)
GOLDEN_EXAMPLES = [
    {
        "question": "How do I enable tracing in my Python app?",
        "question_id": "Q01",
        "expected_category": "tracing",
        "expected_severity": "P3",
        "expected_keywords": ["LANGSMITH_TRACING", "LANGSMITH_API_KEY"],
    },
    {
        "question": "My traces are not showing up in LangSmith even though I set LANGCHAIN_API_KEY.",
        "question_id": "Q02",
        "expected_category": "sdk_setup",
        "expected_severity": "P2",
        "expected_keywords": ["LANGSMITH_API_KEY", "LANGSMITH_TRACING", "deprecated"],
    },
    {
        "question": "How do I create an evaluation dataset and add examples to it?",
        "question_id": "Q03",
        "expected_category": "evaluation",
        "expected_severity": "P3",
        "expected_keywords": ["create_dataset", "create_examples"],
    },
    {
        "question": "How do I compare two experiments head-to-head in LangSmith?",
        "question_id": "Q04",
        "expected_category": "evaluation",
        "expected_severity": "P3",
        "expected_keywords": ["evaluate_comparative"],
    },
    {
        "question": "How do I push and pull prompts from the LangSmith Prompt Hub?",
        "question_id": "Q05",
        "expected_category": "prompt_management",
        "expected_severity": "P3",
        "expected_keywords": ["pull_prompt", "push_prompt"],
    },
    {
        "question": "How do I attach human feedback to a trace after the fact?",
        "question_id": "Q06",
        "expected_category": "feedback",
        "expected_severity": "P3",
        "expected_keywords": ["create_feedback", "run_id", "score"],
    },
    {
        "question": "Can I run evaluators automatically on live production traces?",
        "question_id": "Q07",
        "expected_category": "monitoring",
        "expected_severity": "P3",
        "expected_keywords": ["Automations", "online"],
    },
    {
        "question": "How do I filter runs in LangSmith by a specific tag or error status?",
        "question_id": "Q08",
        "expected_category": "monitoring",
        "expected_severity": "P3",
        "expected_keywords": ["list_runs", "filter"],
    },
    {
        "question": "How do I write a custom LLM-as-judge evaluator for my experiments?",
        "question_id": "Q09",
        "expected_category": "evaluation",
        "expected_severity": "P3",
        "expected_keywords": ["with_structured_output", "score"],
    },
    {
        "question": "LangGraph is installed but none of my nodes appear as spans in LangSmith.",
        "question_id": "Q10",
        "expected_category": "sdk_setup",
        "expected_severity": "P2",
        "expected_keywords": ["LANGSMITH_TRACING", "true"],
    },
    # ── Ticket-format inputs (structured dict) ─────────────────────────────────
    {
        "question": {
            "subject": "Traces completely missing from UI",
            "description": "I set up LangSmith last week and ran 50+ chains but see zero traces.",
            "error_message": "No traces appear at smith.langchain.com",
            "steps_to_reproduce": "Set LANGCHAIN_TRACING_V2=true, ran chain.invoke(), checked UI",
            "priority": "high",
        },
        "question_id": "Q11",
        "expected_category": "sdk_setup",
        "expected_severity": "P2",
        "expected_keywords": ["LANGSMITH_TRACING", "LANGSMITH_API_KEY"],
    },
    {
        "question": {
            "subject": "evaluate() crashes on summary_evaluators",
            "description": "My custom summary evaluator raises AttributeError when I run evaluate().",
            "error_message": "AttributeError: 'Run' object has no attribute 'outputs'",
            "steps_to_reproduce": "Passed summary_evaluators=[my_fn] to evaluate()",
            "priority": "medium",
        },
        "question_id": "Q12",
        "expected_category": "evaluation",
        "expected_severity": "P3",
        "expected_keywords": ["summary_evaluators", "outputs"],
    },
]


def create_dataset() -> None:
    """Create the golden dataset in LangSmith (idempotent)."""
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"[Dataset] Using existing: {DATASET_NAME}")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden Q&A dataset for LangSmith support agent — 12 developer questions",
        metadata={"project": PROJECT_NAME},
    )
    client.create_examples(
        inputs=[
            {"question": ex["question"], "question_id": ex["question_id"]}
            for ex in GOLDEN_EXAMPLES
        ],
        outputs=[
            {
                "expected_category": ex["expected_category"],
                "expected_severity": ex["expected_severity"],
                "expected_keywords": ex["expected_keywords"],
            }
            for ex in GOLDEN_EXAMPLES
        ],
        dataset_id=dataset.id,
    )
    print(f"[Dataset] Created '{DATASET_NAME}' with {len(GOLDEN_EXAMPLES)} examples")


# ── Target functions ──────────────────────────────────────────────────────────
def _safe_run(raw_input, question_id: str) -> dict:
    """Run the agent on either a plain prompt string or a ticket dict."""
    try:
        return run_input(raw_input, question_id=question_id, thread_id=f"eval-{question_id}")
    except Exception as e:
        print(f"[Warning] run_input failed for {question_id}: {e}")
        return {
            "category": "sdk_setup", "severity": "P3",
            "draft_response": "", "quality_score": 0,
            "used_docs": False, "attempt": 0,
        }


def target_default(inputs: dict) -> dict:
    """Experiment A — default system prompt. Accepts prompt strings and ticket dicts."""
    r = _safe_run(inputs["question"], inputs.get("question_id", "eval"))
    return {
        "category": r["category"],
        "severity": r["severity"],
        "draft_response": r["draft_response"],
        "quality_score": r["quality_score"],
        "used_docs": r["used_docs"],
        "attempt": r["attempt"],
    }


def target_concise(inputs: dict) -> dict:
    """Experiment B — concise prompt variant (3-sentence cap) for pairwise comparison."""
    import prompts

    original = prompts.SYSTEM_PROMPT
    prompts.SYSTEM_PROMPT = original + " Keep your answer to a maximum of 3 sentences."
    r = _safe_run(inputs["question"], inputs.get("question_id", "eval-b"))
    prompts.SYSTEM_PROMPT = original

    return {
        "category": r["category"],
        "severity": r["severity"],
        "draft_response": r["draft_response"],
        "quality_score": r["quality_score"],
        "used_docs": r["used_docs"],
        "attempt": r["attempt"],
    }


# ── Deterministic evaluators ──────────────────────────────────────────────────
def category_accurate(inputs, reference_outputs, outputs) -> dict:
    if not outputs.get("category"):
        return {"key": "category_accurate", "score": 0}
    return {
        "key": "category_accurate",
        "score": int(outputs["category"] == reference_outputs["expected_category"]),
    }


def severity_accurate(inputs, reference_outputs, outputs) -> dict:
    if not outputs.get("severity"):
        return {"key": "severity_accurate", "score": 0}
    return {
        "key": "severity_accurate",
        "score": int(outputs["severity"] == reference_outputs["expected_severity"]),
    }


def used_docs(inputs, reference_outputs, outputs) -> dict:
    return {"key": "used_docs", "score": int(bool(outputs.get("used_docs", False)))}


def response_has_content(inputs, reference_outputs, outputs) -> dict:
    return {"key": "response_has_content", "score": int(len(outputs.get("draft_response", "")) > 50)}



# ── LLM-as-judge evaluators ───────────────────────────────────────────────────
class AnswersQuestion(BaseModel):
    answers: bool = Field(description="True if the response directly addresses the developer's question")
    score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="One-sentence explanation")


class ToneCheck(BaseModel):
    professional: bool = Field(description="True if the tone is professional and clear")
    score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="One-sentence explanation")


answers_llm = judge_llm.with_structured_output(AnswersQuestion)
tone_llm = judge_llm.with_structured_output(ToneCheck)


def answers_question(inputs, reference_outputs, outputs) -> dict:
    """LLM-as-judge: does the response actually answer the developer's question?"""
    draft = outputs.get("draft_response", "")
    if not draft:
        return {"key": "answers_question", "score": 0}
    result = answers_llm.invoke(
        f"Developer question: {inputs['question']}\n\n"
        f"Support response:\n{draft}\n\n"
        "Does the response directly and completely answer the question?"
    )
    return {"key": "answers_question", "score": result.score}


def tone_appropriate(inputs, reference_outputs, outputs) -> dict:
    """LLM-as-judge: is the response professional and suitable for developer support?"""
    draft = outputs.get("draft_response", "")
    if not draft:
        return {"key": "tone_appropriate", "score": 0}
    result = tone_llm.invoke(
        f"Support response:\n{draft}\n\n"
        "Is this response professional, technically precise, and appropriate for a developer audience?"
    )
    return {"key": "tone_appropriate", "score": result.score}


# ── Summary evaluators ─────────────────────────────────────────────
def category_accuracy_summary(runs, examples) -> dict:
    """Dataset-level: % of questions correctly classified by category."""
    correct = sum(
        1 for run, ex in zip(runs, examples)
        if run.outputs and ex.outputs
        and run.outputs.get("category") == ex.outputs.get("expected_category")
    )
    total = sum(1 for r in runs if r.outputs)
    return {"key": "category_accuracy_pct", "score": correct / total if total else 0}


def severity_accuracy_summary(runs, examples) -> dict:
    """Dataset-level: % of questions with correct severity."""
    correct = sum(
        1 for run, ex in zip(runs, examples)
        if run.outputs and ex.outputs
        and run.outputs.get("severity") == ex.outputs.get("expected_severity")
    )
    total = sum(1 for r in runs if r.outputs)
    return {"key": "severity_accuracy_pct", "score": correct / total if total else 0}


def docs_usage_rate_summary(runs, examples) -> dict:
    """Dataset-level: fraction of questions where the agent searched the docs."""
    valid = [r for r in runs if r.outputs]
    used = sum(1 for r in valid if r.outputs.get("used_docs", False))
    return {"key": "docs_usage_rate", "score": used / len(valid) if valid else 0}


def avg_quality_score_summary(runs, examples) -> dict:
    """Dataset-level: average agent quality score, normalised to 0–1."""
    scores = [r.outputs["quality_score"] for r in runs if r.outputs and "quality_score" in r.outputs]
    return {"key": "avg_quality_normalized", "score": (sum(scores) / len(scores) / 5.0) if scores else 0}


# ── Pairwise evaluator ──────────────────────────────────────────────
class PairwiseVerdict(BaseModel):
    preferred: Literal["A", "B", "tie"] = Field(
        description="Which response is more helpful and accurate for the developer"
    )
    reasoning: str = Field(description="One-sentence explanation")


pairwise_llm = judge_llm.with_structured_output(PairwiseVerdict)


def ranked_preference(runs: list, example) -> dict:
    response_a = (runs[0].outputs or {}).get("draft_response", "(no response)") if len(runs) > 0 else "(no response)"
    response_b = (runs[1].outputs or {}).get("draft_response", "(no response)") if len(runs) > 1 else "(no response)"

    result = pairwise_llm.invoke(
        f"Developer question: {example.inputs['question']}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response better helps the developer? Reply A, B, or tie."
    )

    if result.preferred == "A":
        scores = {str(runs[0].id): 1, str(runs[1].id): 0}
    elif result.preferred == "B":
        scores = {str(runs[0].id): 0, str(runs[1].id): 1}
    else:  # tie
        scores = {str(runs[0].id): 1, str(runs[1].id): 1}

    return {"key": "preferred_response", "scores": scores, "comment": result.reasoning}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== LangSmith Support Agent Evaluation ===\n")

    # Step 1: Create golden dataset
    print("\n[Step 1] Creating Dataset if it doesn't exist...")
    create_dataset()

    # Step 2: Experiment A — default prompt
    print("\n[Step 2] Running experiment A (default prompt)...")
    results_a = evaluate(
        target_default,
        data=DATASET_NAME,
        evaluators=[
            category_accurate,
            severity_accurate,
            used_docs,
            response_has_content,
            answers_question,
            tone_appropriate,
        ],
        summary_evaluators=[
            category_accuracy_summary,
            severity_accuracy_summary,
            docs_usage_rate_summary,
            avg_quality_score_summary,
        ],
        experiment_prefix=f"{PROJECT_NAME}-default",
        metadata={"model": MODEL_NAME, "prompt_variant": "default"},
        max_concurrency=2,
    )
    exp_a = results_a.experiment_name
    print(f"[Exp A] Done: {exp_a}")

    # Step 3: Experiment B — concise prompt variant
    print("\n[Step 3] Running experiment B (concise prompt)...")
    results_b = evaluate(
        target_concise,
        data=DATASET_NAME,
        evaluators=[
            category_accurate,
            severity_accurate,
            used_docs,
            response_has_content,
        ],
        experiment_prefix=f"{PROJECT_NAME}-concise",
        metadata={"model": MODEL_NAME, "prompt_variant": "concise"},
        max_concurrency=2,
    )
    exp_b = results_b.experiment_name
    print(f"[Exp B] Done: {exp_b}")

    # Step 4: Pairwise comparison
    print(f"\n[Step 4] Pairwise: {exp_a}  vs  {exp_b}")
    evaluate_comparative(
        [exp_a, exp_b],
        evaluators=[ranked_preference],
    )
    print("[Pairwise] Done — check LangSmith UI for head-to-head results")

