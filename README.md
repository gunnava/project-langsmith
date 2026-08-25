# LangSmith Support Agent

A production-grade AI support agent built to explore agentic workflows, RAG-based retrieval, and LLM observability using LangGraph, LangChain, and LangSmith. Covers every major module: tracing, evaluation, Prompt Hub, human feedback, and monitoring.

## What it does

- **Plain questions** → single-pass chat response with RAG doc retrieval
- **Structured support tickets** → classify (category + severity) → draft response → quality-check loop (up to 3 attempts) → show result

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd langsmith-support-agent
uv sync
```

**2. Configure environment variables**

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and LANGSMITH_API_KEY
```

**3. Run the agent**

```bash
uv run python support_agent.py
```

> First run builds the vector store by crawling LangSmith docs (~7 min, one-time). Subsequent runs load from `langsmith_docs.parquet` in under a second.

##
START
  │
  ├── [plain question] ──► chat_response ──► END
  │
  └── [ticket] ──► classify ──► draft_response ──► quality_check
                                     ▲                   │
                                     │                   ├── score ≥ 4 or 3 attempts ──► END
                                     └── score < 4 ──────┘

Both chat_response and draft_response call search_langsmith_docs (RAG tool) if needed.

## Usage

**Interactive CLI**

```
You: How do I enable tracing in Python?
You: ticket   # → prompts for structured ticket fields
You: quit
```

**Programmatic**

```python
from support_agent import run_input

# Plain question
result = run_input("How do I push a prompt to Prompt Hub?")
print(result["draft_response"])

# Structured ticket
result = run_input({
    "subject": "Traces missing from UI",
    "description": "Set LANGCHAIN_TRACING_V2=true but no traces appear",
    "priority": "high",
})
print(result["category"], result["severity"])
print(result["draft_response"])
```
## Project structure

```
langsmith-support-agent/
├── support_agent.py   # graph construction, run_input, submit_feedback
├── nodes.py           # LangGraph nodes + routing
├── schemas.py         # SupportState, TicketInput, Pydantic models
├── prompts.py         # Prompt Hub: load / push / fallback
├── utils_rag.py       # vectorstore build + search_langsmith_docs tool
├── cli.py             # interactive CLI loop
├── evaluate.py        # dataset, evaluators, pairwise comparison
└── test_agent.py      # smoke tests
```

## Evaluation

Run the full evaluation suite (creates a LangSmith dataset, two experiments, pairwise comparison):

```bash
uv run python evaluate.py
```

## Smoke tests

```bash
uv run python test_agent.py
```

## Key design decisions

- **Dual routing from START**: `add_conditional_edges(START, route_input, ...)` sends plain questions directly to `chat_response` (single pass) and tickets through `classify → draft → quality_check` (loop).
- **k=8 retrieval**: broader than the default k=4 — needed for open-ended questions that span multiple doc sections.
- **Reference page boosted**: `https://docs.langchain.com/langsmith/reference` is loaded via `WebBaseLoader` in addition to the sitemap, doubling its chunk count in the index.
- **Prompt Hub auto-push**: on first run, if the prompt doesn't exist in the Hub yet, it's pushed automatically and retried up to 2 times before falling back to the local default.
