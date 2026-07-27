"""
Prompt Hub integration.
Both prompts (ticket and chat) are loaded from LangSmith Prompt Hub on startup,
auto-pushed on first run, with up to 2 retries before falling back to local defaults.
"""

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
client = Client()

# ── Prompt Hub names ──────────────────────────────────────────────────────────
from schemas import MODEL_NAME, PROJECT_NAME

TICKET_PROMPT_HUB_NAME = f"{PROJECT_NAME}-prompt"
CHAT_PROMPT_HUB_NAME   = f"{PROJECT_NAME}-chat-prompt"

# ── Local fallbacks ───────────────────────────────────────────────────────────
DEFAULT_TICKET_PROMPT = (
    "You are a senior technical support engineer for LangSmith and LangChain. "
    "Answer developer questions accurately using the documentation provided. "
    "Always include exact class names, function signatures, and environment variable names. "
    "Be concise but complete. End with a concrete next step the developer can take. "
    "Always provide links to documentation where applicable."
)

DEFAULT_CHAT_PROMPT = (
    "You are a senior technical support engineer for LangSmith and LangChain. "
    "Answer developer questions accurately using the documentation provided. "
    "Answer the developer's question directly and concisely — 2 to 4 sentences max. "
    "Always include a link to the relevant documentation page. "
)


# ── Shared load/push logic ────────────────────────────────────────────────────
def _push_prompt(hub_name: str, text: str) -> None:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", text)])
    client.push_prompt(hub_name, object=prompt)
    print(f"[Prompt Hub] Pushed '{hub_name}'")


def _load_prompt(hub_name: str, default: str, retries: int = 0) -> str:
    try:
        hub_prompt = client.pull_prompt(hub_name)
        template = hub_prompt.messages[0].prompt.template
        # print(f"[Prompt Hub] Loaded '{hub_name}'")
        return template
    except Exception:
        if retries >= 2:
            print(f"[Prompt Hub] Failed after 2 retries for '{hub_name}', using local fallback")
            return default
        print(f"[Prompt Hub] Not found — pushing default for '{hub_name}' (attempt {retries + 1})")
        _push_prompt(hub_name, default)
        return _load_prompt(hub_name, default, retries + 1)


# ── Public helpers (for manual use / evaluate.py experiments) ─────────────────
def push_system_prompt() -> None:
    _push_prompt(TICKET_PROMPT_HUB_NAME, DEFAULT_TICKET_PROMPT)

def push_chat_system_prompt() -> None:
    _push_prompt(CHAT_PROMPT_HUB_NAME, DEFAULT_CHAT_PROMPT)


# ── Load both prompts at startup ──────────────────────────────────────────────
SYSTEM_PROMPT      = _load_prompt(TICKET_PROMPT_HUB_NAME, DEFAULT_TICKET_PROMPT)
CHAT_SYSTEM_PROMPT = _load_prompt(CHAT_PROMPT_HUB_NAME,   DEFAULT_CHAT_PROMPT)
