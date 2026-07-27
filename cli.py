"""
Interactive CLI for the LangSmith Support Agent.
Imports run_input lazily inside main() to avoid circular imports.
"""
import uuid

def _print_result(result: dict) -> None:
    is_ticket = bool(result.get("ticket_fields"))
    print(f"\n{'─'*60}")
    if is_ticket:
        sev_icon = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢"}.get(result.get("severity", ""), "⚪")
        print(f"  Category : {result.get('category', '—')}")
        print(f"  Severity : {sev_icon} {result.get('severity', '—')}")
        print(f"  Quality  : {result.get('quality_score', 0)}/5  Attempts: {result.get('attempt', 0)}")
    print(f"  Used docs: {result.get('used_docs', False)}")
    print(f"{'─'*60}\n")
    print(result.get("draft_response", ""))
    print()


def _collect_ticket() -> dict:
    print("\n  Fill in ticket fields (press Enter to skip optional fields)")
    subject = input("  Subject   : ").strip()
    description   = input("  Description (required)  : ").strip()
    if not description:
        print("  [cancelled]")
        return {}
    error_message = input("  Error message       : ").strip()
    steps         = input("  Steps already tried : ").strip()
    priority_raw  = input("  Priority [low/medium/high/critical] (default: medium): ").strip().lower()
    priority      = priority_raw if priority_raw in ("low", "medium", "high", "critical") else "medium"

    ticket: dict = {"description": description, "priority": priority}
    if subject:   ticket["subject"]        = subject
    if error_message: ticket["error_message"]      = error_message
    if steps:         ticket["steps_to_reproduce"] = steps
    return ticket


def main() -> None:
    from support_agent import run_input  # lazy to avoid circular import

    print("\n=== LangSmith Support Agent ===")
    print("Type your question, 'ticket' for structured input, or 'quit' to exit.\n")

    question_count = 0

    while True:
        raw = input("You: ").strip()
        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        question_count += 1
        qid = f"Q{question_count:03d}"

        if raw.lower() == "ticket":
            user_input = _collect_ticket()
            if not user_input:
                continue
            thread_id = f"ticket-{qid}-{uuid.uuid4().hex[:8]}"
        else:
            user_input = raw
            thread_id = f"chat-{qid}-{uuid.uuid4().hex[:8]}"

        print(f"\n[{qid}] Running agent...")
        result = run_input(user_input, question_id=qid, thread_id=thread_id)
        _print_result(result)
