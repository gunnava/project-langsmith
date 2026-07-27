"""
Smoke test — exercises both input modes without the interactive loop.
Run: python test_agent.py
"""

from support_agent import run_input, normalize_input

# ── 1. Verify normalize_input handles both types ───────────────────────────────
print("=== normalize_input tests ===\n")

q, fields = normalize_input("How do I enable tracing in Python?")
assert q == "How do I enable tracing in Python?", "plain prompt should pass through unchanged"
assert fields == {}, "plain prompt should produce empty ticket_fields"
print("  plain prompt passes through unchanged")

ticket = {
    "subject": "Traces not appearing in LangSmith",
    "description": "I set my API key but see zero traces in the UI",
    "error_message": "No traces visible at smith.langchain.com",
    "steps_to_reproduce": "Set LANGCHAIN_TRACING_V2=true, ran chain.invoke()",
    "priority": "high",
}
q, fields = normalize_input(ticket)
assert "Subject:" in q
assert "Error message:" in q
assert fields["priority"] == "high"
print("  ticket dict assembles question + extracts ticket_fields")
print(f"  question preview: {q[:80]}...")
print(f"  ticket_fields keys: {list(fields.keys())}\n")


# ── 2. Live agent call — plain prompt ─────────────────────────────────────────
print("=== Live test 1: plain prompt ===\n")
result = run_input("How do I enable tracing in Python?", question_id="TEST-01")

print(f"  category  : {result['category'] or '(chat mode — not classified)'}")
print(f"  severity  : {result['severity'] or '(chat mode — not classified)'}")
print(f"  quality   : {result['quality_score']}/5 (chat mode — no quality loop)")
print(f"  used docs : {result['used_docs']}")
print(f"  attempts  : {result['attempt']}")
print(f"\n  Response preview:\n  {result['draft_response'][:300]}...\n")

# chat path: no classify, no quality loop — only draft_response and attempt are set
assert result["draft_response"], "should have a draft response"
assert result["attempt"] == 1, "chat path runs exactly once"
# used_docs is not guaranteed for chat — LLM may answer from training knowledge
print("  plain prompt test passed\n")


# ── 3. Live agent call — structured ticket ────────────────────────────────────
print("=== Live test 2: structured ticket ===\n")
result = run_input(ticket, question_id="TEST-02")

print(f"  category  : {result['category']}")
print(f"  severity  : {result['severity']}")
print(f"  quality   : {result['quality_score']}/5")
print(f"  used docs : {result['used_docs']}")
print(f"  priority 'high' -> severity: {result['severity']}")
print(f"\n  Response preview:\n  {result['draft_response'][:300]}...\n")

assert result["severity"] in ("P1", "P2"), "high-priority ticket should be P1 or P2"
assert result["used_docs"], "should have searched the docs"
print("  ticket test passed\n")

print("=== All tests passed ===")
