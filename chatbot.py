# chatbot.py
import json
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a friendly assistant helping Alcon marketing employees understand and "
    "improve their marketing material through compliance insights. "
    "Keep responses brief, constructive, and actionable."
)

# ---------------------------------------------------------
# Build fix suggestions
# ---------------------------------------------------------
def build_fix_suggestions(compliance: dict) -> str:
    """Generates fix suggestions based on compliance results."""
    if not compliance:
        return "Fix suggestions will appear after a compliance check."

    fixes = []

    failed = compliance.get("filtered_evaluations", [])
    for p in failed:
        fixes.append(f"- **{p['policy_id']}** — {p['reason']}")

    if not compliance.get("approved_matches"):
        fixes.append("- No strong alignment with approved claims.")
    else:
        fixes.append("- Strengthen similarity to matched approved claims.")

    if not fixes:
        return "No compliance issues found."

    return "\n".join(fixes)


# ---------------------------------------------------------
# Compare previous compliance to current compliance
# ---------------------------------------------------------
def compare_compliance(previous: dict, current: dict) -> str:
    """Compares previous compliance results to current results."""
    if not previous:
        return "This is the first compliance check of the session."

    prev_scores = previous.get("scores", {})
    curr_scores = current.get("scores", {})

    changes = []

    # Score changes
    for cat in ["approved", "fda", "ftc"]:
        if cat in prev_scores and cat in curr_scores:
            diff = curr_scores[cat] - prev_scores[cat]
            if diff > 5:
                changes.append(f"- **{cat.upper()} score improved** by +{diff:.1f}.")
            elif diff < -5:
                changes.append(f"- **{cat.upper()} score decreased** by {diff:.1f}.")

    # Detect newly failed or resolved items
    prev_failed = {p["policy_id"] for p in previous.get("filtered_evaluations", [])}
    curr_failed = {p["policy_id"] for p in current.get("filtered_evaluations", [])}

    added = curr_failed - prev_failed
    removed = prev_failed - curr_failed

    if added:
        changes.append("- New compliance issues detected: " + ", ".join(sorted(added)))
    if removed:
        changes.append("- Issues resolved: " + ", ".join(sorted(removed)))

    return "\n".join(changes) if changes else "Compliance is unchanged."


# ---------------------------------------------------------
# Main Chatbot Response
# ---------------------------------------------------------
def get_chatbot_response(user_prompt: str, client: OpenAI, short_history: list, long_term_state: dict):
    """Generates a chatbot response using both short-term and long-term memory."""
    current = long_term_state.get("last_compliance")
    previous = long_term_state.get("previous_compliance")

    fix_text = build_fix_suggestions(current)
    comparison_text = compare_compliance(previous, current)

    # Build prompt
    model_input = [
        {"role": "system", "content": SYSTEM_PROMPT},

        {"role": "system", "content": "Conversation State:\n" + json.dumps(long_term_state, indent=2)},
        
        *short_history,

        {"role": "assistant", "content": f"Fix Suggestions:\n{fix_text}"},
        {"role": "assistant", "content": f"Comparison to Previous Version:\n{comparison_text}"},

        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=model_input,
    )

    return response.output_text
