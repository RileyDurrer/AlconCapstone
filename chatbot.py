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
    """
    Generate targeted fix suggestions based on the current compliance results.

    This function examines:
      - Policy violations (filtered_evaluations)
      - Approved claim alignment (approved_matches)

    It returns a plain-text bullet list that the chatbot will embed into its response.

    Args:
        compliance (dict):
            The processed compliance results returned by structure_compliance_grade().
            May be None if no compliance check has been run.

    Returns:
        str:
            A formatted string containing recommended improvements or a default message
            if no compliance results exist yet.
    """
    
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
    """
    Compare the previous compliance results to the current results and summarize changes.

    This identifies:
      - Score increases or decreases in FDA / FTC / Approved categories
      - Newly detected policy failures
      - Issues that were fixed since the last version

    Args:
        previous (dict):
            The previous compliance result stored by the controller,
            or None for the first run.
        current (dict):
            The latest compliance result.

    Returns:
        str:
            A human-readable summary describing how compliance has changed.
    """
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
    """
    Generate a chatbot response using short-term conversation memory
    and long-term compliance state.

    The model receives:
      - A system prompt defining its behavior
      - Long-term memory (product, strictness, last + previous compliance results)
      - Recent conversation messages (short_history)
      - Auto-generated fix suggestions
      - Auto-generated comparison between the last two compliance evaluations
      - The new user prompt

    Args:
        user_prompt (str):
            The message the user just submitted.
        client (OpenAI):
            The OpenAI client instance created by the controller.
        short_history (list):
            Recent chat messages in the form:
            [{ "role": "user"|"assistant", "content": str }, ...]
        long_term_state (dict):
            Persistent memory containing:
                - "product"
                - "strictness"
                - "last_compliance"
                - "previous_compliance"
                - "compliance_history"

    Returns:
        str:
            The assistant's text response generated by the LLM.
    """
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
