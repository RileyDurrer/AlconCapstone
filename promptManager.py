import dotenv
import pandas as pd
import json
import os

from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def buildPrompt(marketing_text: str, product: str) -> str:
    """
    Builds a grading prompt that:
    - Rewards similarity to Approved Claims
    - Enforces compliance with FDA + FTC policies
    - Returns IDs so caller can map response -> original DataFrame rows
    """
    approved, fda, ftc = getCompliancePolicies(product)

    # Clean + index for traceability
    approved = approved.reset_index(drop=True)
    fda      = fda.reset_index(drop=True)
    ftc      = ftc.reset_index(drop=True)

    approved["policy_id"] = approved.index
    fda["policy_id"]      = fda.index
    ftc["policy_id"]      = ftc.index

    # Assume policy text is in first column
    approved_text_col = approved.columns[0]
    fda_text_col      = fda.columns[0]
    ftc_text_col      = ftc.columns[0]

    def format_records(df, df_type, text_col):
        return [
            {
                "policy_id": f"{df_type}:{row.policy_id}",
                "text": str(row[text_col])
            }
            for _, row in df.iterrows()
            if pd.notna(row[text_col])
        ]

    approved_records = format_records(approved, "approved", approved_text_col)
    fda_records      = format_records(fda, "fda",      fda_text_col)
    ftc_records      = format_records(ftc, "ftc",      ftc_text_col)

    # All formatted for prompt
    def pretty_list(records, title):
        if not records:
            return f"{title}\n  (none listed)\n"

        lines = "\n".join(
            f"- ({r['policy_id']}) {r['text']}"
            for r in records
        )
        return f"{title}\n{lines}\n"

    approved_section = pretty_list(approved_records, "Approved Claims (positive alignment)")
    fda_section      = pretty_list(fda_records,      "FDA Policies")
    ftc_section      = pretty_list(ftc_records,      "FTC Policies")

    # Final prompt
    prompt = f"""
You are a compliance grader evaluating medical marketing content in the U.S.

Your goal is to:
1) Reward similarity with approved claims.
2) Penalize conflicts with FDA + FTC requirements.
3) Return structured JSON referencing policy IDs.

---

## Marketing Text
{marketing_text}

---

## Policy Groups

{approved_section}

{fda_section}

{ftc_section}

---

## Scoring Rules

### Approved Claims
If text is similar in meaning to an approved claim → high score.
- Very similar wording → 85–100
- Same meaning (rephrased) → 70–85
- Partial alignment → 50–70
- Unrelated → 0–50

### FDA / FTC Policies
Must be followed.
Conflicts → lower score.

---

## Output Rules
- You must evaluate EVERY policy in ALL policy groups (Approved, FDA, FTC). No exceptions.
- Produce exactly one evaluation object per policy.
- If the policy is unrelated, still output it with a high grade (e.g., 80–100) and reason "Unrelated; no conflict."
- Include: policy_id, policy, type, grade, reason.
- Higher scores = stronger compliance.
- Return ONLY JSON in the required format.
- Do NOT include any text outside the JSON structure.

Return ONLY JSON, no commentary.

### Output JSON format
{{
  "evaluations": [
    {{
      "policy_id": "<approved:3 | fda:7 | ftc:5>",
      "policy": "<policy text>",
      "type": "<approved | fda | ftc>",
      "grade": <0-100>,
      "reason": "<brief justification>"
    }}
  ],
  "overall_summary": "<short summary>"
}}
""".strip()

    return prompt


def getCompliancePolicies(product):
    """
    Loads all CSV files under ./compliance_docs/{product}_compliance/
    Returns: {file_stem: DataFrame}
    """
    base_dir = os.getcwd()                 # folder of this .py
    folder = os.path.join(base_dir, "compliance_docs", f"{product}_compliance")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Product folder not found: {folder}")
    
    approved_claims = pd.read_csv(os.path.join(folder, "ApprovedClaims.csv"))
    fda      = pd.read_csv(os.path.join(folder, "FDAPolicies.csv"))
    ftc      = pd.read_csv(os.path.join(folder, "FTCPolicies.csv"))

    return approved_claims, fda, ftc


def getResponseFromLLM(prompt: str) -> dict:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    text = response.output_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


