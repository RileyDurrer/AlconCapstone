import pandas as pd
import json
import os

from openai import OpenAI

def build_compliance_prompt(marketing_text: str, policies: dict, strictness: int):
    """
    Construct the full grading prompt used by the compliance LLM.

    This prompt:
      • Injects all Approved Claims, FDA policies, and FTC policies for the product.
      • Assigns unique policy IDs (e.g., "approved:0", "fda:12").
      • Describes how similarity, compliance, and penalties should be applied.
      • Embeds a strictness parameter that controls leniency (1) → strictness (10).
      • Instructs the model to return structured JSON only.

    Args:
        marketing_text (str):
            The marketing claim or text snippet that should be evaluated.

        policies (dict):
            A dictionary of DataFrames produced by `read_product_policy_files()`, in the form:
            {
                "approved": pd.DataFrame,
                "fda": pd.DataFrame,
                "ftc": pd.DataFrame
            }
            Each DataFrame must contain the policy text in its first column.

        strictness (int):
            A value from 1–10 controlling how harsh the scoring should be:
                1–3:   Lenient – only major issues reduce scores.
                4–7:   Standard – follow normal scoring rules.
                8–10:  Strict – penalties applied aggressively, weak similarity scored lower.

    Returns:
        str:
            A fully formatted prompt that the model can evaluate directly.
            The prompt includes:
                • Marketing text
                • All policy groups with IDs
                • Strictness interpretation
                • Scoring rules for all categories
                • Explicit JSON output schema
                • Instructions to return *only* JSON

    Notes:
        • FDA/FTC policies that score > 85 will be ignored by `structure_compliance_grade()`
          since they represent “no action needed.”
        • Approved claims are never filtered out and always included in the similarity scoring.
    """
    
    
    approved = policies["approved"].reset_index(drop=True)
    fda = policies["fda"].reset_index(drop=True)
    ftc = policies["ftc"].reset_index(drop=True)
    

    # Add policy_id columns
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

## Strictness Level
Strictness = {strictness}   # 1–10

Interpretation:
- Strictness 1–3 (lenient): Only clear conflicts should significantly reduce scores. Unsubstantiated or vague claims should receive moderate-to-high grades.
- Strictness 4–7 (standard): Use scoring rules as written.
- Strictness 8–10 (strict): Be conservative. Require strong evidence for high scores. Penalize weak similarity or even minor risks more heavily.

Higher strictness → lower grades for borderline similarity or mild compliance concerns.
Lower strictness → higher grades unless clear conflict exists.

## Scoring Rules

### Approved Claims
Evaluate how similar the marketing text is to each approved claim.
- Very similar wording → 85–100
- Same meaning (rephrased) → 70–85
- Partial alignment → 50–70
- Unrelated or no alignment → 0–50


### FDA / FTC Policies
These reflect regulatory requirements.
if it scores above a 85 it will be filtered out by the structureComplianceGrade function so it should only get that score if the user doesn't need to worry about it.
everything below a 70 will impose a penalty on the overall score.

- Fully compliant or unrelated → 80–100
- Minor risk or unclear compliance → 60–80
- Clear conflict, violation, or misleading claim → 0–60


---

## Output Rules
- You must evaluate EVERY policy in ALL policy groups (Approved, FDA, FTC). No exceptions.
- Produce exactly one evaluation object per policy.
- For FDA and FTC policies: if the policy is unrelated, output a high grade (>85) so that it can be ignored by later filters and the reason "Unrelated; no conflict."
- For Approved policies: if unrelated, output a low grade (0–50) and the reason "Unrelated; no similarity."
- Include: policy_id, policy, type, grade, reason.
- Higher scores = stronger compliance.
- Return ONLY JSON in the required format.
- Do NOT include any text outside the JSON structure.

Return ONLY JSON, no commentary.

### Output JSON format
{{
    "evaluations": [
    {{
        "policy_id": "<approved_claim:3 | fda:7 | ftc:5>",
        "policy": "<policy text>",
        "type": "<approved claim | fda | ftc>",
        "grade": <0-100>,
        "reason": "<brief justification>"
    }}
    ],
    "overall_summary": "<short summary>"
}}
""".strip()

    return prompt



def read_product_policy_files(product: str):
    """
    Loads all compliance policy CSV files for a given product.

    This reads the Approved Claims, FDA Policies, and FTC Policies
    located in:
        ./compliance_docs/{product}_compliance/

    Args:
        product (str):
            Product name whose corresponding policy folder exists
            under ./compliance_docs.

    Returns:
        dict:
            {
                "approved": pd.DataFrame,
                "fda": pd.DataFrame,
                "ftc": pd.DataFrame
            }

    Raises:
        FileNotFoundError:
            If the product folder or required CSV files are missing.
    """
    base = os.getcwd()
    folder = os.path.join(base, "compliance_docs", f"{product}_compliance")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Policies for '{product}' not found in: {folder}")

    approved = pd.read_csv(os.path.join(folder, "ApprovedClaims.csv"))
    fda = pd.read_csv(os.path.join(folder, "FDAPolicies.csv"))
    ftc = pd.read_csv(os.path.join(folder, "FTCPolicies.csv"))

    return {
        "approved": approved,
        "fda": fda,
        "ftc": ftc
    }


def get_response_from_LLM(prompt: str, client) -> dict:
    """Gets JSON API Response from OPENAI

    Args:
        prompt (str): prompt created by buildCompliancePrompt 

    Returns:
        dict: JSON Response that has been stripped
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    text = response.output_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)

#Removes passing policies from JSON and adds compiled scores
def structure_compliance_grade(response: dict) -> dict:
    """
    Convert raw LLM compliance evaluations into a front-end-ready structure.

    This function:
    - Removes high-scoring FDA/FTC policies (those the user does not need to worry about).
    - Sorts remaining policies by severity (lowest grade = highest priority).
    - Extracts approved-claim matches and computes a rolled-up “approved score”.
    - Computes FDA and FTC category scores using a weighted penalty system.
    - Ensures all outputs are JSON-serializable (no numpy types).

    Args:
        response (dict):
            Raw JSON dictionary returned from the LLM.
            Must contain:
            {
                "evaluations": [ { policy_id, policy, type, grade, reason }, ... ],
                "overall_summary": str
            }

    Returns:
        dict:
            A normalized structure ready for UI consumption:
            {
                "scores": {
                    "approved": int,
                    "fda": float,
                    "ftc": float
                },
                "filtered_evaluations": [
                    { policy_id, policy, type, grade, reason }, ...
                ],
                "approved_matches": [
                    { policy_id, policy, type, grade, reason }, ...
                ],
                "approved_match_summary": str,
                "overall_summary": str
            }
    """
    ignoreScore = 90
    df = pd.DataFrame(response["evaluations"])

    # SAFETY: derive type from policy_id
    df["type"] = df["policy_id"].str.split(":").str[0]

    # Filter relevant
    passing_mask = (
        (df["type"] != "approved") &
        ~(
            ((df["type"] == "fda") | (df["type"] == "ftc")) &
            (df["grade"] >= ignoreScore)
        )
    )

    df_filtered = df[passing_mask].sort_values(
        by="grade", ascending=True
    )

    # Convert early → pure Python
    filtered_list = df_filtered.to_dict(orient="records")

    # Approved logic
    approved_df = df[df["type"] == "approved"]
    approved_score = (
        int(approved_df["grade"].max()) if not approved_df.empty else 0
    )

    approved_matches_df = approved_df[approved_df["grade"] >= 50]
    approved_matches = approved_matches_df.to_dict(orient="records")

    approved_match_summary = (
        "Matched approved claims found."
        if approved_matches else
        "No matching approved claims."
    )

    # Score logic
    def compute_category_score(df, category, threshold=70):
        scores = df[df["type"] == category]["grade"]
        if scores.empty:
            return 0
        base = float(scores.mean())
        penalty = float((scores < threshold).mean() * 100)
        return max(0, min(100, base - penalty))

    fda_score = compute_category_score(df, "fda")
    ftc_score = compute_category_score(df, "ftc")

    # Return everything JSON-safe and clean
    return {
        "scores": {
            "approved": approved_score,
            "fda": fda_score,
            "ftc": ftc_score,
        },
        "filtered_evaluations": filtered_list,
        "approved_matches": approved_matches,
        "approved_match_summary": approved_match_summary,
        "overall_summary": response.get("overall_summary", ""),
    }