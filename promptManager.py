import dotenv
import pandas as pd
import json

from openai import OpenAI

client = OpenAI()
dotenv.load_dotenv()

def buildPrompt(marketing_text, product):
    """
    Builds an LLM grading prompt that evaluates a marketing text
    against dynamically retrieved compliance policies.
    """
    policies = getCompliancePolicies(product)
    policies_formatted = "\n".join(
        [f"{i+1}. {p}" for i, p in enumerate(policies)]
    )

    prompt = f"""
        You are a compliance grader evaluating marketing content.

        **Task:** Review the provided marketing text and assign a numeric score (0–100)
        for how well it complies with each listed policy or regulation.

        ### Marketing Text
        {marketing_text}

        ### Compliance Policies and Regulations
        {policies_formatted}

        ### Instructions
        - Base your grades on factual compliance alignment, accuracy of claims, and regulatory adherence.
        - Give a **brief reason** for each score.
        - Higher scores mean stronger compliance.

        ### Output Format
        Return results as structured JSON:
        {{
        "evaluations": [
            {{
            "policy": "<policy_name>",
            "grade": <0-100>,
            "reason": "<brief justification>"
            }}
        ],
        "overall_summary": "<short summary of overall compliance>"
        }}
        """
    return prompt.strip()

def getCompliancePolicies(product):
    # Load compliance policies from a CSV file
    policies_df = pd.read_csv('compliance_policies.csv')
    policies = policies_df['policy_text'].tolist()
    return policies

def getResponseFromLLM(prompt):
    """Send the compliance prompt to the LLM and parse JSON output."""

    response = client.responses.create(
        model="gpt-4.1-mini",  # or "gpt-5" if available
        input=prompt,
        response_format={"type": "json_object"}
    )

    # Parse JSON safely
    content = response.output[0].content[0].text
    return json.loads(content)

