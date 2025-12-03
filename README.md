# Alcon Marketing Compliance Grader – README

## Overview

The Alcon Marketing Compliance Grader is a backend system designed to analyze marketing text for compliance with approved Alcon product claims, FDA requirements, and FTC advertising regulations.
It uses OpenAI’s API to evaluate content, generate structured JSON results, and provide chatbot-based guidance for revisions.

**This system includes:**

A Compliance Engine (grading + scoring + filtering)

A Memory-based Chatbot for feedback and iteration

A Controller Layer that manages state, policies, and messaging

CSV-driven compliance data storage for each product

It is designed to run as a backend service, with an interactive Jupyter Notebook front-end for visualization and manual testing.

```text
AlconCapstone/
│
├── controller.py                # Main controller: policies, grading, chatbot state
├── complianceGrader.py          # Prompt builder, OpenAI call, results structuring
├── chatbot.py                   # Chat assistant using memory + compliance state
├── interface.ipynb              # Jupyter demo interface for live testing
│
├── compliance_docs/
│   ├── ClareonPanOptix/
│   │   ├── ApprovedClaims.csv
│   │   ├── FDAPolicies.csv
│   │   ├── FTCPolicies.csv
│   │
│   ├── TOTAL30/
│       ├── ApprovedClaims.csv
│       ├── FDAPolicies.csv
│       ├── FTCPolicies.csv
│
├── .env                         # API key / config values (ignored)
├── .env.example                 # Template for required environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## Compliance Check

### 1. Load Policies (per-product)

Each product folder contains three CSVs:

ApprovedClaims.csv
FDAPolicies.csv
FTCPolicies.csv

The controller loads these once:

self.policies = {
    "approved": df1,
    "fda": df2,
    "ftc": df3
}

These DataFrames are then passed into the compliance prompt builder.

### 2. Build the LLM Prompt

build_compliance_prompt() composes a structured prompt containing:

- Marketing text
- All policies grouped + labeled with policy_id
- Strictness level (1–10)

Output JSON formatting rules
Grading rules for each policy type
This ensures output is always machine-readable.

### 3. Run Compliance Grading

OpenAI’s API receives the built prompt and returns full JSON:

result = get_response_from_LLM(prompt)

*This JSON includes:*
List of all evaluations

- Grade per policy
- Low-level reasons
  Summary text

### 4. Structure + Filter the Results

The compliance engine:

**Filters out irrelevant FDA/FTC items**
(score ≥ 85 = user does NOT need to worry)

**Computes category scores**
Approved highest match
FDA combined score
FTC combined score

**Identifies relevant approved claims**
(grade ≥ 50)

**Converts everything into JSON-safe lists**
(no pandas objects, no numpy types)

**Returned structure:**

{
  "scores": { "approved": 82, "fda": 94, "ftc": 77 },
  "filtered_evaluations": [...],
  "approved_matches": [...],
  "approved_match_summary": "Matched approved claims found.",
  "overall_summary": "..."
}

## Chatbot Architecture (with Memory)

The chatbot uses 3 types of memory:

**1. Conversation Memory**
(Recent user + assistant messages)
Used for natural chat context.

**2. State Memory**
(Stored in the controller)
Includes:

- most recent compliance result
- previous compliance results
- strictness
- product being evaluated

**3. Fix Suggestion Generation**
Chatbot automatically creates targeted suggestions based on:

- policy violations
- missing alignment with approved claims
- severity of issues
  historical improvements (optional)

## Environment Setup

.env File

Create a .env file in the project root:
OPENAI_API_KEY=your_key_here
STRICTNESS=5

STRICTNESS ranges 1–10 and adjusts how harshly violations are judged.

## Installing Dependencies

pip install -r requirements.txt
or in venv:
pip install -r requirements.txt

## Using the Controller in a Script

from controller import ComplianceController
ctrl = ComplianceController(product="TOTAL30")
material = "Feels great all month."
result = ctrl.run_compliance_check(material)
print(result)

response = ctrl.chat_bot_message("How do I fix this?")
print(response)

## the Interactive Notebook

The Jupyter Notebook provides:
Live chat interface
/grade command to re-run compliance

**Conversation memory**
Example cell:
message = input("You: ")
response = ctrl.chat_bot_message(message)
print(response)

**CLI Version**
You can run the simplified CLI through main.py.
python main.py

## Adding a New Product

Create folder:
compliance_docs/NewProduct/

Add:

- ApprovedClaims.csv
- FDAPolicies.csv
- FTCPolicies.csv

**In controller:**
ctrl = ComplianceController(product="NewProduct")

## Planned Extensions

Streaming responses
Web front-end (React or Streamlit)
Multi-product / multi-version support
Policy “weights” for more advanced scoring
Exportable audit summaries

## Contact:

If you need help extending or integrating the system, message anytime.
https://www.linkedin.com/in/rileydurrer/
