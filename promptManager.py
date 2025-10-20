import dotenv
dotenv.load_dotenv()

def buildPrompt(marketing_txt, product):
    system_prompt = """
        You are a compliance grader.
        For each compliance policy, output:
        - rule: the policy text
        - pass: 2 if the text passes, 0 if it fails, 1 if a warning
        - reason: short explanation

        Return ONLY valid JSON in the format:
        {
        "results": [
            {"rule": "...", "pass": 0 or 1, "reason": "..."},
            {"rule": "...", "pass": 0 or 1, "reason": "..."}
        ]
        }
        """
    
    
    prompt = f"Grade the following marketing material based on how it complies with the following policies and regulation \n\n{marketing_txt}\n\nProduct Description:"
    return prompt

def getResponseFromLLM(prompt):