import dotenv
import os
import json

import pandas as pd
from openai import OpenAI

from complianceGrader import buildCompliancePrompt, getResponseFromLLM, structureComplianceGrade, load_product_policies
from chatbot import getChatbotResponse

dotenv.load_dotenv()

class ComplianceController:
    """Creates an instance of the ComplianceController to 
    manage compliance checks and chatbot interactions. 
    Creating a new instance will reset chat history and 
    compliance grade history.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.strictness = int(os.getenv("STRICTNESS", 5))

        # Short-term memory (chat history)
        self.chat_history = []

        # Long-term semantic memory
        self.state = {
            "product": None,
            "last_compliance": None,
            "previous_compliance": None
        }

        # Loaded policies (approved / fda / ftc)
        self.policies = None

    # -----------------------------------------------------
    # MEMORY: short-term (last 10 messages)
    # -----------------------------------------------------
    def add_message(self, role, content):
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

    # -----------------------------------------------------
    # PRODUCT LOADING
    # -----------------------------------------------------
    def set_product(self, product: str):
        """
        Sets the active product and loads all compliance policies once.
        """
        self.state["product"] = product
        self.policies = load_product_policies(product)

        self.add_message("assistant", f"Product set to: {product}")

    # -----------------------------------------------------
    # COMPLIANCE CHECK
    # -----------------------------------------------------
    def run_compliance_check(self, marketing_text: str):
        """
        Uses the already-loaded product policies.
        """
        if not self.state["product"]:
            raise ValueError("No product selected. Call set_product(product) first.")

        prompt = buildCompliancePrompt(
            marketing_text,
            policies=self.policies,
            strictness=self.strictness
        )

        result = getResponseFromLLM(prompt, self.client)
        compliance = structureComplianceGrade(result)

        # Update semantic memory
        self.state["previous_compliance"] = self.state["last_compliance"]
        self.state["last_compliance"] = compliance

        # Short-term context
        self.add_message("assistant", "Compliance check updated.")

        return compliance
    
    # -----------------------------------------------------
    # CHATBOT MESSAGE
    # -----------------------------------------------------
    def chat_bot_message(self, user_message: str):
        # Add user message to short-term memory
        self.add_message("user", user_message)

        # Generate chatbot reply using the memory architecture
        reply = getChatbotResponse(
            user_prompt=user_message,
            client=self.client,
            short_history=self.chat_history,
            long_term_state=self.state,
        )

        # Store assistant reply in short-term memory
        self.add_message("assistant", reply)

        # Return the text back to the notebook UI
        return reply


