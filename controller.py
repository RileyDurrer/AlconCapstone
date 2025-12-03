import dotenv
import os
import json

import pandas as pd
from openai import OpenAI

from complianceGrader import build_compliance_prompt, get_response_from_LLM, structure_compliance_grade, read_product_policy_files
from chatbot import get_chatbot_response

dotenv.load_dotenv()

class ComplianceController:
    """
    Manages compliance checks, LLM interactions, chatbot memory, and policy loading
    for a single user session.

    Responsibilities:
    - Load compliance policies once per product.
    - Generate structured compliance evaluations.
    - Maintain short-term and long-term chat memory.
    - Produce conversational chatbot responses that incorporate compliance context.

    Attributes:
        strictness (int):
            Global grading strictness (1–10), loaded from environment.
        client (OpenAI):
            OpenAI API client reused across all calls.
        chat_history (list[dict]):
            Conversation memory in message-format:
            [{ "role": "user"|"assistant", "content": str }, ...]
        state (dict):
            Long-term session memory:
            {
                "current_compliance": dict | None,
                "previous_compliance": dict | None,
                "policies": dict | None
            }
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
        """
        Adds a new message to the controller's chat history and enforces
        a rolling memory window.

        The chat history is stored as a list of message dictionaries, each with:
        - "role": one of ("user", "assistant")
        - "content": the message text

        To keep the history manageable for the model, this function caps the list
        at 10 messages. When the limit is exceeded, the oldest message is removed.

        Args:
            role (str):
                The speaker role (e.g., "user" or "assistant").
            content (str):
                The text content of the message.

        Returns:
            None
        """
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

    # -----------------------------------------------------
    # PRODUCT LOADING
    # -----------------------------------------------------
    def load_product_policies(self, product: str):
        """
        Loads and stores all compliance policy DataFrames for the specified product.

        This updates the controller's state with:
        - the selected product name
        - the loaded policy DataFrames (approved, fda, ftc)

        Args:
            product (str):
                The product name whose compliance policy files should be loaded
                from ./compliance_docs/{product}_compliance/.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the product folder or required CSV files are missing.
        """
        self.state["product"] = product
        self.policies = read_product_policy_files(product)

        self.add_message("assistant", f"Product set to: {product}")

    # -----------------------------------------------------
    # COMPLIANCE CHECK
    # -----------------------------------------------------
    def run_compliance_check(self, marketing_text: str) -> dict:
        """
        Manages compliance checks, LLM interactions, chatbot memory, and policy loading
        for a single user session.

        Responsibilities:
        - Load compliance policies once per product.
        - Generate structured compliance evaluations.
        - Maintain short-term and long-term chat memory.
        - Produce conversational chatbot responses that incorporate compliance context.

        Attributes:
            strictness (int):
                Global grading strictness (1–10), loaded from environment.
            client (OpenAI):
                OpenAI API client reused across all calls.
            chat_history (list[dict]):
                Conversation memory in message-format:
                [{ "role": "user"|"assistant", "content": str }, ...]
            state (dict):
                Long-term session memory:
                {
                    "current_compliance": dict | None,
                    "previous_compliance": dict | None,
                    "policies": dict | None
                }
        """
        if not self.state["product"]:
            raise ValueError("No product selected. Call set_product(product) first.")

        prompt = build_compliance_prompt(
            marketing_text,
            policies=self.policies,
            strictness=self.strictness
        )

        result = get_response_from_LLM(prompt, self.client)
        compliance = structure_compliance_grade(result)

        # Update semantic memory
        self.state["previous_compliance"] = self.state["last_compliance"]
        self.state["last_compliance"] = compliance

        # Short-term context
        self.add_message("assistant", "Compliance check updated.")

        return compliance
    
    # -----------------------------------------------------
    # CHATBOT MESSAGE
    # -----------------------------------------------------
    def chat_bot_message(self, user_message: str) -> str:
        """
        Generate a chatbot response incorporating conversation history,
        compliance context, and improvement suggestions.

        Steps:
        - Add user message to chat history.
        - Call getChatbotResponse() with short-term + long-term memory.
        - Store assistant reply in history.
        - Return assistant message.

        Args:
            user_message (str):
                The user's new chat input.

        Returns:
            str:
                The chatbot's reply text.
        """

        # Add user message to short-term memory
        self.add_message("user", user_message)

        # Generate chatbot reply using the memory architecture
        reply = get_chatbot_response(
            user_prompt=user_message,
            client=self.client,
            short_history=self.chat_history,
            long_term_state=self.state,
        )

        # Store assistant reply in short-term memory
        self.add_message("assistant", reply)

        # Return the text back to the notebook UI
        return reply


