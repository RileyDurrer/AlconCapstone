import dotenv
import os

import pandas as pd
import openai as OpenAI

from complianceGrader import buildCompliancePrompt
from complianceGrader import getResponseFromLLM
from complianceGrader import structureComplianceGrade

from chatbot import getChatbotResponse

dotenv.load_dotenv()
STRICTNESS = int(os.getenv("STRICTNESS", 5))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chatHistory=[]
complianceGrade=None


def run_compliance_check(marketing_text: str, product: str):
    """Generates a detailed complience score for a given text input and Alcon product

    Args:
        marketing_text (str): Textual Marketing Material for an alcon product
        product (str): name of an Alcon product, must be added as a folder in the compliance_docs folder 

    Returns:
        _type_: _description_
    """
    prompt = buildCompliancePrompt(marketing_text, product=product, strictness=STRICTNESS)
    result = getResponseFromLLM(prompt, client)
    compliance = structureComplianceGrade(result)
    complianceGrade=compliance
    #Fix by turning json into string
    chatHistory.append("Updated Compliance: "+compliance)
    return compliance

def chat_bot_message(message: str, client=client, history=chatHistory, compliance=complianceGrade):
    chatHistory.append("User: "+message)
    chatbotResponse=getChatbotResponse(message, client, history, compliance)
    chatHistory.append("Chatbot: "+chatbotResponse)
    
    












