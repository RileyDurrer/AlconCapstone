from openai import OpenAI

    
def getChatbotResponse(user_prompt: str, client) -> dict:
    system_prompt="You are a friendly chatbot with the goal of helping Alcon marketing empoyees understand any complience issues their marketing material may have and guide them towrds how they can fix it, you will be given the chat history followed by their current compliance grade and finally the users prompt. Keep it brief and constructive"
    response = client.responses.create(
        model="gpt-4.1-mini", 
        input=[
            {"role": "system", "content": system_prompt}
            {"role": "context", "content": chatHistory}
            {"role": "compliance_grade", "content": ComplianceGrade}
            {"role": "user", "content": user_prompt}
        ]
    )
    return response