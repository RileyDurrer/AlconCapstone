import dotenv
dotenv.load_dotenv()
import pandas as pd

from openai import OpenAI

import promptBuilder as pb

OpenAI.api_key = dotenv.get("OPENAI_API_KEY")



#input marketing material replace later with website output
marketing_txt = input("Please provide the marketing material: ")

prompt = pb.buildPrompt(marketing_txt)
print(prompt)



response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a mentor."},
        {"role": "user", "content": "Suggest 3 project ideas in AI + fintech."}
    ],
    max_tokens=400,
    temperature=0.7,
    n=2,
    response_format={"type": "json"},
)

print(response.choices[0].message.content)




