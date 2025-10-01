import dotenv
dotenv.load_dotenv()

from openai import OpenAI

import promptBuilder as pb

OpenAI.api_key = 

#input marketing material replace later with website output
marketing_txt = input("Please provide the marketing material: ")

prompt = pb.buildPrompt(marketing_txt)
print(prompt)




