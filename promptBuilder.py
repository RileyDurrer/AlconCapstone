def buildPrompt(marketing_txt):
    prompt = f"Grade the following marketing material based on how it complies with the following policies and regulation \n\n{marketing_txt}\n\nProduct Description:"
    return prompt