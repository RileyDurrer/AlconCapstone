import promptBuilder as pb

#input marketing material replace later with website output
marketing_txt = input("Please provide the marketing material: ")

prompt = pb.buildPrompt(marketing_txt)
print(prompt)




