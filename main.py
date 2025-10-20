#from fastapi import FastAPI, UploadFile, File, Form
import os

import promptManager
import outputFormating



#Get marketing material
marketing_material = input("Enter marketing material: ") 

# Select product
product=None
while product==None:
    product_num=input("1: Clareon® Toric Intraocular Lens (IOL) or 2:  PRECISION7® Contact Lenses")

    if product_num=="1":
        product="Clareon® Toric Intraocular Lens (IOL)"

    elif product_num=="2":
        product="PRECISION7® Contact Lenses"
    else:
        print("Invalid input, please enter 1 or 2.")
    

prompt=promptManager.buildPrompt(marketing_material, product)
print("Prompt built: ", prompt)

#Get response from LLM
output=promptManager.getResponseFromLLM(prompt)

print("Output from LLM: ", output)

#Format output
formatted_output=outputFormating.formatOutput(output)
print("Formatted Output: ", formatted_output)