#from fastapi import FastAPI, UploadFile, File, Form
import os
import json
import promptManager
import outputFormating
import pandas as pd

#Get marketing material
marketing_material = input("Enter marketing material: ") 

# Select product - as you add more products it would be better to start using IDs but we are using names for ease of reading here.
product=None
while product is None:
    product_num=input("1: Clareon® Toric Intraocular Lens (IOL) or 2:  PRECISION7® Contact Lenses")

    if product_num=="1":
        product="ClareonPanOptix"

    elif product_num=="2":
        product="TOTAL30"
    else:
        print("Invalid input, please enter 1 or 2.")
    
    print("Selected product: ", product)
    

prompt=promptManager.buildPrompt(marketing_material, product)
print("Prompt built: ", prompt)

#Get response from LLM
output=promptManager.getResponseFromLLM(prompt)

print("Output from LLM: ", output)

#Format output
#formatted_output=outputFormating.formatOutput(output)
#print("Formatted Output: ", formatted_output)