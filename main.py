from controller import run_compliance_check
import json

#Running this program in the terminal does not give you access to the built in chatbot
def main():
    #Get marketing material
    text = input("Enter marketing material: ") 

    # Select product - as you add more products it would be better to start using IDs but its using names for ease of understanding here.
    product=None
    while product is None:
        product_num=input("1: Clareon® Toric Intraocular Lens (IOL) or 2:  PRECISION7® Contact Lenses")

        if product_num=="1":
            product="ClareonPanOptix"

        elif product_num=="2":
            product="TOTAL30"
        else:
            print("Invalid input, please enter 1 or 2.")
            
    result = run_compliance_check(text, product_num)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
