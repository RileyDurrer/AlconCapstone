from controller import ComplianceController
import json

def main():
    print("=== Alcon Compliance Checker ===\n")

    # Create controller instance (sets up memory + OpenAI client)
    ctrl = ComplianceController()

    # -----------------------------
    # SELECT PRODUCT
    # -----------------------------
    product_map = {
        "1": "ClareonPanOptix",
        "2": "TOTAL30"
    }

    product_choice = None
    while product_choice not in product_map:
        product_choice = input(
            "Select product:\n"
            "  1: Clareon® Toric Intraocular Lens (IOL)\n"
            "  2: PRECISION7® Contact Lenses\n"
            "\nEnter number: "
        )
        if product_choice not in product_map:
            print("Invalid input. Please enter 1 or 2.\n")

    product = product_map[product_choice]
    ctrl.set_product(product)

    print(f"\n✔ Product set to: {product}\n")

    # -----------------------------
    # GET MARKETING MATERIAL
    # -----------------------------
    text = input("Enter marketing material:\n> ")

    # -----------------------------
    # RUN COMPLIANCE CHECK
    # -----------------------------
    print("\nRunning compliance check...\n")

    try:
        compliance = ctrl.run_compliance_check(text)
    except Exception as e:
        print(f"❌ Error running compliance check: {e}")
        return

    # Pretty JSON output
    print("=== Compliance Result ===")
    print(json.dumps(compliance, indent=2))

    print("\nDone.")

if __name__ == "__main__":
    main()
