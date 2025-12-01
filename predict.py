import sys

def load_weights(filename="weights.txt"):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                raise ValueError("weights.txt must contain two lines: theta0 and theta1")
            theta0 = float(lines[0].strip())
            theta1 = float(lines[1].strip())
            return theta0, theta1
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error reading weights: {e}")
        sys.exit(1)

def predict_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    theta0, theta1 = load_weights()

    try:
        mileage = float(input("Enter car mileage: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        sys.exit(1)

    price = predict_price(mileage, theta0, theta1)
    print(f"Predicted price: {price:.2f}")

if __name__ == "__main__":
    main()
