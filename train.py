#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def load_data(filename):
    """Load mileage and price data from a CSV file"""
    mileage = []
    price = []
    try:
        with open(filename) as f:
            r = csv.DictReader(f)
            if "km" not in r.fieldnames or "price" not in r.fieldnames:
                raise ValueError("CSV must contain 'km' and 'price' columns")
            for i, row in enumerate(r, start=2):
                try:
                    k_val = float(row['km'])
                    p_val = float(row['price'])
                    if k_val < 0 or p_val < 0:
                        raise ValueError(f"Negative value at row {i}: \
                                           km={k_val}, price={p_val}")
                    mileage.append(k_val)
                    price.append(p_val)
                except ValueError as err:
                    print(f"Skipping row {i}: {err}")
        if len(mileage) == 0:
            raise ValueError("No valid data found in file")
        return np.array(mileage), np.array(price)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for file '{filename}'.")
        sys.exit(1)
    except ValueError as err:
        print(f"Error reading CSV: {err}")
        sys.exit(1)


def train(x, y, lr=0.01, epochs=1000):
    m = len(y)
    theta0, theta1 = 0, 0

    for _ in range(epochs):
        error = theta0 + theta1 * x - y
        theta0 -= lr * np.sum(error) / m
        theta1 -= lr * np.sum(error * x) / m
    return theta0, theta1


def predict(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train.py <dataset.csv>")
        sys.exit(1)
    filename = sys.argv[1]
    x, y = load_data(filename)
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_norm = (x - x_mean) / x_std;

    theta0, theta1 = train(x_norm, y, lr=0.01, epochs=2000)
    with open("weights.txt", "w") as f:
        f.write(f"{theta0}\n{theta1}\n")
        f.write(f"{x_mean}\n{x_std}\n")

    y_pred = predict(x_norm, theta0, theta1)

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color=(0.337, 0.463, 0.965), label='Data points')
    plt.plot(x, y_pred, color=(0.958, 0.372, 0), label='Model prediction')
    plt.xlabel("km")
    plt.ylabel("Price")
    plt.title("Linear Regression - Car Price Prediction by Mileage")
    plt.legend()
    plt.show()
