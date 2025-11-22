import joblib
import pandas as pd
import sys
import numpy as np


def load_models():
    try:
        iso = joblib.load("iso_model.pkl")
        xgb = joblib.load("xgb_model.pkl")
        print("Models loaded successfully.\n")
        return iso, xgb
    except Exception as e:
        print("ERROR: Could not load models. Train and save them first.")
        print("Details:", e)
        sys.exit(1)


def get_user_input():
    print("\nEnter transaction values:")

    values = []

    time = float(input("Time: "))
    amount = float(input("Amount: "))
    values.extend([time, amount])

    print("\nEnter V1 to V28:")
    for i in range(1, 29):
        val = float(input(f"V{i}: "))
        values.append(val)

    FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def predict_single(model, X):
    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0][1]

    print("\nPrediction:")
    print("FRAUD DETECTED" if pred == 1 else "NOT fraud")
    print(f"Fraud probability: {prob:.4f}\n")

def predict_from_csv(iso_model, xgb_model):
    path = input("\nEnter path to CSV file: ")

    try:
        df = pd.read_csv(path)
        print("File loaded successfully.")
    except Exception as e:
        print("ERROR: Could not read file.\nDetails:", e)
        return

    REQUIRED = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    if not all(col in df.columns for col in REQUIRED):
        print("ERROR: CSV does not contain required features.")
        return

    X = df[REQUIRED]
    print(f"Rows loaded: {len(X)}")

    print("\nRunning predictions...")

    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred == -1, 1, 0)
    iso_scores = iso_model.decision_function(X)
    iso_prob = -iso_scores
    
    xgb_pred = xgb_model.predict(X)
    xgb_prob = xgb_model.predict_proba(X)[:, 1]

    output_df = pd.DataFrame({
        "IF_Pred": iso_pred,
        "IF_Prob": iso_prob,
        "XGB_Pred": xgb_pred,
        "XGB_Prob": xgb_prob
    })

    output_path = "predictions_output.csv"
    output_df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}\n")


def main():
    iso_model, xgb_model = load_models()

    while True:
        print("\n=========== CREDIT CARD FRAUD DETECTION ===========")
        print("1. Predict manually (enter each feature)")
        print("2. Predict for entire dataset (CSV)")
        print("3. Exit")
        print("===================================================")

        choice = input("Select an option: ").strip()

        if choice == "1":
            X = get_user_input()
            predict_single(xgb_model, X)

        elif choice == "2":
            predict_from_csv(iso_model, xgb_model)

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Try again.\n")


if __name__ == "__main__":
    main()
