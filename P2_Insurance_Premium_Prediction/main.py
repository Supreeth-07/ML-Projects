import pandas as pd
from Src.train import train_and_save_model
from Src.predict import predict_new_data

def main():
    # Step 1: Load sample training data (or load your full dataset)

    df = pd.read_csv(r"D:\Python Projects\4-ML-Projects\P2_Insurance_Premium_Prediction\Data\insurance.csv")

    # Step 2: Train and save the model
    train_and_save_model(df)

    # Step 3: Predict new data
    result = predict_new_data()
    print("\n🔮 Predictions for New Customers:")
    print(result)

    # Optional: save output
    result.to_csv(r"D:\Python Projects\4-ML-Projects\P2_Insurance_Premium_Prediction\Data\insurance - Test Output.csv", index=True)

if __name__ == "__main__":
    main()
