from Src.data_loader import load_data
from Src.model_trainer import train_model, save_model
from Src.model_evaluator import evaluate_model
from Src.predictor import load_model, predict

def main():
    # 1. Load Data
    df = load_data(r'D:\Python Projects\4-ML-Projects\P1_Add_Two_Numbers\Data\add.csv')

    # 2. Train Model
    model, X_test, y_test = train_model(df)

    # 3. Save Model
    save_model(model)

    # 4. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    print("📊 Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 5. Predict on new values
    loaded_model = load_model()
    preds = predict(loaded_model, [10, 50, 100], [5, 20, 300])
    print("\n🔮 Predictions:")
    for x, y, pred in zip([10, 50, 100], [5, 20, 300], preds):
        print(f"{x} + {y} = {pred:.2f}")

if __name__ == '__main__':
    main()
