import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
from .preprocess import encode_features

def train_and_save_model(df: pd.DataFrame, model_path=r'D:\Python Projects\4-ML-Projects\P2_Insurance_Premium_Prediction\Model\top_model.pkl'):
    df = encode_features(df)

    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))

    print(f"Train R²: {train_score:.3f}")
    print(f"Test R²: {test_score:.3f}")

    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")