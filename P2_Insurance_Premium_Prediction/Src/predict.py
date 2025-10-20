import pandas as pd
import joblib
from .preprocess import encode_features

def predict_new_data(input_path=r'D:\Python Projects\4-ML-Projects\P2_Insurance_Premium_Prediction\Data\insurance - Test.csv',
                     model_path=r'D:\Python Projects\4-ML-Projects\P2_Insurance_Premium_Prediction\Model\top_model.pkl') -> pd.DataFrame:
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    df_encoded = encode_features(df)
    predictions = model.predict(df_encoded)
    df['Predicted Charges'] = predictions
    return df