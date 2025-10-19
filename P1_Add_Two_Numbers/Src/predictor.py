import pickle
import pandas as pd

def load_model(filename=r'D:\Python Projects\4-ML-Projects\P1_Add_Two_Numbers\Model\model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, x_values, y_values):
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    predictions = model.predict(df)
    return predictions
