import pandas as pd

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
    return df