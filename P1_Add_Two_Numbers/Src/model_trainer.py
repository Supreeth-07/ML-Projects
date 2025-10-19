import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(df):
    X = df[['x', 'y']]
    y = df['sum']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def save_model(model, filename=r'D:\Python Projects\4-ML-Projects\P1_Add_Two_Numbers\Model\model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

