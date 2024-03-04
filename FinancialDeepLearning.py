import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def download_stock_data(symbol):
    """
    Télécharge les données historiques d'une action.
    """
    data = yf.download(symbol, start='2010-01-01', end='2023-01-01')
    return data

def feature_engineering(data):
    """
    Ajoute des indicateurs techniques comme caractéristiques.
    """
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_RSI(data['Close'], 14)
    return data.dropna()

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def preprocess_data(data):
    """
    Prétraitement des données : normalisation et création de séquences.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])  # Prédiction du prix de fermeture
    X, y = np.array(X), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape):
    """
    Construit un modèle LSTM pour la prédiction du prix des actions.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue la performance du modèle sur l'ensemble de test.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")

def predict_future_prices(model, data, days=60, future_days=30):
    """
    Prédit les prix futurs à partir des dernières données disponibles.
    """
    last_60_days = data[-60:].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(last_60_days)
    
    future_prices = []
    for _ in range(future_days):
        x = scaled_data[-60:].reshape(1, 60, 1)
        prediction = model.predict(x)
        future_prices.append(prediction[0][0])
        scaled_data = np.append(scaled_data, prediction)[1:].reshape(-1, 1)
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices

def main():
    symbol = 'AAPL'
    data = download_stock_data(symbol)
    data_prepared = feature_engineering(data)
    X_train, X_test, y_train, y_test = preprocess_data(data_prepared[['Close', 'SMA_50', 'SMA_200', 'RSI']].values)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
    
    # Visualisation des pertes
    plt.plot(model.history.history['loss'], label='Train Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Évaluation du modèle
    evaluate_model(model, X_test, y_test)

    # Prédiction des prix futurs
    future_prices = predict_future_prices(model, data_prepared['Close'])
    print("Future prices prediction:", future_prices)

if __name__ == "__main__":
    main()

