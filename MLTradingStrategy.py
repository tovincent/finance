import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Téléchargement des données
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

# Calcul des indicateurs techniques
data['EMA_10'] = data['Adj Close'].ewm(span=10, adjust=False).mean()
data['RSI'] = data['Adj Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean() / \
              data['Adj Close'].diff().abs().rolling(window=14).mean() * 100
data['MACD'] = data['Adj Close'].ewm(span=12, adjust=False).mean() - data['Adj Close'].ewm(span=26, adjust=False).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Préparation des données pour le modèle
data['Target'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
features = ['EMA_10', 'RSI', 'MACD', 'Signal_Line']
X = data[features].dropna()
y = data['Target'].shift(-1).dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions et évaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Simulation de stratégie de trading
data['Predictions'] = np.nan
data.iloc[(len(data) - len(predictions)):, data.columns.get_loc('Predictions')] = predictions
data['Strategy'] = data['Predictions'].shift(1) * data['Adj Close'].pct_change()
data['Returns'] = data['Adj Close'].pct_change()
data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Strategy Returns vs. Market Returns')
plt.show()
