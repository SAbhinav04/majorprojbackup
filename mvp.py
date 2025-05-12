import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import os

df1 = pd.read_csv('stress_data_new.csv')
df2 = pd.read_csv('stress_data_secondary.csv')


data = pd.concat([df1, df2], ignore_index=True)


# Create additional features if possible
#data['SleepQuality'] = data['SleepHours'] * data['HRV']  # example engineered feature
data['SleepQuality'] = data['SleepHours'] / (data['ScreenTime'] + 1)
# Select features
features = ['SleepHours', 'Steps', 'HeartRate', 'ScreenTime', 'SleepQuality']

target = 'StressLevel'

# Scale features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Create sequences
SEQ_LEN = 10
sequences = []
labels = []

for i in range(len(data) - SEQ_LEN):
    sequences.append(data[features].iloc[i:i+SEQ_LEN].values)
    labels.append(data[target].iloc[i+SEQ_LEN])

X = np.array(sequences)
y = np.array(labels)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load existing model or create new one
model_path = 'stress_lstm_model.keras'
if os.path.exists(model_path):
    print("Loading previous model...")
    model = load_model(model_path)
else:
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ_LEN, len(features))))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Dense(1, activation='linear'))  # if not bounded
    model.compile(loss='mse', optimizer='adam')


    # Train
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
              callbacks=[early_stop, lr_reduce])

    model.save(model_path)

# Predict
pred = model.predict(X_test).flatten()

plt.hist(y, bins=20)
plt.title("Stress Level Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Frequency")
plt.show()

import seaborn as sns

df = pd.read_csv('stress_data_new.csv')
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5  # manually calculate RMSE

r2 = r2_score(y_test, pred)

print(f"MAE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Residual plot
residuals = y_test - pred
plt.figure(figsize=(10,5))
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals Plot")
plt.xlabel("Actual Stress")
plt.ylabel("Error (Actual - Predicted)")
plt.grid(True)
plt.show()

# Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label='Actual')
plt.plot(pred[:50], label='Predicted')
plt.title("Actual vs Predicted (first 50 samples)")
plt.legend()
plt.show()

