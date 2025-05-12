import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import os
# Path to the new CSV file you generated
new_data_csv = 'stress_data_1000_rows.csv'

# Master data path (if you want to append new data to a master dataset)
master_csv = 'master_data.csv'
model_path = 'xgboost_stress_model.joblib'

# Load new data
df_new = pd.read_csv(new_data_csv)

# If a master file exists, load it and append new data
if os.path.exists(master_csv):
    master_data = pd.read_csv(master_csv)
    updated_data = pd.concat([master_data, df_new], ignore_index=True)
else:
    # If no master file exists, start with new data as the dataset
    updated_data = df_new

# Save the updated master dataset (optional if you're maintaining it)
updated_data.to_csv(master_csv, index=False)

# Select features and target
features = ['Steps', 'HeartRate', 'Calories', 'SleepHours', 'ScreenTime']
target = 'StressLevel'

updated_data.dropna(subset=features + [target], inplace=True)

# Scale features
scaler = MinMaxScaler()
updated_data[features] = scaler.fit_transform(updated_data[features])

# Split data into X and y
X = updated_data[features]
y = updated_data[target]

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a new XGBoost model
print("Training model on updated master dataset...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, model_path)

# Predict and evaluate
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual')
plt.plot(pred[:50], label='Predicted')
plt.title("Actual vs Predicted Stress Levels")
plt.xlabel("Sample")
plt.ylabel("Stress Level")
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot
residuals = y_test - pred
plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals Plot")
plt.xlabel("Actual Stress Level")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()
