import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Load the dataset






df = pd.read_csv("stress_data_1000_rows.csv")

features = ["Steps", "HeartRate", "Calories", "SleepHours", "ScreenTime"]
target = "StressLevel"

# Drop missing values if any
df.dropna(subset=features + [target], inplace=True)

# Scale features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Save scaler for future prediction use
joblib.dump(scaler, "scaler.joblib")

# Prepare inputs
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "xgboost_stress_model.joblib")

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual')
plt.plot(y_pred[:50], label='Predicted')
plt.title("Actual vs Predicted Stress Levels")
plt.xlabel("Sample")
plt.ylabel("Stress Level")
plt.legend()
plt.grid(True)
plt.show()
