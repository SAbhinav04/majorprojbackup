import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Load the trained XGBoost model
model_path = 'xgboost_stress_model.joblib'
model = joblib.load(model_path)

# Sample input data (user data)
user_data = {
    'Steps': 8000,        # Steps taken
    'HeartRate': 85,      # Heart rate (beats per minute)
    'Calories': 1800,     # Calories consumed
    'SleepHours': 6,      # Hours of sleep
    'ScreenTime': 9       # Hours of screen time
}

# Function to calculate stress level using a formula (for transparency)
def calculate_stress_formula(data):
    # Formula to calculate stress based on user data
    stress = (data['HeartRate'] * 0.25) + (data['Calories'] * 0.05) - (data['SleepHours'] * 2.5) + (data['ScreenTime'] * 0.3)
    return stress

# Function to give suggestions based on user data
def give_suggestions(data):
    suggestions = []

    # Sleep Hours Suggestion
    if data['SleepHours'] < 7:
        suggestions.append("Sleep is below the recommended hours. Try to get at least 7-8 hours of sleep for better health.")
    
    # High Heart Rate Suggestion
    if data['HeartRate'] > 100:
        suggestions.append("Your heart rate is high. Consider lowering your physical activity or consulting a doctor if it persists.")
    
    # High Screen Time Suggestion
    if data['ScreenTime'] > 8:
        suggestions.append("Excessive screen time detected. Try to reduce screen time for better mental health.")
    
    # Low Activity Suggestion
    if data['Calories'] < 2000 and data['Steps'] < 5000:
        suggestions.append("Low physical activity and calorie intake. Try increasing your steps and improving your diet.")

    return suggestions

# Preprocess the input data for XGBoost model (if necessary)
user_input = pd.DataFrame([user_data])

# Predict stress level using the trained XGBoost model
predicted_stress = model.predict(user_input)

# Calculate stress level using the formula for transparency
calculated_stress = calculate_stress_formula(user_data)

# Get suggestions based on user data
suggestions = give_suggestions(user_data)

# Display the results
print(f"Predicted Stress Level (XGBoost): {predicted_stress[0]:.2f} out of 100")
print(f"Calculated Stress Level (Formula): {calculated_stress:.2f} out of 100")
print("Suggestions for improvement:")
for suggestion in suggestions:
    print(f"- {suggestion}")