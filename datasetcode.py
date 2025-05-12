import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=3000, seed=42):
    np.random.seed(seed)
    
    data = {
        "Steps": np.random.randint(1000, 15000, n_samples),
        "HeartRate": np.random.randint(60, 130, n_samples),
        "Calories": np.random.randint(1200, 3500, n_samples),
        "SleepHours": np.random.uniform(3.5, 9.5, n_samples),
        "ScreenTime": np.random.uniform(1.0, 10.0, n_samples),
    }

    df = pd.DataFrame(data)

    # Logic-based stress score out of 100
    stress = []

    for i in range(n_samples):
        hr = df.loc[i, "HeartRate"]
        steps = df.loc[i, "Steps"]
        sleep = df.loc[i, "SleepHours"]
        screen = df.loc[i, "ScreenTime"]
        cal = df.loc[i, "Calories"]
        
        # Directional impact
        hr_score = (hr - 60) / 70  # Normalize 60–130
        steps_score = 1 - min(steps / 10000, 1)  # Higher steps = lower stress
        sleep_score = abs(sleep - 7.5) / 4  # Ideal 7.5 hours
        screen_score = min(screen / 10, 1)  # More screen time = more stress
        cal_score = abs(cal - 2200) / 1300  # Deviation from 2200 kcal

        # Weighted sum (adjust weights based on experimentation)
        stress_level = (
            0.3 * hr_score +
            0.25 * steps_score +
            0.2 * sleep_score +
            0.15 * screen_score +
            0.1 * cal_score
        ) * 100
        
        stress.append(round(min(max(stress_level, 0), 100), 2))

    df["StressLevel"] = stress
    return df

# Save dataset
df_generated = generate_synthetic_data(3000)
df_generated.to_csv("stress_data_3000_rows.csv", index=False)
print("Dataset generated and saved as 'stress_data_1000_rows.csv'")
