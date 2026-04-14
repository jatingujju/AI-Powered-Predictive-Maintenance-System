import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

temperature = np.random.normal(70, 10, rows)
vibration = np.random.normal(3, 1, rows)
pressure = np.random.normal(30, 5, rows)

# Failure logic (simulate real-world)
failure = (temperature > 85) | (vibration > 5) | (pressure > 40)
failure = failure.astype(int)

df = pd.DataFrame({
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'failure': failure
})

df.to_csv("data/data.csv", index=False)

print("Dataset Generated Successfully!")