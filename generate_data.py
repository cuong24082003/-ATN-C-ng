import pandas as pd
import numpy as np

print("Script is running...")

np.random.seed(42)

normal = pd.DataFrame({
    "requests_per_min": np.random.normal(12, 3, 900),
    "session_duration": np.random.normal(300, 50, 900),
    "failed_login": np.random.randint(0, 2, 900)
})

anomaly = pd.DataFrame({
    "requests_per_min": np.random.normal(60, 10, 100),
    "session_duration": np.random.normal(30, 10, 100),
    "failed_login": np.random.randint(3, 7, 100)
})

data = pd.concat([normal, anomaly])
data.to_csv("data.csv", index=False)

print("Dataset created with 1000 rows")
