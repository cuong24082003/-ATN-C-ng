import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

print("Training models...")

df = pd.read_csv("data.csv")

df["fail_ratio"] = df["failed_login"] / (df["requests_per_min"] + 1)
df["activity_score"] = df["requests_per_min"] / (df["session_duration"] + 1)

X = df

iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X)
joblib.dump(iso, "iso_model.pkl")

svm = OneClassSVM(nu=0.1)
svm.fit(X)
joblib.dump(svm, "svm_model.pkl")

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X)
joblib.dump(lof, "lof_model.pkl")

print("Models trained and saved!")
