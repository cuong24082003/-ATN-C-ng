from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

print("Loading AI models...")
iso = joblib.load("iso_model.pkl")
svm = joblib.load("svm_model.pkl")
lof = joblib.load("lof_model.pkl")
print("Models loaded!")

# ================== FIREWALL MEMORY ==================
blocked_ips = set()      # IP bị block
ip_attack_count = {}     # Số lần bị phát hiện ANOMALY
BLOCK_THRESHOLD = 3      # Sau 3 lần mới block

# ================== DATA MODEL ==================
class UserData(BaseModel):
    requests_per_min: float
    session_duration: float
    failed_login: int

# ================== FEATURE ENGINEERING ==================
def feature_engineering(data):
    fail_ratio = data.failed_login / (data.requests_per_min + 1)
    activity_score = data.requests_per_min / (data.session_duration + 1)

    return pd.DataFrame([{
        "requests_per_min": data.requests_per_min,
        "session_duration": data.session_duration,
        "failed_login": data.failed_login,
        "fail_ratio": fail_ratio,
        "activity_score": activity_score
    }])

# ================== LOG SYSTEM ==================
def log_anomaly(ip, data, score):
    file = "anomaly_log.csv"

    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": ip,
        "requests_per_min": data.requests_per_min,
        "session_duration": data.session_duration,
        "failed_login": data.failed_login,
        "score": score
    }])

    if not os.path.exists(file):
        row.to_csv(file, index=False)
    else:
        row.to_csv(file, mode="a", header=False, index=False)

# ================== ROOT ==================
@app.get("/")
def root():
    return {"status": "AI Anomaly Detection System Running"}

# ================== PREDICT ==================
@app.post("/predict")
def predict(data: UserData, request: Request):
    client_ip = request.client.host

    # 🚫 Nếu đã bị block
    if client_ip in blocked_ips:
        raise HTTPException(status_code=403, detail="Blocked by AI Defense")

    X = feature_engineering(data)

    iso_pred = 1 if iso.predict(X)[0] == -1 else 0
    svm_pred = 1 if svm.predict(X)[0] == -1 else 0
    lof_pred = 1 if lof.predict(X)[0] == -1 else 0

    score = iso_pred + svm_pred + lof_pred
    result = "ANOMALY" if score >= 2 else "NORMAL"

    # 🚨 Nếu AI phát hiện bất thường
    if result == "ANOMALY":
        log_anomaly(client_ip, data, score)

        ip_attack_count[client_ip] = ip_attack_count.get(client_ip, 0) + 1

        # Block sau nhiều lần
        if ip_attack_count[client_ip] >= BLOCK_THRESHOLD:
            blocked_ips.add(client_ip)

    return {
        "IP": client_ip,
        "IsolationForest": iso_pred,
        "OneClassSVM": svm_pred,
        "LOF": lof_pred,
        "AnomalyScore(0-3)": score,
        "AttackCount": ip_attack_count.get(client_ip, 0),
        "Blocked": client_ip in blocked_ips,
        "FinalResult": result
    }

# ================== VIEW LOGS ==================
@app.get("/logs")
def get_logs():
    file = "anomaly_log.csv"
    if not os.path.exists(file):
        return []

    try:
        df = pd.read_csv(file)
        return df.to_dict(orient="records")
    except:
        return []

# ================== VIEW BLOCKED IPS ==================
@app.get("/blocked")
def get_blocked_ips():
    return {"blocked_ips": list(blocked_ips)}

# ================== UNBLOCK ALL ==================
@app.post("/unblock_all")
def unblock_all():
    blocked_ips.clear()
    ip_attack_count.clear()
    return {"message": "All IPs unblocked"}

# ================== STATS ==================
@app.get("/stats")
def stats():
    file = "anomaly_log.csv"
    if not os.path.exists(file):
        return {"attacks": 0}

    df = pd.read_csv(file)
    return {"attacks": len(df)}
