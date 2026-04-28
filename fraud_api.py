from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
from datetime import datetime
import shap
import os

# =========================
# CONFIG
# =========================
THRESHOLD = 0.5
app = FastAPI()

# =========================
# MODEL LOAD
# =========================
data = joblib.load("fraud_model.pkl")
model = data["model"]
feature_order = data["features"]

# =========================
# SHAP INIT (GLOBAL)
# =========================
explainer = shap.TreeExplainer(model)

# =========================
# FEATURE ENGINEERING
# =========================
def build_features(x):
    x["amount_to_balance_ratio"] = x["transaction_amount"] / (x["account_balance"] + 1)
    x["risk_distance_interaction"] = x["merchant_risk_score"] * x["merchant_distance_km"]
    x["high_risk_flag"] = 1 if x["merchant_risk_score"] > 7 else 0
    x["night_transaction"] = 1 if 0 <= x["transaction_hour"] <= 6 else 0
    return x

# =========================
# LOGGING
# =========================
def log_request(data, result):

    log = {
        "log_id": str(datetime.now().timestamp()),
        "time": str(datetime.now()),
        "input": data,
        "result": result
    }

    print("LOG SAVED")

    if not os.path.exists("logs.json"):
        open("logs.json", "w").close()

    with open("logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")

# =========================
# INPUT SCHEMA
# =========================
class FraudInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    transaction_amount: float
    transaction_hour: float
    num_transactions_today: float
    is_foreign_transaction: float
    merchant_risk_score: float = Field(..., ge=0, le=10)
    merchant_distance_km: float
    prev_fraud_flag: float
    account_balance: float

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "API çalışıyor"}

# =========================
# PREDICT (MAIN)
# =========================
@app.post("/predict")
def predict(data: FraudInput):

    try:
        input_dict = data.model_dump()

        # feature engineering
        full = build_features(input_dict)

        features = [full[col] for col in feature_order]
        arr = np.array(features).reshape(1, -1)

        prob = model.predict_proba(arr)[0][1]
        prob = float(np.clip(prob, 0, 1))

        prediction = int(prob > THRESHOLD)

        log_request(input_dict, {
            "probability": prob,
            "prediction": prediction
        })

        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": prediction,
            "risk_level": "HIGH" if prob > THRESHOLD else "LOW",
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# SHAP EXPLAIN
# =========================
@app.post("/explain")
def explain(data: FraudInput):

    input_dict = data.model_dump()
    full = build_features(input_dict)

    features = [full[col] for col in feature_order]
    arr = np.array(features).reshape(1, -1)

    shap_values = explainer.shap_values(arr)

    explanation = dict(zip(feature_order, shap_values[0]))

    log_request(input_dict, {
        "type": "explain",
        "explanation": explanation
    })

    return {"explanation": explanation}

# =========================
# THRESHOLD CONTROL
# =========================
@app.post("/set_threshold")
def set_threshold(threshold: float):

    global THRESHOLD
    THRESHOLD = threshold

    return {"new_threshold": THRESHOLD}