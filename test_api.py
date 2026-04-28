import requests

res = requests.post("http://127.0.0.1:8000/predict", json={
 "age": 30,
 "transaction_amount": 1000,
 "transaction_hour": 2,
 "num_transactions_today": 1,
 "is_foreign_transaction": 0,
 "merchant_risk_score": 5,
 "merchant_distance_km": 10,
 "prev_fraud_flag": 0,
 "account_balance": 5000,
 "amount_to_balance_ratio": 0.3,
 "risk_distance_interaction": 50,
 "high_risk_flag": 0,
 "night_transaction": 1
})

print(res.status_code)
print(res.json())