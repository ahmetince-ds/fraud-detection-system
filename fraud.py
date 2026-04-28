''' import pandas as pd
import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV
import joblib
from fastapi import FastAPI
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,auc
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# =========================
# 1. DATA LOAD
# =========================
df = pd.read_csv('C:\\Users\\Win10\\Desktop\\credit\\fraud.csv')


# =========================
# 2. CLEANING
# =========================
df['transaction_hour'] = df['transaction_hour'].fillna(df['transaction_hour'].median())
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)


# =========================
# 3. FEATURE ENGINEERING
# =========================
df["amount_to_balance_ratio"] = df["transaction_amount"] / (df["account_balance"] + 1)

df["risk_distance_interaction"] = df["merchant_risk_score"] * df["merchant_distance_km"]

df["high_risk_flag"] = (df["merchant_risk_score"] > 7).astype(int)

df["night_transaction"] = df["transaction_hour"].apply(lambda x: 1 if 0 <= x <= 6 else 0)


# =========================
# 4. FEATURES / TARGET
# =========================
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("FEATURE SHAPE:", X.shape)


# =========================
# 5. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# 6. LOGISTIC REGRESSION (PIPELINE)
# =========================
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=5000, solver='liblinear'))
])

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_proba = log_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC =====")
print(classification_report(y_test, log_pred))
print("LOGISTIC ROC-AUC:", roc_auc_score(y_test, log_proba))


# =========================
# 7. RANDOM FOREST
# =========================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST =====")
print(classification_report(y_test, rf_pred))
print("RF ROC-AUC:", roc_auc_score(y_test, rf_proba))


# =========================
# 8. XGBOOST (BASE MODEL)
# =========================
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("\n===== XGBOOST =====")
print(classification_report(y_test, xgb_pred))
print("XGB ROC-AUC:", roc_auc_score(y_test, xgb_proba))


# =========================
# 9. CROSS VALIDATION
# =========================
cv_scores = cross_val_score(log_model, X, y, cv=5, scoring='roc_auc')

print("\n===== CROSS VALIDATION =====")
print("LOGISTIC CV ROC-AUC:", cv_scores.mean())
print("STD:", cv_scores.std())


# =========================
# 10. THRESHOLD TUNING
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, log_proba)

f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

best_idx = np.nanargmax(f1)
best_threshold = thresholds[best_idx]

print("\nBEST THRESHOLD:", best_threshold)

custom_pred = (log_proba >= best_threshold).astype(int)

print("\n===== THRESHOLD MODEL =====")
print(classification_report(y_test, custom_pred))


# =========================
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, custom_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix (Tuned Threshold)")
plt.show()


# =========================
# 12. GRIDSEARCH (XGBOOST TUNING)
# =========================
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 300],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

best_xgb_pred = best_xgb.predict(X_test)
best_xgb_proba = best_xgb.predict_proba(X_test)[:, 1]

print("\n===== BEST XGBOOST =====")
print(classification_report(y_test, best_xgb_pred))
print("BEST XGB ROC-AUC:", roc_auc_score(y_test, best_xgb_proba))

print("\nBEST PARAMS:", grid.best_params_)


# =========================
# 13. FEATURE IMPORTANCE
# =========================
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": best_xgb.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTOP FEATURES:")
print(fi.head(10))


# =========================
# 14. FINAL INSIGHT BLOCK
# =========================
print("\n===== ANALYSIS =====")
print("ROC-AUC improved models compared.")
print("Recall is critical for fraud detection.")
print("Feature engineering impacted model behavior.")
print("Best model: XGBoost (after tuning).")

# ===================================
# EKSTRA PRO MODEL
# 1 SHAP EXPLAINABILITY (model neden karar verdi)

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# global importance
shap.summary_plot(shap_values,X_test)
# tek bir sample açıklama
shap.force_plot(explainer.expected_value,shap_values[0],X_test.iloc[0])

# ========================================================
# 2 PR-AUC CURVE (fraud için en önemli metrik)
probs = best_xgb.predict_proba(X_test)[:,1]
precision,recall,_ = precision_recall_curve(y_test,probs)
pr_auc = auc(recall,precision)
print('PR-AUC:', pr_auc)
plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ========================================
# 3 MODEL CALİBRATİON (PLATT SCALİNG)
calibrated_model = CalibratedClassifierCV(best_xgb,method='sigmoid',cv=3)
calibrated_model.fit(X_train,y_train)
cal_probs = calibrated_model.predict_proba(X_test)[:,1]
print('Calibrated ROC-AUC:', roc_auc_score(y_test,cal_probs))

# ====================================

# 4 İMBALANCE HANDLİNG (scale_pos_weight)
neg = sum(y_train == 0)
pos = sum(y_train == 1)

scale = neg/pos

xgb_balanced = XGBClassifier(n_estimators=300,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=scale,
                            eval_metric='logloss')

xgb_balanced.fit(X_train,y_train)
bal_pred =xgb_balanced.predict(X_test)
print(classification_report(y_test,bal_pred))

# ===============================================7
# 5 - FASTAPI DEPLOYMENT ( MODEL API YAPMA )
from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# =========================
# MODELİ YÜKLE
# =========================
model = joblib.load("fraud_model.pkl")

# =========================
# TEST ENDPOINT
# =========================
@app.get("/")
def home():
    return {"message": "Fraud API çalışıyor"}

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: dict):
    try:
        # gelen JSON -> feature array
        features = np.array(list(data.values())).reshape(1, -1)

        # prediction probability
        prob = model.predict_proba(features)[0][1]

        return {
            "fraud_probability": float(prob),
            "prediction": int(prob > 0.5)
        }

    except Exception as e:
        return {"error": str(e)}

# MODELİ KAYDETME
joblib.dump(best_xgb, "fraud_model.pkl")

''' # BURASI GÜNCELLENDİ BEN TEK DOSYADA YAPMIŞTIM TÜM SİSTEMİ BU BİRAZ SORUN ÇIKARTMA İHTİMALİ YÜKSEK 