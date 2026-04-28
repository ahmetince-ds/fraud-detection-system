import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# =========================
# DATA LOAD
# =========================
df = pd.read_csv("C:\\Users\\Win10\\Desktop\\credit\\fraud.csv")

df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# =========================
# LOAD TRAINED MODEL
# =========================
data = joblib.load("fraud_model.pkl")
model = data["model"]

# =========================
# SHAP EXPLAINABILITY
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP summary plot gösteriliyor...")

shap.summary_plot(shap_values, X)

# =========================
# PR-AUC ANALYSIS
# =========================
probs = model.predict_proba(X)[:, 1]

precision, recall, _ = precision_recall_curve(y, probs)
pr_auc = auc(recall, precision)

print("PR-AUC:", pr_auc)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importances.head(10))