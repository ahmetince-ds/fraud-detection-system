import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests

# =========================
# INIT
# =========================
st.title("💳 Fraud Detection Dashboard")

LOG_FILE = "logs.json"

# =========================
# LOAD LOGS (SAFE)
# =========================
logs = []

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except:
                continue

df = pd.DataFrame(logs)

# =========================
# EMPTY STATE
# =========================
if df.empty:
    st.warning("Henüz log yok")
    st.stop()

# =========================
# BASIC INFO
# =========================
st.write("### 📊 Genel Bilgi")
st.write("Total Requests:", len(df))
st.dataframe(df.tail())

# =========================
# THRESHOLD CONTROL
# =========================
st.write("### ⚙️ Risk Threshold")

new_threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

if st.button("Güncelle"):
    requests.post(
        "http://127.0.0.1:8000/set_threshold",
        params={"threshold": new_threshold}
    )
    st.success(f"Yeni threshold: {new_threshold}")

# =========================
# SAFE PARSING
# =========================
def safe_prob(x):
    try:
        return x.get("probability", None)
    except:
        return None

def safe_pred(x):
    try:
        return x.get("prediction", None)
    except:
        return None

df["prob"] = df["result"].apply(safe_prob)
df["prediction"] = df["result"].apply(safe_pred)

df = df.dropna(subset=["prob"])

# =========================
# CHART 1
# =========================
st.subheader("📈 Fraud Probability Trend")
st.line_chart(df["prob"])

# =========================
# CHART 2
# =========================
st.subheader("📊 Distribution")

fig, ax = plt.subplots()
ax.hist(df["prob"], bins=20)
st.pyplot(fig)

# =========================
# STATS
# =========================
st.subheader("📌 Stats")

st.write("Avg Probability:", df["prob"].mean())
st.write("Max Probability:", df["prob"].max())
st.write("Fraud Count:", df["prediction"].sum())

# =========================
# SHAP (LAST LOG)
# =========================
st.write("### 🧠 Son İşlem Açıklaması (SHAP)")

last = logs[-1] if logs else None

if last and "result" in last and "explanation" in last["result"]:
    explanation = last["result"]["explanation"]

    exp_df = pd.DataFrame({
        "feature": list(explanation.keys()),
        "impact": list(explanation.values())
    }).sort_values(by="impact", ascending=False)

    st.bar_chart(exp_df.set_index("feature"))
else:
    st.info("SHAP explanation yok")


# =========================
# LAST TRANSACTION STATUS
# =========================
st.write("### 🚨 Son İşlem Durumu")

last = logs[-1]

if "result" in last and "prediction" in last["result"]:
    if last["result"]["prediction"] == 1:
        st.error("⚠️ FRAUD DETECTED!")
        st.metric("Risk", "HIGH")
    else:
        st.success("✅ Normal işlem")
        st.metric("Risk", "LOW")
else:
    st.warning("Prediction yok")

# =========================
# PROBABILITY METRIC
# =========================
if "result" in last and "probability" in last["result"]:
    st.metric("Fraud Probability", round(last["result"]["probability"], 4))