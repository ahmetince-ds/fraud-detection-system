import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import datetime
# DATA
df = pd.read_csv(r"C:\Users\Win10\Desktop\credit\fraud.csv")

# CLEANING
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# FEATURES
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# IMBALANCE
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# SAVE (PRODUCTION FORMAT)
joblib.dump({
    "model": model,
    "features": X.columns.tolist()
}, "fraud_model.pkl")

print("MODEL KAYDEDİLDİ")

# ===================================================
# MODEL VERSİON KONTROL

version =  datetime.datetime.now().strftime('%Y%m%d_%H%M')
joblib.dump({'model':model,
            'features':X.columns.tolist(),
            'version':version},f'fraud_model_{version}.pkl')

