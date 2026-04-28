# 💳 Fraud Detection System

## 🚀 Proje Açıklaması
Bu proje, kredi kartı işlemlerinde dolandırıcılığı tespit etmek için geliştirilmiş uçtan uca bir makine öğrenmesi sistemidir.

## 🧠 Kullanılan Teknolojiler
- Python
- XGBoost
- FastAPI
- Streamlit
- SHAP

## ⚙️ Sistem Mimarisi
1. Model eğitimi (train_model.py)
2. API servisi (fraud_api.py)
3. Logging sistemi (logs.json)
4. Dashboard (dashboard.py)

## 📊 Özellikler
- Fraud tahmini (probability)
- SHAP explainability (neden fraud dedi)
- Threshold ayarı (dinamik risk yönetimi)
- Gerçek zamanlı dashboard

## ▶️ Çalıştırma

```bash
python train_model.py
uvicorn fraud_api:app --reload
streamlit run dashboard.py
```

## 📌 Not
Dashboard verileri logs.json üzerinden canlı olarak okunur.