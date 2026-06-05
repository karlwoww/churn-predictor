import pandas as pd
import numpy as np
import joblib
import shap
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
threshold_path = os.path.join(BASE_DIR, "models", "threshold.txt")

model = joblib.load(model_path)

with open(threshold_path, "r") as f:
    THRESHOLD = float(f.read())

print(f"Модель загружена. Порог: {THRESHOLD}")

catboost_model = model.named_steps['classifier']
explainer = shap.TreeExplainer(catboost_model)

preprocessor = model.named_steps['preprocessor']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = preprocessor.named_transformers_['cat'].feature_names_in_
cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_names)

class ClientData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI(
    title="Churn Prediction API",
    description="Сервис прогнозирования оттока клиентов телеком-оператора",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "threshold": THRESHOLD}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(client: ClientData):
    try:
        client_df = pd.DataFrame([client.model_dump()])
        probability = model.predict_proba(client_df)[0, 1]
        churn_risk = probability >= THRESHOLD

        return {
            "probability": round(float(probability), 4),
            "churn_risk": bool(churn_risk),
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(client: ClientData):
    try:
        client_df = pd.DataFrame([client.model_dump()])
        X_transformed = model.named_steps['preprocessor'].transform(client_df)
        shap_values = explainer.shap_values(X_transformed)[0]
        base_value = explainer.expected_value

        contributions = []
        for name, val in zip(all_feature_names, shap_values):
            contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "direction": "increase_risk" if val > 0 else "decrease_risk"
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(float(base_value), 4),
            "top_contributors": contributions[:5]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))