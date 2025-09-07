# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
model = joblib.load("models/logistic.pkl")
app = FastAPI(title="Score", version="1.0.0")

class Features(BaseModel):
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: float
    AMT_CREDIT: float

@app.get("/ping")
def ping():
    return {"ping": "OK"}

@app.post("/predict")
def predict(feat: Features):
    x = np.array([[feat.EXT_SOURCE_1, feat.EXT_SOURCE_2, feat.EXT_SOURCE_3,
                   feat.DAYS_BIRTH, feat.AMT_CREDIT]])
    prob = float(model.predict_proba(x)[0, 1])
    return {"prob_default": prob}
