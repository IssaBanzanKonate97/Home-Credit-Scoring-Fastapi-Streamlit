import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

CSV = "application_train.csv"
FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "AMT_CREDIT"]
TARGET = "TARGET"

df = pd.read_csv(CSV)
df = df[FEATURES + [TARGET]].copy()
df = df.dropna()
X = df[FEATURES]
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"AUC = {auc:.3f}")

import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/logistic.pkl")
print("Modèle sauvegardé -> models/logistic.pkl")
