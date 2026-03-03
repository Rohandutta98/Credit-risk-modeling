# ================================
# CREDIT RISK MODEL - FINAL VERSION
# ================================

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# ---------------------------
# 1. LOAD DATA
# ---------------------------

df = pd.read_csv("data/loan_data.csv")

# Reduce dataset size for faster training
df = df.sample(50000, random_state=42)

print("Dataset Loaded Successfully")
print("Total Rows:", len(df))

# ---------------------------
# 2. SELECT IMPORTANT NUMERIC FEATURES
# ---------------------------

features = [
    'loan_amnt',
    'int_rate',
    'annual_inc',
    'dti',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'total_acc',
    'mort_acc',
    'pub_rec_bankruptcies',
    'loan_status'
]

df = df[features]

# ---------------------------
# 3. FILTER TARGET
# ---------------------------

df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})

df = df.dropna()

print("After Cleaning Rows:", len(df))

# ---------------------------
# 4. SPLIT FEATURES & TARGET
# ---------------------------

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 5. SCALE DATA
# ---------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)
# ---------------------------
# 6. HANDLE IMBALANCE USING SMOTE
# ---------------------------

print("Before SMOTE:", np.bincount(y_train))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE:", np.bincount(y_resampled))

# ---------------------------
# 7. TRAIN RANDOM FOREST
# ---------------------------

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_resampled, y_resampled)

# ---------------------------
# 8. EVALUATE MODEL
# ---------------------------

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ---------------------------
# 9. SHAP EXPLAINABILITY
# ---------------------------

# ---------------------------
# 9. SHAP EXPLAINABILITY (Optimized Version)
# ---------------------------

print("Generating SHAP Summary Plot...")

sample_for_shap = X_resampled[:500]

explainer = shap.TreeExplainer(model)

shap_values = explainer(sample_for_shap)

shap.summary_plot(shap_values.values, sample_for_shap)

# ---------------------------
# 10. SAVE MODEL
# ---------------------------

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")