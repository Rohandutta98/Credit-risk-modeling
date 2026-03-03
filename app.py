import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# -------------------------------
# Load Model & Scaler
# -------------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Risk App", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.write("Predict whether a loan applicant is High Risk or Low Risk")

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("Applicant Information")

loan_amnt = st.sidebar.number_input("Loan Amount", 1000, 500000, 15000)
int_rate = st.sidebar.number_input("Interest Rate (%)", 5.0, 35.0, 12.0)
annual_inc = st.sidebar.number_input("Annual Income", 10000, 500000, 60000)
dti = st.sidebar.number_input("Debt-to-Income Ratio", 0.0, 50.0, 15.0)
open_acc = st.sidebar.number_input("Open Accounts", 0, 50, 5)
pub_rec = st.sidebar.number_input("Public Records", 0, 10, 0)
revol_bal = st.sidebar.number_input("Revolving Balance", 0, 100000, 10000)
total_acc = st.sidebar.number_input("Total Accounts", 1, 100, 20)
mort_acc = st.sidebar.number_input("Mortgage Accounts", 0, 10, 1)
pub_rec_bankruptcies = st.sidebar.number_input("Bankruptcies", 0, 5, 0)

# -------------------------------
# Adjustable Threshold Slider
# -------------------------------

threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# -------------------------------
# Prediction Button
# -------------------------------

if st.sidebar.button("Predict Risk"):

    # Create DataFrame with correct feature names
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "total_acc": total_acc,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Apply adjustable threshold
    if probability > threshold:
        st.error("⚠ High Risk Applicant (Likely Default)")
    else:
        st.success("✅ Low Risk Applicant")

    # Show probability
    st.write(f"Default Probability: {round(probability*100,2)}%")
    st.progress(float(probability))

    # -------------------------------
    # SHAP Explainability
    # -------------------------------

    st.subheader("🔍 Feature Contribution (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact on Risk": shap_values.values[0]
    }).sort_values(by="Impact on Risk", key=abs, ascending=False)

    st.dataframe(shap_df)

    st.info("Positive values increase default risk. Negative values reduce risk.")