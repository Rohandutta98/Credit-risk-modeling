# 💳 Credit Risk Prediction App

An end-to-end Machine Learning project to predict loan default risk using financial applicant data.

This project includes:
- Data preprocessing
- Handling imbalanced data using SMOTE
- XGBoost model training
- SHAP explainability
- Interactive Streamlit web application
- Online deployment

---

## 📌 Problem Statement

Financial institutions face significant losses due to loan defaults.  
The objective of this project is to predict whether a loan applicant is likely to default based on financial and credit attributes.

---

## 📊 Dataset

- LendingClub Loan Dataset
- Cleaned and preprocessed financial records
- Target variable: `loan_status`
  - 0 → Fully Paid (Low Risk)
  - 1 → Charged Off (High Risk)

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SMOTE (Imbalanced-learn)
- SHAP (Explainable AI)
- Streamlit

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Removed irrelevant features
- Handled missing values
- Feature scaling using StandardScaler

### 2️⃣ Handling Class Imbalance
- Applied SMOTE to balance minority class

### 3️⃣ Model Training
- XGBoost Classifier
- Hyperparameter tuning
- Evaluation using ROC-AUC

### 4️⃣ Model Explainability
- Integrated SHAP values
- Feature contribution displayed inside web app

---

## 📈 Model Performance

- ROC-AUC Score: ~0.75+
- Balanced classification performance
- Adjustable risk threshold in UI

---

## 🚀 Web Application Features

- User-friendly interface
- Adjustable risk threshold slider
- Real-time default probability prediction
- SHAP-based feature contribution table

---

## 🖥️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-modeling.git
cd credit-risk-modeling