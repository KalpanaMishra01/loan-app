import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details below to calculate loan approval status.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married/Other"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education_level = st.selectbox("Education Level", ["Undergraduate (0)", "Graduate (1)", "Postgraduate (2)"], index=1)

with col2:
    st.subheader("💼 Employment & Income")
    employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_cat = st.selectbox("Employer Category", ["Private", "MNC", "Government", "Unemployed"])
    income = st.number_input("Applicant Income", min_value=0.0, value=50000.0, step=1000.0)
    co_income = st.number_input("Co-applicant Income", min_value=0.0, value=0.0, step=1000.0)
    savings = st.number_input("Total Savings Balance", min_value=0.0, value=10000.0, step=500.0)

with col3:
    st.subheader("📊 Financials & Loan")
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0)
    existing_loans = st.number_input("Number of Existing Loans", min_value=0, value=0)
    dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    loan_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=15000.0, step=1000.0)
    collateral = st.number_input("Collateral Value", min_value=0.0, value=20000.0, step=1000.0)
    loan_term = st.number_input("Loan Term (in months)", min_value=1, value=360)
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Personal", "Other"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

edu_map = {"Undergraduate (0)": 0, "Graduate (1)": 1, "Postgraduate (2)": 2}
edu_val = edu_map[education_level]

loan_purpose_car = 1.0 if loan_purpose == "Car" else 0.0
loan_purpose_edu = 1.0 if loan_purpose == "Education" else 0.0
loan_purpose_home = 1.0 if loan_purpose == "Home" else 0.0
loan_purpose_pers = 1.0 if loan_purpose == "Personal" else 0.0

marital_single = 1.0 if marital == "Single" else 0.0

emp_salaried = 1.0 if employment == "Salaried" else 0.0
emp_self = 1.0 if employment == "Self-employed" else 0.0
emp_unemp = 1.0 if employment == "Unemployed" else 0.0

prop_semi = 1.0 if property_area == "Semiurban" else 0.0
prop_urban = 1.0 if property_area == "Urban" else 0.0

gender_male = 1.0 if gender == "Male" else 0.0

emp_cat_gov = 1.0 if employer_cat == "Government" else 0.0
emp_cat_mnc = 1.0 if employer_cat == "MNC" else 0.0
emp_cat_pvt = 1.0 if employer_cat == "Private" else 0.0
emp_cat_unemp = 1.0 if employer_cat == "Unemployed" else 0.0

raw_features_df = pd.DataFrame([{
    'Applicant_Income': income,
    'Coapplicant_Income': co_income,
    'Age': age,
    'Dependents': dependents,
    'Credit_Score': credit_score,
    'Existing_Loans': existing_loans,
    'DTI_Ratio': dti_ratio,
    'Savings': savings,
    'Collateral_Value': collateral,
    'Loan_Amount': loan_amount,
    'Loan_Term': loan_term,
    'Education_Level': edu_val,
    'Loan_Purpose_Car': loan_purpose_car,
    'Loan_Purpose_Education': loan_purpose_edu,
    'Loan_Purpose_Home': loan_purpose_home,
    'Loan_Purpose_Personal': loan_purpose_pers,
    'Marital_Status_Single': marital_single,
    'Employment_Status_Salaried': emp_salaried,
    'Employment_Status_Self-employed': emp_self,
    'Employment_Status_Unemployed': emp_unemp,
    'Property_Area_Semiurban': prop_semi,
    'Property_Area_Urban': prop_urban,
    'Gender_Male': gender_male,
    'Employer_Category_Government': emp_cat_gov,
    'Employer_Category_MNC': emp_cat_mnc,
    'Employer_Category_Private': emp_cat_pvt,
    'Employer_Category_Unemployed': emp_cat_unemp
}])

# Force the columns to align with your exact training layout 
feature_order = [
    'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 'Credit_Score', 
    'Existing_Loans', 'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 
    'Loan_Term', 'Education_Level', 'Loan_Purpose_Car', 'Loan_Purpose_Education', 
    'Loan_Purpose_Home', 'Loan_Purpose_Personal', 'Marital_Status_Single', 
    'Employment_Status_Salaried', 'Employment_Status_Self-employed', 'Employment_Status_Unemployed', 
    'Property_Area_Semiurban', 'Property_Area_Urban', 'Gender_Male', 
    'Employer_Category_Government', 'Employer_Category_MNC', 'Employer_Category_Private', 
    'Employer_Category_Unemployed'
]
raw_features_df = raw_features_df[feature_order]

st.write("---")

if st.button("Run Eligibility Analysis", type="primary"):
    scaled_features = scaler.transform(raw_features_df)
    prediction = model.predict(scaled_features)
    
    if prediction == 1:
        st.success("🎉 **Application Approved!** The profile satisfies safety guidelines for loan assignment.")
    else:
        st.error("❌ **Application Rejected.** The input profile fails current algorithmic thresholds.")
