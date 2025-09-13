# save as 3_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
import shap
import plotly.express as px

# Load model & encoders
model = joblib.load('salary_prediction_model.joblib')
le_dict = joblib.load('label_encoders.joblib')

st.set_page_config(page_title="💼 Salary Predictor & Career Advisor", layout="wide")
st.title("💼 IT Job Transition Salary Predictor & Career Advisor")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Your Details:")
    current_company = st.selectbox("Current Company", le_dict['current_company'].classes_)
    target_company = st.selectbox("Target Company", le_dict['target_company'].classes_)
    years_of_experience = st.slider("Years of Experience", 0.0, 25.0, step=0.5)
    current_salary = st.number_input("Current Salary (Annual INR)", min_value=100000)
    gender = st.selectbox("Gender", le_dict['gender'].classes_)
    location = st.selectbox("Location", le_dict['location'].classes_)
    current_role = st.selectbox("Current Role", le_dict['current_role'].classes_)
    education = st.selectbox("Education Level", le_dict['education'].classes_)
    certifications = st.multiselect("Certifications", le_dict['certifications'].classes_)
    skills = st.multiselect("Skills", le_dict['skills'].classes_)
    salary_goal = st.number_input("🎯 Desired Salary Goal (INR)", value=2000000)

    submitted = st.form_submit_button("Predict Expected Salary")

if submitted:
    # Handle multi-selects (take 1st selected or 'None')
    cert_input = certifications[0] if certifications else 'None'
    skill_input = skills[0] if skills else 'None'

    # Build input DataFrame
    input_df = pd.DataFrame([{
        'current_company': current_company,
        'target_company': target_company,
        'years_of_experience': years_of_experience,
        'current_salary': current_salary,
        'gender': gender,
        'location': location,
        'current_role': current_role,
        'sector': 'IT',
        'education': education,
        'certifications': cert_input,
        'skills': skill_input,
        'current_company_tier':  le_dict['current_company'].transform([current_company])[0] % 5 + 1,  # simple tier mock
        'target_company_tier':  le_dict['target_company'].transform([target_company])[0] % 5 + 1
    }])

    # Encode
    for col in le_dict:
        input_df[col] = le_dict[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    hike_percent = ((prediction - current_salary) / current_salary) * 100

    st.success(f"💼 **Predicted Salary:** ₹{int(prediction):,}")
    st.info(f"📈 **Expected Hike:** {hike_percent:.1f}%")

    # Dynamic Hike Recommendation
    exp_factor = 45000  # rough growth per year
    needed_years = max(0, (salary_goal - current_salary) / exp_factor)
    st.subheader("🛠 Dynamic Hike Recommendation")
    st.write(f"💡 To reach ₹{int(salary_goal):,}, you may need ~{needed_years:.1f} years of experience in your current company **OR** switch to a Tier-1 company to reach this target faster.")

    # Mocked Job Market Trends
    st.subheader("🌐 Job Market Trends (Mocked Data)")
    role_demand = {
        'Bangalore': np.random.randint(800, 1500),
        'Pune': np.random.randint(400, 1000),
        'Hyderabad': np.random.randint(500, 1200),
    }
    for city, jobs in role_demand.items():
        status = "🔥 High demand" if jobs > 1000 else "Moderate demand"
        st.write(f"- 📍 {city}: {jobs}+ open jobs ({status})")

    # Career Path Suggestion
    st.subheader("🚀 Career Path Suggestion")
    career_paths = {
        'Software Developer': ['Senior Developer', 'Tech Lead', 'Solution Architect'],
        'Senior Developer': ['Tech Lead', 'Solution Architect'],
        'QA Engineer': ['Senior QA Engineer', 'QA Lead', 'QA Manager'],
        'Data Scientist': ['Senior Data Scientist', 'Lead Data Scientist', 'AI Architect'],
        'DevOps Engineer': ['Senior DevOps', 'DevOps Lead', 'Cloud Architect']
    }
    path = career_paths.get(current_role, ['Tech Lead', 'Manager', 'Director'])
    st.markdown(f"**Your optimal next move:**\n- Current: {current_role}\n- ➔ Next: {path[0]}\n- ➔ Future: {path[1]} *(avg salary bump: +65%)*")

    # Salary Negotiation Tips
    st.subheader("💬 Salary Negotiation Tips")
    st.write(f"At {target_company}, with {years_of_experience} years exp:")
    tips = f"- Highlight your **{cert_input} certification + {skill_input} skills**.\n" \
           "- Back it up with measurable achievements (projects, cost-savings).\n" \
           "- Aim for at least a **25–30% hike.**"
    st.markdown(tips)

    # Bonus: Feature Importance (load existing SHAP plot)
    st.subheader("🔍 Feature Importance (from training)")
    st.image('shap_feature_importance.png')
