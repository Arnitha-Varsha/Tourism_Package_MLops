import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Title and description
st.set_page_config(page_title="Tourism Package Purchase Predictor", layout="centered")
st.title("f334 Tourism Package Purchase Predictor")
st.write("Predict whether a customer is likely to purchase the newly introduced Wellness Tourism Package based on customer profile and interaction data.")

# Load the trained model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("f680 Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# User Inputs organized in categories
st.subheader("f9d1 Customer Demographics")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    # FIXED: Set max_value to 500,000 so the default value of 250,000 is safely within limits
    monthly_income = st.number_input("Monthly Income (INR)", min_value=0, max_value=500000, value=250000, step=1000)

st.subheader("✈f3fe Travel Preferences")
col3, col4 = st.columns(2)
with col3:
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting (< 5yo)", min_value=0, max_value=5, value=0)
    num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=2)
with col4:
    property_star = st.slider("Preferred Property Star Rating", min_value=1, max_value=5, value=3)
    passport = st.selectbox("Has Valid Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

st.subheader("☎f3fe Sales & Marketing Interactions")
col5, col6 = st.columns(2)
with col5:
    typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
with col6:
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)

# Preprocess custom text outputs back to numerical features
passport_val = 1 if passport == "Yes" else 0
own_car_val = 1 if own_car == "Yes" else 0

# Map inputs to match features used in training exactly
input_data = {
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport_val,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car_val,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}

# Predict Button
if st.button("f52e Predict Purchase Decision", use_container_width=True):
    features_df = pd.DataFrame([input_data])

    # FIXED: Reorder columns to align exactly with model training
    try:
        training_cols = model.feature_names_in_
        features_df = features_df[training_cols]
    except AttributeError:
        pass

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    st.markdown("--- ")
    if prediction == 1:
        st.success(f"f389 **High Likelihood!** This customer is highly likely to purchase the Wellness Tourism Package (Probability: {probability * 100:.2f}%).")
    else:
        st.info(f"❌ **Unlikely.** This customer is less likely to purchase the package (Probability of purchase: {probability * 100:.2f}%).")
