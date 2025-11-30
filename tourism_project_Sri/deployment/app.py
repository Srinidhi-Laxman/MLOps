import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# -------------------------
# Configuration
# -------------------------
HF_MODEL_REPO = "Srinidhiiiiii/visit-with-us-wellness-xgb"
MODEL_FILENAME = "tourism_best_model_v1.joblib"

# -------------------------
# Download & load model
# -------------------------
model = None
try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, repo_type="model", token=os.getenv("HF_TOKEN"))
    model = joblib.load(model_path)
    st.write(f"Loaded model from Hugging Face: {HF_MODEL_REPO}/{MODEL_FILENAME}")
except Exception as e:
    st.warning(f"Could not download model from Hugging Face ({HF_MODEL_REPO}).\nError: {e}\nFalling back to local file if present.")
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        st.write(f"Loaded local model file: {MODEL_FILENAME}")
    else:
        st.error("Model not available. Please upload the model to HF or place it locally.")
        st.stop()

# -------------------------
# Streamlit UI
# -------------------------
st.title("Visit with Us — Wellness Package Purchase Predictor")
st.write(
    """
    Enter customer details and interaction info below.  
    The model will predict whether the customer is likely to purchase the *Wellness Tourism Package*.
    """
)

# --- Customer details
st.header("Customer Details")
age = st.number_input("Age", min_value=16, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["1", "2", "3"])
occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Freelancer", "Business", "Student", "Retired", "Other"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed", "Other"])
num_trips = st.number_input("Number of Trips per Year (Avg.)", min_value=0, max_value=50, value=2)
passport = st.selectbox("Has Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns Car?", ["No", "Yes"])
num_children = st.number_input("Number of Children (below 5)", min_value=0, max_value=10, value=0)
designation = st.text_input("Designation (optional)", value="")  # free text
monthly_income = st.number_input("Monthly Income (gross)", min_value=0.0, max_value=1000000.0, value=30000.0, step=100.0)

# --- Interaction details
st.header("Sales Interaction Details")
pitch_score = st.slider("Pitch Satisfaction Score (1-10)", min_value=0, max_value=10, value=7)
product_pitched = st.selectbox("Product Pitched", ["Wellness", "Adventure", "Family", "Romantic", "Business", "Other"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=50, value=1)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=300.0, value=10.0, step=0.5)

# Assemble input into DataFrame matching training columns (raw — pipeline should handle preprocessing)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": int(city_tier),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

st.subheader("Input Preview")
st.dataframe(input_df.T, width=700)

# Prediction
if st.button("Predict Purchase Probability"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[:, 1][0]
            pred = int(proba >= 0.5)
        else:
            pred = int(model.predict(input_df)[0])
            proba = None

        label = "Will Purchase Wellness Package" if pred == 1 else "Will NOT Purchase Wellness Package"
        st.subheader("Prediction")
        st.success(label)

        if proba is not None:
            st.write(f"Predicted probability of purchase: **{proba:.3f}**")
            st.info("Tip: adjust the classification threshold depending on business priorities (precision vs recall).")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
