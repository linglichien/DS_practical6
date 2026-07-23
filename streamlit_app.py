import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---- Page setup ----
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---- Load saved artifacts ----
rf = joblib.load("rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---- Header ----
st.title("🏠 California Housing Price Predictor")
st.caption("Random Forest model trained on California housing data")
st.divider()

# ---- Sidebar inputs ----
st.sidebar.header("Property Details")

longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.5, 0.01)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 36.5, 0.01)
housing_median_age = st.sidebar.slider("Housing Median Age (years)", 1, 52, 25)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=1, value=2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, value=400)
population = st.sidebar.number_input("Population", min_value=1, value=1000)
households = st.sidebar.number_input("Households", min_value=1, value=400)
median_income = st.sidebar.slider("Median Income ($10,000s)", 0.0, 15.0, 3.5, 0.1)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

predict_btn = st.sidebar.button("Predict Price", use_container_width=True, type="primary")

# ---- Build input row matching training columns exactly ----
def build_input_row():
    row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity_INLAND": 1 if ocean_proximity == "INLAND" else 0,
        "ocean_proximity_ISLAND": 1 if ocean_proximity == "ISLAND" else 0,
        "ocean_proximity_NEAR BAY": 1 if ocean_proximity == "NEAR BAY" else 0,
        "ocean_proximity_NEAR OCEAN": 1 if ocean_proximity == "NEAR OCEAN" else 0,
    }
    # Reorder to match exactly what the model was trained on
    return pd.DataFrame([row])[feature_columns]

# ---- Main area: map + prediction ----
col1, col2 = st.columns(2)
with col1:
    st.subheader("Location")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

with col2:
    st.subheader("Prediction")
    if predict_btn:
        input_df = build_input_row()
        prediction = rf.predict(input_df)[0] * 100000
        st.metric("Estimated Value", f"${prediction:,.0f}")
    else:
        st.info("Set property details in the sidebar, then click **Predict Price**.")

st.divider()

# ---- Feature importance (always visible, not just after prediction) ----
with st.expander("📊 What drives this model's predictions?"):
    importances = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=False)
    st.bar_chart(importances)
