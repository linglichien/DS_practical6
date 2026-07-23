import streamlit as st
import numpy as np
import joblib

# ---- Load saved artifacts (must be in the same folder as this script) ----
rf = joblib.load("rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="California Housing Price Predictor", layout="centered")

st.title("California Housing Price Predictor")
st.write("Deploying a Random Forest model trained on California housing data.")

# ---- Build one numeric input per feature ----
st.subheader("Enter feature values")
user_inputs = []
for col in feature_columns:
    val = st.number_input(col, value=0.0, format="%.4f")
    user_inputs.append(val)

# ---- Predict ----
# No scaling needed - Random Forest is invariant to feature scale
if st.button("Predict Price"):
    raw_input = np.array([user_inputs])
    prediction = rf.predict(raw_input)
    st.success(f"Predicted Value: ${prediction[0] * 100000:,.2f}")

    # Optional: show feature importance alongside the prediction
    with st.expander("See what drives this model's predictions"):
        import pandas as pd
        importances = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=False)
        st.bar_chart(importances)
