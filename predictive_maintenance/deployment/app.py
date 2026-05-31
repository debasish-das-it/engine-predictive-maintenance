
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="debasishdas1985/engine-predictive-maintenance-model",
    filename="engine-predictive-maintenance-model.joblib",
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write(
    "This app predicts whether an engine is likely to require maintenance "
    "based on six real-time sensor readings. It is intended as an internal "
    "decision-support tool for reliability and maintenance engineers."
)
st.write("Kindly enter the latest engine sensor readings below to assess its condition.")

# Layout — two columns for sensor groups
col1, col2 = st.columns(2)

with col1:
    st.subheader("Engine & Lubrication")
    engine_rpm = st.number_input(
        "Engine RPM (revolutions per minute)",
        min_value=0, max_value=3000, value=700,
    )
    lub_oil_pressure = st.number_input(
        "Lubricating Oil Pressure (bar)",
        min_value=0.0, max_value=10.0, value=2.5, step=0.1,
    )
    lub_oil_temp = st.number_input(
        "Lubricating Oil Temperature (°C)",
        min_value=0.0, max_value=150.0, value=84.1, step=0.1,
    )

with col2:
    st.subheader("Fuel & Coolant")
    fuel_pressure = st.number_input(
        "Fuel Pressure (bar)",
        min_value=0.0, max_value=30.0, value=11.8, step=0.1,
    )
    coolant_pressure = st.number_input(
        "Coolant Pressure (bar)",
        min_value=0.0, max_value=10.0, value=3.2, step=0.1,
    )
    coolant_temp = st.number_input(
        "Coolant Temperature (°C)",
        min_value=0.0, max_value=150.0, value=81.6, step=0.1,
    )

# Build the input frame in the exact feature order used during training
input_data = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp,
}])

# Classification threshold (tuned for recall on the maintenance-required class)
classification_threshold = 0.50

# Predict button
if st.button("Predict Engine Condition", use_container_width=True):
    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = int(prediction_proba >= classification_threshold)

        st.divider()
        if prediction == 1:
            st.error("Expected Outcome: Engine is LIKELY to REQUIRE MAINTENANCE")
            st.metric("Maintenance Risk", f"{prediction_proba*100:.2f}%")
        else:
            st.success("Expected Outcome: Engine appears to be in HEALTHY condition")
            st.metric("Maintenance Risk", f"{prediction_proba*100:.2f}%")
        st.divider()

        with st.expander("View input sent to the model"):
            st.dataframe(input_data, use_container_width=True)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all sensor readings are filled correctly.")

# Footer
st.markdown("---")
st.caption(
    "Powered by Engine Predictive Maintenance MLOps Pipeline | "
    "XGBoost Model v1.0 | Confidence Threshold: 50%"
)
