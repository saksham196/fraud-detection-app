import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection App")
st.markdown("### Detect fraudulent transactions using Machine Learning")

st.divider()

# ===============================
# 🔹 Single Prediction Section
# ===============================
st.subheader("🔍 Single Transaction Check")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Transaction Time", min_value=0.0)

with col2:
    amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Check Transaction", use_container_width=True):

    v_features = [0]*28
    features = np.array([time] + v_features + [amount]).reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Normal Transaction (Confidence: {1-prob:.2f})")

st.divider()

# ===============================
# 🔹 Batch Prediction Section
# ===============================
st.subheader("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Preview of Data:")
    st.dataframe(data.head())

    if st.button("Run Batch Prediction"):
        
        # Ensure correct format
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        data["Prediction"] = predictions
        data["Fraud_Probability"] = probabilities

        st.write("Results:")
        st.dataframe(data.head())

        # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )