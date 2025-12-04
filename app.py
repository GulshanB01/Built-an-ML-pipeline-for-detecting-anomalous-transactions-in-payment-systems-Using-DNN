import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model  # Keras model loader

# ---------- Load model + scaler + feature columns ----------
@st.cache_resource
def load_artifacts():
    model = load_model("trained_model/model.h5")
    scaler = joblib.load("trained_model/scaler.pkl")
    feature_columns = joblib.load("trained_model/feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# These are the numeric columns you scaled in training
NUM_COLS = ["amount", "balance_diff_org", "balance_diff_dest"]

# ---------- Streamlit UI ----------
st.title("🚨 Online Payment Anomaly / Fraud Detection")
st.write(
    "This app predicts whether an online transaction is likely "
    "**fraudulent** or **legit** based on the features you enter."
)

st.sidebar.header("🧾 Transaction Details")

# 1. Step (time step in dataset)
step = st.sidebar.number_input("Step (time step, e.g. 1–744)", min_value=1, value=1, step=1)

# 2. Transaction type
tx_type = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

# 3. Amount
amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)

# 4. Account balances
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0, step=100.0)
newbalanceOrg = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=4000.0, step=100.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=2000.0, step=100.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=3000.0, step=100.0)

# Add this ABOVE the button, in the sidebar:
threshold = st.sidebar.slider(
    "Fraud Threshold (probability)",
    min_value=0.01,
    max_value=0.99,
    value=0.30,
    step=0.01,
    help="If fraud probability is above this value, the transaction is flagged as fraud."
)
# ------------------------
#  Prediction Threshold UI
# ------------------------
threshold = st.sidebar.slider(
    "Fraud Threshold (probability)",
    min_value=0.01,
    max_value=0.99,
    value=0.30,
    step=0.01,
    help="Model flags fraud if probability >= threshold."
)

# ------------------------
#        Predict Button
# ------------------------
# ------------------------
#        Predict Button
# ------------------------
if st.button("🔍 Predict Fraud"):

    # ---- Recompute engineered features ----
    balance_diff_org = newbalanceOrg - oldbalanceOrg
    balance_diff_dest = newbalanceDest - oldbalanceDest

    raw = pd.DataFrame([{
        "step": step,
        "amount": amount,
        "balance_diff_org": balance_diff_org,
        "balance_diff_dest": balance_diff_dest,
        "type": tx_type
    }])

    # ---- One-hot encode the type (same as training) ----
    encoded = pd.get_dummies(raw, columns=["type"], drop_first=True)

    # ---- Make sure all expected columns exist ----
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    # ---- Scale numeric columns ----
    encoded[NUM_COLS] = scaler.transform(encoded[NUM_COLS])

    # ---- Predict ----
    proba = float(model.predict(encoded)[0][0])

    # ---- Apply threshold ----
    is_fraud = proba >= threshold
    label_text = "🚨 **Fraudulent Transaction**" if is_fraud else "✅ **Legit Transaction**"

    # ---- Display Results ----
    st.subheader("Result")
    st.markdown(label_text)
    st.write(f"**Fraud Probability:** `{proba:.4f}`")
    st.write(f"**Threshold Used:** `{threshold}`")

    with st.expander("See model input features"):
        st.dataframe(encoded)
