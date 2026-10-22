import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="IPO One-Month Price Predictor", page_icon="📈", layout="centered")
st.title("📈 IPO One-Month Price Predictor")
st.write(
    "Enter values for the model's input features to estimate the share price one month after listing. "
    "This is an educational model, not investment advice."
)

required = [
    ART / "ipo_model.keras",
    ART / "x_scaler.joblib",
    ART / "y_scaler.joblib",
    ART / "fill_values.joblib",
    ART / "feature_columns.json",
]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error(
        "The trained model artifacts are missing: " + ", ".join(missing) +
        ". Run `python train_model.py` first, then restart the app."
    )
    st.stop()

@st.cache_resource
def load_artifacts():
    model = load_model(ART / "ipo_model.keras")
    x_scaler = joblib.load(ART / "x_scaler.joblib")
    y_scaler = joblib.load(ART / "y_scaler.joblib")
    fill_values = joblib.load(ART / "fill_values.joblib")
    features = json.loads((ART / "feature_columns.json").read_text())
    return model, x_scaler, y_scaler, fill_values, features

model, x_scaler, y_scaler, fill_values, features = load_artifacts()

st.caption("Model inputs are based on the feature columns selected during training.")
st.info("Use the same units and definitions as your training dataset. Required features: " + ", ".join(features))

with st.form("ipo_inputs"):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        default = float(fill_values.get(feature, 0.0))
        # Keep input widgets numeric. Users should enter values in the same units as training.
        with cols[i % 2]:
            values[feature] = st.number_input(
                feature,
                value=default,
                format="%.6f",
                help="Enter this value in the same units used in your training dataset."
            )
    submitted = st.form_submit_button("Predict one-month price", type="primary")

if submitted:
    try:
        row = pd.DataFrame([[values[f] for f in features]], columns=features)
        scaled_x = x_scaler.transform(row.to_numpy(dtype=float))
        scaled_prediction = model.predict(scaled_x, verbose=0).reshape(-1, 1)
        target_units = float(y_scaler.inverse_transform(scaled_prediction)[0, 0])
        predicted_inr = target_units * 1000.0

        if not np.isfinite(predicted_inr):
            raise ValueError("The model returned a non-finite prediction.")
        st.subheader("Estimated share price after one month")
        st.metric("Predicted price", f"₹{predicted_inr:,.2f}")
        st.caption(
            "Conversion used: inverse MinMax scaling, then × 1,000 because "
            "`Price_after_1m` stores the rupee price divided by 1,000."
        )
        st.warning(
            "This is a model estimate, not a guaranteed market price. Predictions can be unreliable "
            "if inputs differ from the training data or the model has limited historical examples."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
