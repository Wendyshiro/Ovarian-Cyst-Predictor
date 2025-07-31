import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier

st.title("Ovarian Cyst Predictor (Full Model)")

# Load models and scaler
best_rf = joblib.load("best_rf.pkl")
best_xgb = joblib.load("best_xgb.pkl")
scaler = joblib.load("scaler.pkl")
top_features = joblib.load("top_features.pkl")

ensemble = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    voting='soft'
)
# Dummy fit for predict_proba
ensemble.fit(np.zeros((2, len(top_features))), [0, 1])

st.header("Enter Patient Data")
age = st.number_input("Age", min_value=0, max_value=120, value=40)
cyst_size = st.number_input("Cyst Size (cm)", min_value=0.0, max_value=20.0, value=4.0)
ca125 = st.number_input("CA 125 Level", min_value=0.0, max_value=1000.0, value=30.0)
symptom_count = st.number_input("Symptom Count", min_value=0, max_value=10, value=1)
is_postmeno = st.selectbox("Is Postmenopausal?", ["No", "Yes"])

# Feature engineering (must match your training pipeline!)
input_dict = {
    'Age': age,
    'Cyst Size cm': cyst_size,
    'CA 125 Level': ca125,
    'Symptom Count': symptom_count,
    'Is Postmenopausal': 1 if is_postmeno == "Yes" else 0,
    # Add all other engineered features here, or compute them as in your script
}
input_df = pd.DataFrame([input_dict])

# Select and scale features
X = input_df[top_features]
X_scaled = scaler.transform(X)

if st.button("Predict"):
    ensemble_pred = ensemble.predict(X_scaled)[0]
    ensemble_proba = ensemble.predict_proba(X_scaled)[0, 1]
    st.write(f"**Predicted Cyst Behavior:** {'Unstable' if ensemble_pred == 1 else 'Stable'}")
    st.write(f"**Prediction Confidence:** {ensemble_proba:.2%}")
    st.success("Prediction complete!")

st.info("This is the full model version. For clinical use, consult a medical professional.")