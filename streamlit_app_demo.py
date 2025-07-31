import streamlit as st

st.title("Ovarian Cyst Predictor (Demo)")

st.header("Enter Patient Data")
age = st.number_input("Age", min_value=0, max_value=120, value=40)
cyst_size = st.number_input("Cyst Size (cm)", min_value=0.0, max_value=20.0, value=4.0)
ca125 = st.number_input("CA 125 Level", min_value=0.0, max_value=1000.0, value=30.0)
symptom_count = st.number_input("Symptom Count", min_value=0, max_value=10, value=1)
is_postmeno = st.selectbox("Is Postmenopausal?", ["No", "Yes"])

if st.button("Predict"):
    st.write("**Predicted Cyst Behavior:** Unstable (Demo)")
    st.write("**Prediction Confidence:** 85% (Demo)")
    st.info("This is a demo. For real predictions, use the full model version.")