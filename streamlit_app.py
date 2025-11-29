import streamlit as st
import pandas as pd
from utils import predict_transaction

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

st.title("💳 Credit Card Fraud Detection Demo (Hybrid ML Model)")
st.markdown("""
This demo uses a **hybrid model** combining:
- **XGBoost fraud probability**
- **Isolation Forest anomaly score**

Adjust parameters or upload custom data to test the model.
""")

# Sidebar controls
st.sidebar.header("Model Settings")
alpha = st.sidebar.slider("Alpha (supervised weight)", 0.0, 1.0, 0.7)
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

# CSV upload
uploaded = st.file_uploader("Upload CSV (no header)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, header=None)
    st.subheader("Uploaded Transactions")
    st.dataframe(df.head())

    if st.button("Predict Uploaded File"):
        with st.spinner("Predicting..."):
            results = predict_transaction(df.values.tolist(), alpha, threshold)
        st.subheader("Results (first 20 shown)")
        st.write(results[:20])

# Manual input
st.markdown("---")
st.subheader("Predict a Single Transaction Manually")

n_features = st.number_input("Number of features", value=30, min_value=1)
cols = st.columns(3)

values = []
for i in range(n_features):
    values.append(cols[i % 3].number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict Manual Input"):
    with st.spinner("Predicting..."):
        result = predict_transaction([values], alpha, threshold)[0]
    st.json(result)
