# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

# ------------------- Title -------------------
st.title("🚢 Export Clearance Approval Prediction App")

# ---------------- Load cleaned data ----------------
DATA_PATH = "Cleaned_Transaction_Data.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"🚫 File not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# --------------- Sidebar Input ----------------
st.sidebar.header("🧾 User Input")

sample = {}
categorical_inputs = {
    'Country of FD Name': df['Country of FD Name'].dropna().unique().tolist(),
    'Item Category': df['Item Category'].dropna().unique().tolist(),
    'Invoice Status': df['Invoice Status'].dropna().unique().tolist(),
    'Extra Fields Final Destination': df['Extra Fields Final Destination'].dropna().unique().tolist(),
    'Sourcing Type': df['Sourcing Type'].dropna().unique().tolist(),
}

# --- Collect categorical inputs ---
for col in categorical_inputs:
    sample[col] = st.sidebar.selectbox(f"{col}:", categorical_inputs[col])

# --- Collect numerical inputs ---
numerical_columns = [
    'Quantity', 'Item Net Weight', 'Item Grams', 'Rate',
    'Invoice Quantity', 'Minimum Order Value', 'Load Deviation'
]

for col in numerical_columns:
    default_val = float(df[col].mean()) if col in df.columns else 0.0
    sample[col] = st.sidebar.number_input(f"{col}:", value=default_val)

# Convert input to DataFrame
input_df = pd.DataFrame([sample])

# ---------------- Display Input ----------------
st.subheader("🧮 User Input Summary")
st.dataframe(input_df)

# ---------------- Simulate Prediction ----------------
st.subheader("📢 Prediction Result")
st.warning("⚠️ No trained model found. Prediction is simulated based on input.")

# Optional: Match user input against similar past records
if 'Authorization Status' in df.columns:
    try:
        from sklearn.neighbors import NearestNeighbors

        feature_cols = numerical_columns  # use numerical features for matching
        nbrs = NearestNeighbors(n_neighbors=1).fit(df[feature_cols].dropna())
        distance, index = nbrs.kneighbors(input_df[feature_cols])

        matched_row = df.iloc[index[0][0]]
        prediction = matched_row.get("Authorization Status", "Unknown")
        st.success(f"🟢 Most similar past transaction was: **{prediction}**")

    except Exception as e:
        st.error(f"⚠️ Unable to simulate prediction: {e}")
else:
    st.info("ℹ️ Add a column 'Authorization Status' in the CSV to simulate predictions.")
