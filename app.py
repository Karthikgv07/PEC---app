# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Title
st.title("Export Clearance Approval Prediction App")

# Load cleaned data
DATA_PATH = "Cleaned_Transaction_Data.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"🚫 File not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Input section
st.sidebar.header("User Input")

sample = {}
categorical_inputs = {
    'Country of FD Name': df['Country of FD Name'].dropna().unique().tolist(),
    'Item Category': df['Item Category'].dropna().unique().tolist(),
    'Invoice Status': df['Invoice Status'].dropna().unique().tolist(),
    'Extra Fields Final Destination': df['Extra Fields Final Destination'].dropna().unique().tolist(),
    'Sourcing Type': df['Sourcing Type'].dropna().unique().tolist(),
}

for col in categorical_inputs:
    sample[col] = st.sidebar.selectbox(f"{col}:", categorical_inputs[col])

for col in ['Quantity', 'Item Net Weight', 'Item Grams', 'Rate', 'Invoice Quantity', 'Minimum Order Value', 'Load Deviation']:
    default_val = float(df[col].mean()) if col in df.columns else 0.0
    sample[col] = st.sidebar.number_input(f"{col}:", value=default_val)

input_df = pd.DataFrame([sample])

# ----------- Placeholder since model is missing -------------
st.subheader("Prediction (Not available)")
st.warning("⚠️ Prediction model files not found. Only UI demo is active.")
st.dataframe(input_df)
