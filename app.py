# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Export Clearance Approval Prediction App")

# Load cleaned data
df = pd.read_csv("cleaned_transactions_Data.csv")

# Load model & pipeline
model = pickle.load(open("model/rf_smote_model.pkl", "rb"))
pipeline = joblib.load("model/clean_export_pipeline.pkl")

# Input features
st.sidebar.header("User Input")

sample = {}
categorical_inputs = {
    'Country of FD Name': df['Country of FD Name'].unique().tolist(),
    'Item Category': df['Item Category'].unique().tolist(),
    'Invoice Status': df['Invoice Status'].unique().tolist(),
    'Extra Fields Final Destination': df['Extra Fields Final Destination'].unique().tolist(),
    'Sourcing Type': df['Sourcing Type'].unique().tolist(),
}

for col in categorical_inputs:
    sample[col] = st.sidebar.selectbox(f"{col}:", categorical_inputs[col])

for col in ['Quantity', 'Item Net Weight', 'Item Grams', 'Rate', 'Invoice Quantity', 'Minimum Order Value', 'Load Deviation']:
    sample[col] = st.sidebar.number_input(f"{col}:", value=float(df[col].mean()))

input_df = pd.DataFrame([sample])

# Preprocess and predict
transformed_input = pipeline.transform(input_df)
prediction = model.predict(transformed_input)[0]
prediction_proba = model.predict_proba(transformed_input)[0][1]

# Output
st.subheader("Prediction")
st.write("✅ Approved" if prediction == 1 else "❌ Not Approved")
st.write(f"Approval Probability: {prediction_proba:.2f}")
