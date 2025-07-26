import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings for clean UI
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Export Authorization Predictor",
    page_icon="Prediction",
    layout="wide"
)

# --- Model Loader ---
@st.cache_resource
def load_model():
    """Load the trained ML model from a pickle file."""
    try:
        model = joblib.load('best_export_clearance_model.pkl')
        return model
    except FileNotFoundError:
        st.error(" Model file not found. Please upload 'best_export_clearance_model.pkl' to the same folder.")
        return None
    except Exception as e:
        st.error(f" Unexpected error while loading model: {e}")
        return None

# Load model once
model = load_model()

# --- UI Header ---
st.title(" Export Authorization Status Predictor")
st.markdown("""
This app predicts whether a transaction will be **Approved** or **Not Approved** based on your data.  
Upload a `.csv` file with proper column format.
""")

# --- Upload File ---
uploaded_file = st.file_uploader(" Upload your CSV file", type="csv")

if uploaded_file is not None:
    if model is None:
        st.warning(" Model not loaded. Cannot make predictions.")
    else:
        try:
            # Load CSV
            input_df = pd.read_csv(uploaded_file)
            original_df = input_df.copy()
            st.success(" File uploaded successfully!")

            # --- Feature Engineering (must match training pipeline exactly) ---
            input_df['Load Deviation'] = input_df['Quantity'] - input_df['Minimum Order Value']
            input_df['Transaction_Value'] = input_df['Quantity'] * input_df['Rate']

            # Expected columns
            required_features = [
                'Country of FD Name', 'Item Category', 'Quantity', 'Item Net Weight', 'Rate',
                'Invoice Quantity', 'Invoice Status', 'Extra Fields Final Destination',
                'Sourcing Type', 'Minimum Order Value', 'Load Deviation', 'Transaction_Value'
            ]

            # --- Check Required Columns ---
            missing = [col for col in required_features if col not in input_df.columns]
            if missing:
                st.error(f" Missing columns: {', '.join(missing)}")
            else:
                X = input_df[required_features]

                # --- Predictions ---
                st.info(" Making predictions...")
                preds = model.predict(X)
                probs = model.predict_proba(X)

                # Add prediction output
                original_df['Predicted Status'] = ['Approved' if p == 1 else 'Not Approved' for p in preds]
                original_df['Approval Probability (%)'] = [f"{p[1]*100:.2f}" for p in probs]

                # --- Show Results ---
                st.subheader(" Prediction Results")
                st.dataframe(original_df[[
                    'Country of FD Name', 'Item Category', 'Quantity', 'Transaction_Value',
                    'Predicted Status', 'Approval Probability (%)'
                ]])

                # --- Download Button ---
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label=" Download Predictions as CSV",
                    data=convert_df(original_df),
                    file_name='predicted_results.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("📌 Please check if the file format and data types are correct.")
