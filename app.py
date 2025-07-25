import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings in the deployed app
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Export Authorization Predictor",
    page_icon="🚢",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the pre-trained model from the .pkl file."""
    try:
        # The path should correspond to the file in your GitHub repository
        model = joblib.load('best_export_clearance_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_export_clearance_model.pkl' is in the GitHub repository.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Main Application ---
st.title("🚢 Export Authorization Status Predictor")
st.markdown("""
This application predicts whether an export transaction will be **Approved** or **Not Approved**.

Upload a CSV file with your transaction data to get predictions. Please ensure your CSV has the required columns as used in the training data.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")

if uploaded_file is not None and model is not None:
    try:
        # Load the user's data
        input_df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully! Processing...")
        
        # Keep a copy of the original data for display purposes
        original_df = input_df.copy()

        # --- Feature Engineering (Must be IDENTICAL to the training script) ---
        st.write("Applying feature engineering...")
        
        # Re-create the exact same features that the model was trained on
        input_df['Load Deviation'] = input_df['Quantity'] - input_df['Minimum Order Value']
        input_df['Transaction_Value'] = input_df['Quantity'] * input_df['Rate']

        # Ensure all required features for the model are present
        required_features = [
            'Country of FD Name', 'Item Category', 'Quantity', 'Item Net Weight', 'Rate',
            'Invoice Quantity', 'Invoice Status', 'Extra Fields Final Destination',
            'Sourcing Type', 'Minimum Order Value', 'Load Deviation', 'Transaction_Value'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_features if col not in input_df.columns]
        if missing_cols:
            st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
        else:
            X_predict = input_df[required_features]

            # --- Prediction ---
            st.write("Making predictions...")
            predictions = model.predict(X_predict)
            prediction_proba = model.predict_proba(X_predict)

            # --- Display Results ---
            original_df['Predicted Status'] = ['Approved' if pred == 1 else 'Not Approved' for pred in predictions]
            original_df['Approval Probability (%)'] = [f"{proba[1]*100:.2f}" for proba in prediction_proba]

            st.subheader("Prediction Results")
            # Display relevant original columns plus the new prediction columns
            st.dataframe(original_df[[
                'Country of FD Name', 'Item Category', 'Quantity', 'Transaction_Value', 
                'Predicted Status', 'Approval Probability (%)'
            ]])

            # --- Download Button ---
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_results = convert_df_to_csv(original_df)
            st.download_button(
                label="Download Full Results as CSV",
                data=csv_results,
                file_name="predicted_authorizations.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.warning("Please ensure your CSV file is formatted correctly.")
