import streamlit as st
import pandas as pd
import joblib
import warnings

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
    """Load the pre-trained model from the file."""
    try:
        model = joblib.load('export_clearance_model_final.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'export_clearance_model_final.pkl' is in the repository.")
        return None

model = load_model()

# --- Main Application ---
st.title("🚢 Export Authorization Status Predictor")
st.markdown("""
This application predicts whether an export transaction will be **Approved** or **Not Approved**.
Upload a CSV file with transaction data to get predictions. Please ensure your CSV has the required columns.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")

if uploaded_file is not None and model is not None:
    try:
        # Load the uploaded data
        input_df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        
        # --- Feature Engineering ---
        # Create the same features as in your training script
        st.write("Applying feature engineering...")
        
        # Basic Features
        input_df['Transaction_Value'] = input_df['Quantity'] * input_df['Rate']
        
        # Advanced Aggregation Features
        # Note: These aggregations are based on the *uploaded data itself* for prediction
        input_df['Load_Deviation'] = input_df['Quantity'] - input_df['Minimum Order Value']
        input_df['Value_per_Weight'] = input_df['Transaction_Value'] / (input_df['Item Net Weight'] + 1e-6)

        item_agg = input_df.groupby('Item Category')['Transaction_Value'].agg(['mean', 'std', 'max']).add_prefix('Item_Value_')
        input_df = input_df.join(item_agg, on='Item Category')

        country_agg = input_df.groupby('Country of FD Name')['Quantity'].agg(['mean', 'sum']).add_prefix('Country_Quant_')
        input_df = input_df.join(country_agg, on='Country of FD Name')

        input_df.fillna(0, inplace=True)

        # Ensure all required features are present
        required_features = [
            'Quantity', 'Item Net Weight', 'Rate', 'Minimum Order Value', 'Transaction_Value',
            'Load_Deviation', 'Value_per_Weight', 'Item_Value_mean', 'Item_Value_std',
            'Item_Value_max', 'Country_Quant_mean', 'Country_Quant_sum',
            'Country of FD Name', 'Item Category', 'Sourcing Type'
        ]
        
        X_predict = input_df[required_features]

        # --- Prediction ---
        st.write("Making predictions...")
        predictions = model.predict(X_predict)
        prediction_proba = model.predict_proba(X_predict)

        # --- Display Results ---
        result_df = input_df.copy()
        result_df['Predicted Status'] = ['Approved' if pred == 1 else 'Not Approved' for pred in predictions]
        result_df['Approval Probability (%)'] = [f"{proba[1]*100:.2f}" for proba in prediction_proba]

        st.subheader("Prediction Results")
        st.dataframe(result_df[['Country of FD Name', 'Item Category', 'Quantity', 'Predicted Status', 'Approval Probability (%)']])

        # --- Download Button ---
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_results = convert_df_to_csv(result_df)
        st.download_button(
            label="Download Results as CSV",
            data=csv_results,
            file_name="predicted_authorizations.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure your CSV file is formatted correctly and includes all necessary columns.")
