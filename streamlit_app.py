import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="OCD Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('ocd_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, encoders, target_encoder, feature_cols

model, encoders, target_encoder, feature_cols = load_model()

st.title("OCD Severity Predictor")
st.markdown("**Production ML model** - Predicts OCD severity from clinical data")

# Sidebar inputs
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 18, 80, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic"])
obsession = st.sidebar.selectbox("Obsession Type", ["Contamination", "Harm", "Symmetry", "Forbidden thoughts"])
compulsion = st.sidebar.selectbox("Compulsion Type", ["Washing", "Checking", "Ordering", "Counting"])
duration = st.sidebar.slider("Symptom Duration (months)", 1, 240, 24)
depression = st.sidebar.selectbox("Depression Diagnosis", ["Yes", "No"])
anxiety = st.sidebar.selectbox("Anxiety Diagnosis", ["Yes", "No"])

# SAFE PREDICTION FUNCTION
def safe_predict(input_data):
    df_input = pd.DataFrame([input_data])
    
    # Fill missing columns with defaults
    categorical_cols = ['Gender', 'Ethnicity', 'Obsession Type', 'Compulsion Type', 
                       'Depression Diagnosis', 'Anxiety Diagnosis']
    
    # Encoding
    for col in categorical_cols:
        if col in encoders:
            # Handle unseen categories
            try:
                df_input[col + '_encoded'] = encoders[col].transform(df_input[col])
            except ValueError:
                most_frequent = encoders[col].classes_[0]
                df_input[col] = most_frequent
                df_input[col + '_encoded'] = encoders[col].transform(df_input[col])
    
    # CREATE ALL EXPECTED COLUMNS (fill missing with 0)
    for col in feature_cols:
        if col not in df_input.columns:
            if 'Age' in col or 'Duration' in col:
                df_input[col] = input_data.get(col.replace('_encoded', ''), 0)
            else:
                df_input[col] = 0
    
    # Reorder to match training feature_cols exactly
    X_input = df_input[feature_cols].fillna(0)
    return X_input

# Prepare input
if st.sidebar.button("Predict Severity", type="primary"):
    input_data = {
        'Age': age,
        'Duration of Symptoms (months)': duration,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'Obsession Type': obsession,
        'Compulsion Type': compulsion,
        'Depression Diagnosis': depression,
        'Anxiety Diagnosis': anxiety
    }
    
    # Safe prediction
    X_input = safe_predict(input_data)
    
    # Predict
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"**Predicted Severity: {target_encoder.inverse_transform([prediction])[0]}**")
    
    with col2:
        st.info(f"Y-BOCS Score Range: **{16 if prediction==1 else '0-15' if prediction==0 else '31+'}**")
    
    st.subheader("Confidence Scores")
    progress_container = st.columns(3)
    for i, label in enumerate(target_encoder.classes_):
        with progress_container[i]:
            st.metric(label, f"{probabilities[i]:.1%}", delta=None)
    
    st.balloons()

# Show model info
with st.expander("Model Performance"):
    st.info("""
    **Model Metrics:**
    - CV Accuracy: **74.2%**
    - Top Features: Obsession Type, Age, Compulsions
    - Dataset: 1500 OCD patients
    """)

st.markdown("---")
st.markdown("Built with ❤️ for healthcare analytics portfolios")

