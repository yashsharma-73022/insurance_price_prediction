import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Config ---
st.set_page_config(page_title="Insurance Predictor", layout="centered")

# --- Custom Neon CSS ---
st.markdown("""
    <style>
        body {
            background-color: #0f0f0f;
            color: #39ff14;
        }
        .main {
            background-color: #0f0f0f;
        }
        h1, h2, h3, .stTextInput label, .stSelectbox label, .stSlider label, .stRadio label {
            color: #39ff14 !important;
            text-shadow: 0 0 5px #39ff14;
        }
        .stButton > button {
            background-color: #000;
            color: #39ff14;
            border: 1px solid #39ff14;
            box-shadow: 0 0 5px #39ff14;
        }
        .stButton > button:hover {
            background-color: #39ff14;
            color: #000;
            box-shadow: 0 0 20px #39ff14;
        }
        .stImage img {
            border: 2px solid #39ff14;
            box-shadow: 0 0 20px #39ff14;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Top Image ---
st.image("medical_image.webp", use_container_width=True)

# --- Title ---
st.title("💰 Medical Insurance Cost Prediction")

# --- Load the model ---
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("pipeline.pkl"):
            st.error("❌ File 'pipeline.pkl' not found.")
            return None
        with open("pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# --- Input form ---
if model is not None:
    with st.form("predict_form"):
        st.subheader("📝 Enter Patient Information")

        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        children = st.slider("Children", 0, 5, 0)
        smoker = st.radio("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        st.markdown("### 📦 Input Summary:")
        st.dataframe(input_data.style.highlight_max(axis=0), use_container_width=True)

        try:
            prediction = model.predict(input_data)[0]
            st.markdown(f"""
                <div style='font-size:24px; font-weight:bold; margin-top:20px;
                            color:#39ff14; text-shadow: 0 0 10px #39ff14;'>
                    💸 Estimated Insurance Cost: ${prediction:,.2f}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("⚠️ Prediction failed.")
            st.exception(e)
else:
    st.error("❌ Model not loaded. Prediction disabled.")

# --- Developer credit ---
st.markdown("""
    <hr style="border: none; border-top: 1px solid #39ff14; margin-top: 40px;" />
    <div style='text-align: center; font-weight: bold; color: #39ff14; text-shadow: 0 0 10px #39ff14; font-size: 18px;'>
        Developed by Yash Sharma
    </div>
""", unsafe_allow_html=True)
