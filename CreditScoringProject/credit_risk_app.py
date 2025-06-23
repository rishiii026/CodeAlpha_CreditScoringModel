import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
df = pd.read_csv("german_credit_full.csv")
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Risk", axis=1)
y = df["Risk"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression()
log_model.fit(X_scaled, y)

rf_model = RandomForestClassifier()
rf_model.fit(X_scaled, y)

# Streamlit UI Setup
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
   


<style>
/* App Background */
.stApp {
    background: #000000;
    color: #f5f5f5;
}

/* Glassmorphism Box */
.glass-box {
    background: rgba(255, 255, 255, 0.04);
    color: #f5f5f5;
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* Headers and Text */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSubheader {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    color: #ffffff !important;
    background: linear-gradient(to right, #1a1a1a, #333333) !important;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.4em;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
}

.stButton > button:hover {
    background: linear-gradient(to right, #2a2a2a, #444444) !important;
}

/* Form Fields (Dropdown, Slider, etc.) */
.stSelectbox, .stNumberInput, .stRadio, .stTextInput, .stDateInput,
.stSlider > div[data-baseweb="slider"] {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 0.6em !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

/* Hover / Focus state for input fields */
.stSelectbox:hover, .stNumberInput:hover, .stRadio:hover, .stTextInput:hover, .stDateInput:hover,
.stSlider > div[data-baseweb="slider"]:hover {
    background-color: #222222 !important;
    transition: background-color 0.2s ease-in-out;
}
</style>




""", unsafe_allow_html=True)



st.title("💼 Credit Risk Evaluation Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Applicant Info", "📈 Prediction", "📊 Credit Score"])

with tab1:
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.subheader("📥 Fill in Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox('Age', list(range(18, 76)), index=12)
        sex = st.selectbox('Sex', ['male', 'female'])
        job = st.radio('Job Type', [0, 1, 2, 3], horizontal=True)
        housing = st.selectbox('Housing', ['own', 'rent', 'for free'])
        saving = st.selectbox('Saving Accounts', ['little', 'moderate', 'quite rich', 'rich'])

    with col2:
        checking = st.selectbox('Checking Account', ['none', 'little', 'moderate', 'rich'])
        credit_amount = st.number_input('Credit Amount', 500, 10000, step=100)
        duration = st.selectbox('Loan Duration (months)', list(range(6, 61)), index=6)
        purpose = st.selectbox('Purpose', ['radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 'business'])
        model_choice = st.radio("Select Model", ["Logistic Regression", "Random Forest"], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

input_data = {
    'Age': age,
    'Sex': label_encoders['Sex'].transform([sex])[0],
    'Job': job,
    'Housing': label_encoders['Housing'].transform([housing])[0],
    'Saving accounts': label_encoders['Saving accounts'].transform([saving])[0],
    'Checking account': label_encoders['Checking account'].transform([checking])[0],
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': label_encoders['Purpose'].transform([purpose])[0]
}

user_df = pd.DataFrame([input_data])
user_scaled = scaler.transform(user_df)

if model_choice == "Logistic Regression":
    pred = log_model.predict(user_scaled)
    prob = log_model.predict_proba(user_scaled)[0][pred[0]]
else:
    pred = rf_model.predict(user_scaled)
    prob = rf_model.predict_proba(user_scaled)[0][pred[0]]

risk_result = label_encoders['Risk'].inverse_transform(pred)[0]
credit_score = int(300 + prob * 550)

with tab2:
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.subheader("🔍 Prediction Output")
    score_placeholder = st.empty()
    for s in range(300, credit_score + 1, 10):
        score_placeholder.metric("📈 Credit Score", value=s)
        time.sleep(0.01)

    if risk_result == "good":
        st.success(f"✅ Result: GOOD CREDIT ({prob*100:.2f}% confidence)")
        st.markdown("✔️ This applicant is likely to be approved for credit.")
    else:
        st.error(f"❌ Result: BAD CREDIT ({prob*100:.2f}% confidence)")
        st.markdown("⚠️ High risk profile. Loan might be denied.")

    st.progress(prob if risk_result == 'good' else 1 - prob)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.subheader("📊 Score Visualization")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        title={'text': "Simulated Credit Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#764ba2"},
            'steps': [
                {'range': [300, 579], 'color': '#ff4d4d'},
                {'range': [580, 669], 'color': '#ffa500'},
                {'range': [670, 739], 'color': '#ffff66'},
                {'range': [740, 799], 'color': '#90ee90'},
                {'range': [800, 850], 'color': '#32cd32'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.info("💡 Pro Tip: Maintain good payment history and keep balances low to increase your credit score.")
