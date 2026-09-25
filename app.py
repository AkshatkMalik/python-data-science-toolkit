import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & CINEMATIC DARK STYLING
# ==========================================
st.set_page_config(
    page_title="Titanic Survival Intelligence App",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cinematic dark mode, glassmorphism cards, and sleek typography
st.markdown(
    """
    <style>
    /* Global Cinematic Background Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a1f2c 0%, #0d1117 100%);
        color: #f0f6fc;
    }
    
    /* Glassmorphism Container Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    
    /* Header styling */
    h1, h2, h3 {
        letter-spacing: -0.5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. MODEL LOADING WITH ERROR HANDLING
# ==========================================
@st.cache_resource
def load_model():
  # Load the trained Logistic Regression model from pickle file
  with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f"⚠️ Critical Error: Could not load `titanic_model.pkl`. Ensure it is"
      f" inside the `titanic-app` folder. Details: {e}"
  )
  st.stop()


# ==========================================
# 3. SIDEBAR USER CONTROLS (WITH EXPLICIT LABELS)
# ==========================================
st.sidebar.markdown("## 🧭 Passenger Profile Setup")
st.sidebar.markdown(
    "Configure passenger demographics below to evaluate real-time survival"
    " probability."
)


def user_input_features():
  # Explicit labels prevent blank input boxes
  pclass = st.sidebar.selectbox(
      "Passenger Class (Pclass)",
      options=[1, 2, 3],
      format_func=lambda x: (
          "1st Class (Upper)"
          if x == 1
          else ("2nd Class (Middle)" if x == 2 else "3rd Class (Lower)")
      ),
      help=(
          "Socio-economic status: 1 = Upper class, 2 = Middle class, 3 = Lower"
          " class"
      ),
  )

  sex = st.sidebar.selectbox(
      "Sex / Gender",
      options=["Male", "Female"],
      help="Biological sex of the passenger.",
  )

  age = st.sidebar.slider(
      "Age (Years)",
      min_value=0.42,
      max_value=80.0,
      value=28.0,
      step=1.0,
      help="Age of the passenger in years.",
  )

  sibsp = st.sidebar.number_input(
      "Siblings / Spouses Aboard (SibSp)",
      min_value=0,
      max_value=8,
      value=0,
      step=1,
      help="Number of siblings or spouses traveling with the passenger.",
  )

  parch = st.sidebar.number_input(
      "Parents / Children Aboard (Parch)",
      min_value=0,
      max_value=6,
      value=0,
      step=1,
      help="Number of parents or children traveling with the passenger.",
  )

  fare = st.sidebar.number_input(
      "Ticket Fare ($)",
      min_value=0.0,
      max_value=512.33,
      value=32.20,
      step=1.0,
      help="Ticket price paid for the voyage.",
  )

  embarked = st.sidebar.selectbox(
      "Port of Embarkation",
      options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
      help="Port where the passenger boarded.",
  )

  # Data Preprocessing/Encoding mapping to match model training features
  sex_val = 1 if sex == "Female" else 0

  # Map embarked to numerical code matching training schema
  embarked_map = {
      "Southampton (S)": 0,
      "Cherbourg (C)": 1,
      "Queenstown (Q)": 2,
  }
  embarked_val = embarked_map[embarked]

  # Compile into a DataFrame ensuring exact feature match including 'Embarked'
  data = {
      "Pclass": [pclass],
      "Sex": [sex_val],
      "Age": [age],
      "SibSp": [sibsp],
      "Parch": [parch],
      "Fare": [fare],
      "Embarked": [embarked_val],
  }
  return pd.DataFrame(data)


input_df = user_input_features()


# ==========================================
# 4. MAIN DASHBOARD UI & VISUALS
# ==========================================
st.title("🚢 Titanic Survival Intelligence Dashboard")
st.markdown(
    """
    ### Production-Grade Logistic Regression Analytics
    Welcome to the interactive risk-assessment portal. This application analyzes historical 
    demographic attributes and socio-economic markers to predict passenger survival outcomes.
    """
)

st.divider()

# Aesthetic Summary Metric Display Cards
st.subheader("📊 Active Simulation Parameter Overview")
m1, m2, m3, m4 = st.columns(4)

with m1:
  st.metric(
      label="Socioeconomic Class",
      value=f"Class {int(input_df['Pclass'].values[0])}",
  )
with m2:
  gender_str = "Female" if input_df["Sex"].values[0] == 1 else "Male"
  st.metric(label="Demographic Profile", value=gender_str)
with m3:
  st.metric(label="Age Parameter", value=f"{input_df['Age'].values[0]:.1f} yrs")
with m4:
  st.metric(label="Fare Allocation", value=f"${input_df['Fare'].values[0]:.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 5. PREDICTION & INTERACTIVE VISUALIZATIONS
# ==========================================
if st.button(
    "✨ Execute Predictive Analysis", use_container_width=True
):
  try:
    # Run model prediction & probability inference
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    survive_prob = probabilities[0][1] * 100
    perish_prob = probabilities[0][0] * 100

    st.divider()
    st.subheader("🎯 Predictive Output & Insights")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
      if prediction[0] == 1:
        st.success(
            "### Status: PREDICTED SURVIVOR ✨\nBased on model weights,"
            " historical indicators favor survival for this configuration."
        )
      else:
        st.error(
            "### Status: HIGH RISK / PERISHED ⚠️\nBased on model weights,"
            " historical indicators show vulnerability for this configuration."
        )

      st.metric(
          label="Calculated Survival Probability Index",
          value=f"{survive_prob:.2f}%",
      )

    with res_col2:
      st.markdown("#### 📈 Probability Distribution Chart")
      chart_df = pd.DataFrame(
          {
              "Outcome State": ["Perished Risk", "Survival Probability"],
              "Probability (%)": [perish_prob, survive_prob],
          }
      )
      # Native interactive Streamlit bar chart with styled visual metrics
      st.bar_chart(chart_df, x="Outcome State", y="Probability (%)", color="#FF4B4B")

  except Exception as err:
    st.error(
        f"Inference execution error: {err}. Please verify column schema"
        " consistency."
    )

# Footer credit
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #8b949e; font-size: 0.9rem;'>Titanic"
    " Machine Learning Deployment Toolkit | Built with Streamlit & Scikit-Learn</p>",
    unsafe_allow_html=True,
)