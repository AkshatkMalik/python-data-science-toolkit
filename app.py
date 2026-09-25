import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Titanic Survival Intelligence App",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for an enhanced dark mode look and polished components
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .reportview-container {
        background: #0e1117;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. MODEL & DEPENDENCY LOADING
# ==========================================
@st.cache_resource
def load_model():
  # Loads the serialized logistic regression model
  with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f"Error loading model file (`titanic_model.pkl`). Please ensure it is"
      f" placed in the `titanic-app` directory. Details: {e}"
  )
  st.stop()


# ==========================================
# 3. SIDEBAR & USER INPUT CONTROLS
# ==========================================
st.sidebar.header("🧭 Passenger Profile Setup")
st.sidebar.markdown(
    "Adjust the features below to simulate passenger demographics and evaluate"
    " survival probability using a trained Logistic Regression pipeline."
)


def user_input_features():
  # Categorical & Numerical inputs with explicit user-friendly labels
  pclass = st.sidebar.selectbox(
      "Passenger Class (Pclass)",
      options=[1, 2, 3],
      format_func=lambda x: (
          "1st Class (Upper)"
          if x == 1
          else ("2nd Class (Middle)" if x == 2 else "3rd Class (Lower)")
      ),
      help=(
          "Socio-economic status indicator (1 = Upper class, 3 = Lower class)."
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
      help="Age in years. Infants under 1 are represented as decimals.",
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
      "Ticket Fare (£ / $)",
      min_value=0.0,
      max_value=512.33,
      value=32.20,
      step=1.0,
      help="Passenger fare paid for the voyage.",
  )

  embarked = st.sidebar.selectbox(
      "Port of Embarkation",
      options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
      help="Port where the passenger boarded the ship.",
  )

  # Map categorical inputs to the encoding expected by your backend model
  sex_val = 1 if sex == "Female" else 0  # Adjust based on your model's encoding

  # Map embarked port string to model categorical codes if applicable
  embarked_map = {
      "Southampton (S)": 0,
      "Cherbourg (C)": 1,
      "Queenstown (Q)": 2,
  }
  embarked_val = embarked_map[embarked]

  # Compile features into a dataframe structure matching training columns
  # Note: Ensure this matches the exact feature columns/order your model was trained on.
  data = {
      "Pclass": pclass,
      "Sex": sex_val,
      "Age": age,
      "SibSp": sibsp,
      "Parch": parch,
      "Fare": fare,
  }
  features = pd.DataFrame(data, index=[0])
  return features


input_df = user_input_features()


# ==========================================
# 4. MAIN INTERFACE & DASHBOARD
# ==========================================
st.title("🚢 Titanic Survival Intelligence Dashboard")
st.markdown(
    """
    ### Predictive Analytics & Risk Assessment Tool
    This application leverages a production-grade **Logistic Regression model** trained on historical 
    Titanic passenger manifests. It evaluates critical survival vectors such as socio-economic status, 
    age profiles, and ticket pricing to estimate survival probability in real-time.
    """
)

st.divider()

# Display input summary metrics for professional presentation
st.subheader("📊 Current Simulation Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(
      label="Passenger Class",
      value=f"Class {int(input_df['Pclass'].values[0])}",
  )
with col2:
  gender_label = (
      "Female" if input_df["Sex"].values[0] == 1 else "Male"
  )
  st.metric(label="Demographic", value=gender_label)
with col3:
  st.metric(label="Age Profile", value=f"{input_df['Age'].values[0]:.1f} yrs")
with col4:
  st.metric(label="Ticket Fare", value=f"${input_df['Fare'].values[0]:.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 5. PREDICTION & VISUALIZATION LOGIC
# ==========================================
if st.button(
    "Run Survival Prediction Analysis", type="primary", use_container_width=True
):
  try:
    # Perform prediction and probability estimation
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    survival_prob = prediction_proba[0][1] * 100
    perish_prob = prediction_proba[0][0] * 100

    st.divider()
    st.subheader("🎯 Predictive Insights & Results")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
      if prediction[0] == 1:
        st.success(
            "### Outcome: Predicted SURVIVED ✨\nBased on historical patterns,"
            " this passenger profile matches favorable survival profiles."
        )
      else:
        st.error(
            "### Outcome: Predicted DID NOT SURVIVE ⚠️\nBased on historical"
            " patterns, this passenger profile faces high vulnerability risk."
        )

      st.metric(
          label="Calculated Survival Probability",
          value=f"{survival_prob:.2f}%",
      )

    with res_col2:
      st.markdown("#### Probability Distribution Breakdown")
      chart_data = pd.DataFrame(
          {
              "Outcome": ["Perished", "Survived"],
              "Probability (%)": [perish_prob, survival_prob],
          }
      )
      st.bar_chart(chart_data, x="Outcome", y="Probability (%)", color="#4CAF50")

  except Exception as e:
    st.error(
        f"An error occurred during model inference: {e}. Please check that"
        " your feature columns match the training schema."
    )

# Footer info
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #8b949e;'>Developed as part of Data"
    " Science Portfolio & Deployment Toolkit</p>",
    unsafe_allow_html=True,
)