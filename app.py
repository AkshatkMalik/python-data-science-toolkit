import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter the passenger details below to predict their survival probability using Logistic Regression.")

# Create input widgets for user features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
sex_val = 1 if sex == "Female" else 0

age = st.slider("Age", 1, 80, 28)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Ticket Fare ($)", 0.0, 500.0, 32.0)

embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
embarked_val = embarked_map[embarked]

# Prediction button
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 The passenger is **Likely to Survive** (Survival Probability: {probability:.2f})")
    else:
        st.error(f"⚠️ The passenger is **Unlikely to Survive** (Survival Probability: {probability:.2f})")
