import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('uber_scaled (1).pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Uber Trip Prediction", page_icon="🚖", layout="centered")
st.title('🚖 Uber Trip Analysis Prediction')
st.markdown("**Machine Learning Model: Gradient Boosting Regressor**")
st.info("Adjust the input parameters in the sidebar to see the prediction in real time.")


# Sidebar inputs
with st.sidebar:
    st.header('📊 Input Features')
    active_vehicles = st.slider('Active Vehicles', 112.0, 1619.0)
    Hour = st.slider('Hour', 0.0, 24.0)
    DayofWeek = st.slider('Day of Week', 1.0, 7.0)
    Month = st.slider('Month', 1.0, 12.0)
    Day = st.slider('Day', 1.0, 31.0)

# Predict button
if st.button("🚀 Predict"):
    data = pd.DataFrame([[active_vehicles, Hour, DayofWeek, Month, Day]],
                        columns=['active_vehicles', 'Hour', 'DayofWeek', 'Month', 'Day'])

    predictions = best_model.predict(data)

    # Simulate a loading animation
    with st.spinner('Calculating prediction...'):
        time.sleep(1)

    # Stylish display
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Value</h2>
            <h1 style="color:#FF4500;font-size:60px;">{predictions[0]:,.2f}</h1>
            <p style="color:gray;">Estimated Uber demand based on provided features</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Optional: Metric display
    st.metric(label="📈 Predicted Uber Demand", value=f"{predictions[0]:,.2f}")




  
    
