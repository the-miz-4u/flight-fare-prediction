import streamlit as st
import pandas as pd
import numpy as np
import pickle

import gdown
import os

# Download model if not present
if not os.path.exists("flight_rf_model.pkl"):
    url = "https://drive.google.com/uc?id=14hqRLBFueistUCERjS_06nEgAlsC_PEJ"
    gdown.download(url, "flight_rf_model.pkl", quiet=False)

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Flight Fare Genie",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("flight_rf_model.pkl", "rb"))
    model_features = pickle.load(open("model_features.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please ensure 'flight_rf_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop()

# ================= DARK THEME CSS =================
st.markdown("""
<style>
    /* Google Font Import - Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background - Deep Dark Grey */
    .stApp {
        background-color: #0E1117;
    }

    /* Title Styling */
    .main-title {
        color: #FFFFFF;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .sub-title {
        color: #00ADB5; /* Neon Teal Accent */
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Dark Cards for Inputs */
    .dark-card {
        background-color: #161B22; /* Slightly lighter than bg */
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #30363D; /* Subtle border */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Input Labels - Make them White */
    .stSelectbox label, .stDateInput label, .stTimeInput label, .stSlider label {
        color: #E6E6E6 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Button Styling - Neon Effect */
    div.stButton > button {
        background: #00ADB5; /* Teal Color */
        color: #0E1117;
        width: 100%;
        padding: 15px !important;
        font-size: 18px !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background: #00FFF5; /* Bright Neon on Hover */
        box-shadow: 0 0 15px rgba(0, 255, 245, 0.6);
        color: black;
        transform: translateY(-2px);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #4A90E2;
    }
    
    /* Divider Customization */
    hr {
        border-color: #30363D;
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER SECTION =================
st.markdown("<h1 class='main-title'>FLIGHT GENIE</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>PREDICT FLIGHT FARES WITH AI</p>", unsafe_allow_html=True)

# ================= MAIN FORM CONTAINER =================
_, main_col, _ = st.columns([1, 8, 1])

with main_col:
    # Wrap inputs in a custom dark card
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
    
    # Row 1: Route
    c1, c2, c3 = st.columns([3, 1, 3])
    with c1:
        source = st.selectbox("FROM", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"])
    with c2:
        st.markdown("<h3 style='text-align: center; color: #555; margin-top: 30px;'>✈</h3>", unsafe_allow_html=True)
    with c3:
        destination = st.selectbox("TO", ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Details
    c4, c5, c6 = st.columns(3)
    with c4:
        date = st.date_input("JOURNEY DATE")
    with c5:
        airline = st.selectbox("AIRLINE", [
            "IndiGo", "Air India", "Jet Airways", "SpiceJet",
            "Multiple carriers", "GoAir", "Vistara"
        ])
    with c6:
        stops = st.selectbox("STOPS", ["Non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])

    # Row 3: Time
    st.markdown("<br>", unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        dep_time = st.time_input("DEPARTURE TIME")
    with c8:
        arr_time = st.time_input("ARRIVAL TIME")

    # Row 4: Slider
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<label style='color: #E6E6E6; font-size: 0.9rem; font-weight: 500;'>TOTAL DURATION</label>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        duration_hours = st.slider("Hours", 0, 24, 2)
    with dc2:
        duration_mins = st.slider("Minutes", 0, 59, 30)

    st.markdown("</div>", unsafe_allow_html=True) # End Card

    # ================= FEATURE ENGINEERING =================
    Journey_day = date.day
    Journey_month = date.month
    Dep_hour = dep_time.hour
    Dep_min = dep_time.minute
    Arrival_hour = arr_time.hour
    Arrival_min = arr_time.minute
    Total_Duration_Min = duration_hours * 60 + duration_mins

    Total_Stops_map = {"Non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    Total_Stops_val = Total_Stops_map[stops]
    Is_Weekend = 1 if date.weekday() >= 5 else 0

    input_data = {
        "Total_Stops": Total_Stops_val,
        "Journey_day": Journey_day,
        "Journey_month": Journey_month,
        "Dep_hour": Dep_hour,
        "Dep_min": Dep_min,
        "Arrival_hour": Arrival_hour,
        "Arrival_min": Arrival_min,
        "Duration_hours": duration_hours,
        "Duration_mins": duration_mins,
        "Total_Duration_Min": Total_Duration_Min,
        "Is_Weekend": Is_Weekend
    }

    airlines_list = ["Air India", "GoAir", "IndiGo", "Jet Airways", "Multiple carriers", "SpiceJet", "Vistara"]
    for a in airlines_list:
        input_data[f"Airline_{a}"] = 1 if airline == a else 0

    source_list = ["Chennai", "Delhi", "Kolkata", "Mumbai"]
    for s in source_list:
        input_data[f"Source_{s}"] = 1 if source == s else 0

    dest_list = ["Delhi", "Hyderabad", "Kolkata", "New Delhi"]
    for d in dest_list:
        input_data[f"Destination_{d}"] = 1 if destination == d else 0

    input_df = pd.DataFrame([input_data])

    # ================= PREDICTION BUTTON =================
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("CALCULATE PRICE"):
        with st.spinner("Processing..."):
            input_df = input_df.reindex(columns=model_features, fill_value=0)
            prediction = model.predict(input_df)[0]
            
            st.markdown(f"""
            <div class='result-card'>
                <p style='font-size: 1rem; opacity: 0.8; margin: 0;'>Estimated Fare</p>
                <h1 style='font-size: 3rem; margin: 10px 0;'>₹ {int(prediction):,}</h1>
                <p style='font-size: 0.9rem; opacity: 0.8;'>{airline} • {source} ➝ {destination}</p>
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: #555; font-size: 0.8rem;'>
    Designed by <b>Pratham</b> (B.Tech CSE)
</div>
""", unsafe_allow_html=True)