# apps/app.py
import sys
import os
from modules.recommend_page import recommend_page
from modules.home_page import home_page
from modules.chatbot_ui import chatbot_ui
from modules.model_page import model_page
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add parent dir to path

import streamlit as st
from streamlit_option_menu import option_menu
from modules.dataset_upload import upload_dataset
from modules.eda_page import eda_page
# -----------------------
# APP CONFIGURATION
# -----------------------
st.set_page_config(
    page_title="AI-Driven Recommendation System",
    layout="wide"
)

# -----------------------
# SIDEBAR MENU
# -----------------------
with st.sidebar:
    st.title("AI Recommendation System")
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Dataset", "EDA","Model", "Recommendation"],
        icons=["house", "database","eye", "cpu", "stars"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    st.write("**Theme:** 🌞 Light / 🌙 Dark")
    st.write("Developed by [Supriya Naik](https://www.linkedin.com/in/supriya-naik-a4ba47193)")
    st.markdown("© 2025 All Rights Reserved")

# -----------------------
# HOME PAGE
# -----------------------
if selected == "Home":
    home_page()

# -----------------------
# DATASET PAGE
# -----------------------
if selected == "Dataset":
    upload_dataset()  # <- call the function

# -----------------------
# MODEL PAGE
# -----------------------

elif selected == "Model":
    model_page()  # call the new Model page module
    
# ----------------------
# EDA
# ---------------------
   
elif selected == "EDA":
    eda_page()

# -----------------------
# RECOMMENDATION PAGE
# -----------------------
elif selected == "Recommendation":
    recommend_page()