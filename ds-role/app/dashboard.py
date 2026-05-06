import streamlit as st
import os, sys

# PATH SETUP
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from utils import SHARED_CSS

st.set_page_config(
    page_title="Hoax Detector Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# HEADER
st.title("📊 Hoax Detector Dashboard")
st.markdown("Overview seluruh dataset dan insight utama.")

# QUICK NAV
st.markdown("### 🚀 Quick Navigation")
col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_Stance_Dataset.py", label="🗣️ Stance Dataset")

with col2:
    st.page_link("pages/2_Hoax_Fact_Clickbait.py", label="📰 Hoax / Fact / Clickbait")

with col3:
    st.page_link("pages/3_Pers_Dataset.py", label="🏛️ Pers Dataset")

st.markdown("---")

st.info("Gunakan menu di sidebar atau quick navigation untuk eksplor dataset.")