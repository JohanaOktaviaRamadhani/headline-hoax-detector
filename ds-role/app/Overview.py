import streamlit as st
import os, sys

# PATH SETUP
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from utils import SHARED_CSS

st.set_page_config(
    page_title="Overview - Hoax Detector ",
    page_icon="📊",
    layout="wide"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# HEADER
st.title("🔍 Hoax Detector — Project Overview")
st.markdown("""
Proyek ini berfokus pada pengembangan sistem deteksi hoaks berbasis data yang mengolah berbagai sumber berita online di Indonesia. 
Pendekatan yang digunakan mengkombinasikan analisis stance, klasifikasi teks, serta teknik Natural Language Processing (NLP) 
yang disesuaikan dengan karakteristik bahasa Indonesia.
""")
st.divider()

# MAIN CONTENT (2 COLUMN)
col1, col2 = st.columns([1, 1], gap="large")

# LEFT: PROBLEM & SOLUTION
with col1:
    st.markdown("### ⚠️ Permasalahan Utama")
    st.markdown("""
    Penyebaran hoaks di ekosistem media digital Indonesia mengalami eskalasi signifikan, 
    terutama pada periode sensitif seperti pilkada. Volume informasi yang tinggi membuat 
    proses verifikasi manual menjadi tidak scalable.

    👉 Dibutuhkan sistem otomatis yang mampu mengidentifikasi dan mengklasifikasikan 
    stance artikel berita secara konsisten dan efisien.
    """)

    st.markdown("### 🛠️ Solusi yang Dikembangkan")
    st.markdown("""
    Solusi yang dibangun berupa pipeline NLP end-to-end yang terstruktur:

    - Data Gathering (multi-source news)
    - Data Preprocessing (cleaning & normalization)
    - Feature Engineering (text representation)
    - Machine Learning Model (stance classification)

    🎯 Fokus utama: menghasilkan sistem yang robust, scalable, dan data-driven.
    """)

# RIGHT: BUSINESS QUESTIONS
with col2:
    st.markdown("### 📊 Business Questions")

    st.markdown("""
    Untuk memastikan solusi memberikan nilai strategis, analisis difokuskan pada:
    """)
    st.markdown("""
    **1. Pola Linguistik**  
    Apa pola linguistik yang membedakan artikel mendukung vs membantah?
    **2. Feature Importance**  
    Fitur teks mana yang paling informatif dalam meningkatkan performa model?
    **3. Distribusi Sentimen**  
    Bagaimana distribusi sentimen dalam tiap kelas stance, dan apakah ada korelasi yang bisa dimanfaatkan?
    """)
