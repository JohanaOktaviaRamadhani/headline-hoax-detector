import streamlit as st
import os, sys

# PATH SETUP
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from utils import SHARED_CSS

st.set_page_config(
    page_title="Overview - Hoax Detector",
    page_icon="📊",
    layout="wide"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# HEADER & DESKRIPSI UTAMA
st.title("🔍 Hoax Detector — Project Overview")
st.markdown("""
Proyek ini membangun sistem deteksi hoaks berita Indonesia secara otomatis berbasis NLP. 
User cukup memasukkan satu headline, sistem akan melakukan scraping artikel terkait, 
menganalisis clickbait, stance, kepercayaan sumber, hingga menghasilkan persentase kemungkinan hoax secara end-to-end.
""")
st.divider()

# MAIN CONTENT (2 COLUMN)
col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN: PROBLEM, SOLUTION, & METHODOLOGY
with col1:
    st.markdown("### ⚠️ Permasalahan Utama")
    st.markdown("""
    Verifikasi berita secara manual membutuhkan waktu lama dan keahlian khusus, sementara penyebaran hoaks di 
    Indonesia terus meningkat ,terutama pada momen sensitif seperti pilkada dan isu kesehatan publik.

    👉 Dibutuhkan sistem otomatis yang bisa memproses satu headline dan langsung memberikan penilaian 
    apakah berita tersebut hoax atau tidak secara cepat dan terukur.
    """)

    # Memberikan jeda vertikal tambahan antara Permasalahan dan Solusi
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔧 Solusi yang Dikembangkan")
    st.markdown("""
    Pipeline otomatis end-to-end berbasis NLP:
    - User memasukkan 1 headline berita
    - Sistem scraping >1 artikel terkait secara otomatis (menghasilkan kolom: title, source, url, tgl publish, isi, snippet)
    - Setiap artikel dicek clickbait atau tidak → Artikel clickbait langsung dieliminasi
    - Artikel yang lolos dicek stance-nya (Mendukung atau Membantah)
    - Semua artikel (mendukung & membantah) dicek kepercayaan domain-nya menggunakan dataset Pers
    - Dihitung Cosine Similarity antara headline input dengan isi setiap artikel yang tersisa

    🎯 Semakin banyak artikel Membantah dengan similarity tinggi dari sumber terpercaya → semakin tinggi kemungkinan hoax.
    """)
    
    # Memberikan jeda vertikal sebelum Metodologi
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🔄 Metodologi")
    st.info("""
    Gathering Data → Preprocessing → Feature Engineering → Clickbait Detection → 
    Stance Classification → Trusted Domain Check → Cosine Similarity → 
    Hoax Probability Output → Dashboard
    """)

# RIGHT COLUMN: BUSINESS QUESTIONS & OUTPUT SYSTEM
with col2:
    st.markdown("### 📊 Business Questions")
    st.markdown("""
    1. Clickbait Detection  
       Apakah artikel-artikel terkait menggunakan judul yang menyesatkan (clickbait)?
    2. Stance Analysis  
       Dari artikel yang lolos filter clickbait, apakah lebih banyak yang mendukung atau membantah headline yang diuji?
    3. Source Credibility  
       Apakah domain sumber artikel termasuk media terpercaya?
    4. Hoax Probability  
       Berapa persentase kemungkinan headline ini adalah hoax berdasarkan agregasi stance, similarity, dan kredibilitas sumber?
    """)

    # Memberikan jeda vertikal tambahan agar sejajar secara visual dengan kolom kiri
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📤 Output Sistem")
    st.markdown("""
    Sistem menghasilkan dua output utama:

    1. Daftar Berita Terkait  
       Tabel artikel yang telah melewati seluruh tahap filter, berisi:
        
       • Judul berita  
       • Sumber / domain media penerbit  
       • Skor Cosine Similarity terhadap headline  
       • Status stance: Mendukung atau Membantah  
       • Link langsung ke artikel asli  
       
       Dilengkapi filter by stance (Semua / Membantah / Mendukung) dan sorting berdasarkan similarity tertinggi.

    2. Persentase Kemungkinan Hoax  
       Kalkulasi akhir berdasarkan proporsi artikel Membantah yang lolos filter clickbait dan berasal dari domain terpercaya, dibandingkan dengan total artikel.
    """)