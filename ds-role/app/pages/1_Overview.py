import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import SHARED_CSS, COLOR_MAIN, COLOR_MENDUKUNG, COLOR_MEMBANTAH

st.set_page_config(page_title="Overview · Hoax Detector", page_icon="🔍", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────
st.markdown("## 🔍 Hoax Detector — Project Overview")
st.markdown(
    "<div class='info-box'>"
    "Proyek ini bertujuan membangun sistem deteksi hoaks berbasis data dari berbagai sumber berita online Indonesia. "
    "Pendekatan yang digunakan mencakup analisis stance, klasifikasi teks, dan teknik NLP bahasa Indonesia."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── PROBLEM STATEMENT ───────────────────────────────────
st.markdown("<div class='section-title'>🎯 Problem Statement</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("""
    <div class='step-card'>
        <div class='step-num'>1</div>
        <div>
            <div style='font-weight:600;color:#111827;margin-bottom:4px'>Permasalahan Utama</div>
            <div style='color:#6B7280;font-size:0.88rem'>
                Penyebaran informasi hoaks di media digital Indonesia semakin masif,
                terutama saat momen politik seperti pilkada. Diperlukan sistem otomatis
                untuk mendeteksi dan mengklasifikasikan stance artikel berita.
            </div>
        </div>
    </div>
    <div class='step-card'>
        <div class='step-num'>2</div>
        <div>
            <div style='font-weight:600;color:#111827;margin-bottom:4px'>Solusi yang Dikembangkan</div>
            <div style='color:#6B7280;font-size:0.88rem'>
                Membangun pipeline NLP end-to-end: dari pengumpulan data, preprocessing,
                feature engineering, hingga model klasifikasi berbasis machine learning
                untuk mendeteksi stance artikel.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='step-card'>
        <div class='step-num'>3</div>
        <div>
            <div style='font-weight:600;color:#111827;margin-bottom:4px'>Business Questions</div>
            <div style='color:#6B7280;font-size:0.88rem'>
                <ol style='margin:0;padding-left:16px'>
                    <li>Apa pola linguistik yang membedakan artikel mendukung vs membantah?</li>
                    <li>Fitur teks mana yang paling informatif untuk klasifikasi stance?</li>
                    <li>Bagaimana distribusi sentimen antar kelas stance?</li>
                </ol>
            </div>
        </div>
    </div>
    <div class='step-card'>
        <div class='step-num'>4</div>
        <div>
            <div style='font-weight:600;color:#111827;margin-bottom:4px'>Metodologi</div>
            <div style='color:#6B7280;font-size:0.88rem'>
                Gathering Data → Assessing → Cleaning → EDA →
                Feature Engineering → Modeling → Dashboard
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#  DATASET OVERVIEW 
st.markdown("<div class='section-title'>📦 Dataset yang Digunakan</div>", unsafe_allow_html=True)

# Stance dataset card
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("""
    <div class='dataset-card'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
            <span style='font-size:1.5rem'>🗣️</span>
            <div>
                <div class='dataset-card-title'>Stance Dataset</div>
                <span class='badge badge-green'>✅ Tersedia</span>
                <span class='badge badge-blue'>Aktif Digunakan</span>
            </div>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Sumber</span>
            <span class='dict-desc'>Kompasiana</span>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Topik</span>
            <span class='dict-desc'>Pilkada DKI 2017 (Kasus Ahok)</span>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Raw Data</span>
            <span class='dict-desc'>402 baris</span>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Data Final</span>
            <span class='dict-desc'><b>300 baris</b> (102 unknown dibuang)</span>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Kelas</span>
            <span class='dict-desc'>Mendukung (177) · Membantah (123)</span>
        </div>
        <div class='dict-row'>
            <span class='dict-col'>Kolom</span>
            <span class='dict-desc'>8 kolom (setelah preprocessing)</span>
        </div>
        <div style='margin-top:12px'>
            <a href='/Stance_Dataset' target='_self'>
                <button style='background:#6366F1;color:white;border:none;border-radius:8px;
                padding:7px 16px;font-size:0.85rem;font-weight:600;cursor:pointer'>
                    → Lihat Detail
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class='dataset-card' style='border: 2px dashed #E5E7EB; background:#F9FAFB'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
            <span style='font-size:1.5rem'>📰</span>
            <div>
                <div class='dataset-card-title' style='color:#9CA3AF'>Dataset Lain</div>
                <span class='badge badge-gray'>🔜 Coming Soon</span>
            </div>
        </div>
        <div style='color:#9CA3AF;font-size:0.88rem;margin-top:12px;text-align:center;padding:24px 0'>
            Dataset berikutnya akan ditambahkan di sini.<br>
            <span style='font-size:0.8rem'>Halaman 3 akan aktif setelah dataset tersedia.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# PIPELINE OVERVIEW 
st.markdown("<div class='section-title'>🔄 Pipeline Proyek</div>", unsafe_allow_html=True)

pipeline = [
    ("📥", "Gathering Data", "Scraping & pengumpulan dataset dari sumber publik", COLOR_MAIN),
    ("🔎", "Assessing Data", "Evaluasi kualitas, cek duplikat & missing value", "#8B5CF6"),
    ("🧹", "Cleaning Data", "Preprocessing teks: cleaning, slang, stopword, stemming", "#EC4899"),
    ("📊", "EDA", "Exploratory Data Analysis & visualisasi distribusi data", "#F59E0B"),
    ("⚙️", "Feature Engineering", "Ekstraksi fitur numerik dari teks (pos/neg count, sentiment)", "#10B981"),
    ("🤖", "Modeling", "Klasifikasi stance menggunakan machine learning", "#3B82F6"),
    ("📈", "Dashboard", "Visualisasi interaktif hasil analisis", "#6366F1"),
]

cols = st.columns(len(pipeline))
for col, (icon, title, desc, color) in zip(cols, pipeline):
    with col:
        st.markdown(f"""
        <div style='text-align:center;padding:14px 8px;background:#FFFFFF;
                    border:1px solid #E5E7EB;border-radius:12px;
                    border-top:3px solid {color}'>
            <div style='font-size:1.4rem;margin-bottom:6px'>{icon}</div>
            <div style='font-weight:700;font-size:0.82rem;color:#1F2937;margin-bottom:4px'>{title}</div>
            <div style='font-size:0.75rem;color:#6B7280;line-height:1.4'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
