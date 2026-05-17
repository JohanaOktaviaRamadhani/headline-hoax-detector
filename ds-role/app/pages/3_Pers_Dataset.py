import streamlit as st
import pandas as pd
import os

# CONFIG
st.set_page_config(page_title="Pers Dataset", page_icon="🏛️", layout="wide")
st.title("🏛️ Pers Dataset")

# PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT, "dataset")

# LOAD DATA
@st.cache_data
def load_data():
    raw_path = os.path.join(DATA_DIR, "raw", "dewan_pers.csv")
    final_path = os.path.join(DATA_DIR, "final", "final-pers.csv")

    df_raw = pd.read_csv(raw_path)
    df_final = pd.read_csv(final_path)
    return df_raw, df_final
df_raw, df_final = load_data()

# METRICS
raw_count = len(df_raw)
final_count = len(df_final)

# TABS
tab_info, tab_preprocess = st.tabs([
    "📋 Info Dataset",
    "⚙️ Preprocessing"
])

# TAB 1 — INFO DATASET
with tab_info:
    st.markdown("### 📋 Info Dataset")
    st.markdown("""
    Dataset pers ini dibangun menggunakan data hasil scraping dari situs resmi Dewan Pers sebagai acuan media terpercaya. 
    Untuk menjamin kualitasnya, data telah melalui serangkaian tahap praproses (preprocessing), meliputi penyaringan (filtering) jenis media, 
    retensi hanya pada media yang terverifikasi, seleksi atribut relevan, serta pembersihan data dari nilai kosong (missing values) dan duplikasi.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📥 Data Raw", raw_count)
    with col2:
        st.metric("📦 Data Final", final_count)
    with col3:
        st.markdown("""
        **🌐 Sumber Data**  
        Scraping Dewan Pers  
        https://datapers.dewanpers.or.id/site/iframe-verified
        """)

    st.markdown("#### Contoh Data Final")
    preview = df_final.head(10)
    st.dataframe(preview, use_container_width=True, hide_index=True)

# TAB 2 — PREPROCESSING
with tab_preprocess:
    st.markdown("### ⚙️ Pipeline Preprocessing Dataset Pers")
    st.markdown("""
    Pipeline preprocessing dilakukan untuk menyaring dan membersihkan data hasil scraping 
    sehingga hanya menyisakan entitas media yang relevan dan valid. 
    Setiap tahapan berfokus pada reduksi noise serta peningkatan konsistensi struktur data.
    """)

    # CHIP HELPER
    def render_chips(items, color="#F3F4F6", text_color="#111827"):
        return " ".join([
            f"<span style='background:{color};color:{text_color};padding:4px 8px;border-radius:6px;margin:2px;display:inline-block;font-size:0.8rem'>{item}</span>"
            for item in items
        ])

    # PIPELINE STEPS
    steps = [
        ("Load Data",
        "Menggunakan dataset hasil scraping Dewan Pers dengan total <b>2640 baris data</b>."),
        
        ("Filter Jenis Media",
        render_chips(["jenis_media", "=", "Siber"], "#E0F2FE", "#0369A1") +
        "<br>Hanya mempertahankan media berbasis siber."),
        
        ("Filter Media Terverifikasi",
        render_chips(["status", "=", "Terverifikasi Administratif dan Faktual"], "#DCFCE7", "#15803D") +
        "<br>Menyaring media yang telah terverifikasi secara administratif dan faktual oleh Dewan Pers."),
        
        ("Seleksi Kolom",
        render_chips(["nama_media", "website"], "#FEF3C7", "#92400E") +
        "<br>Mengambil atribut utama yang relevan."),
        
        ("Handling Missing Value",
        "Menghapus <b>58 baris</b> yang memiliki nilai kosong untuk menjaga kualitas data."),
        
        ("Remove Duplicate",
        "Menghapus <b>0 data duplikat</b> untuk memastikan setiap entitas unik."),
        
        ("Final Dataset",
        "Dataset akhir berisi <b>875 media siber terverifikasi</b> yang telah bersih dan siap digunakan.")
    ]
    # RENDER CARD
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style='display:flex;gap:12px;padding:12px;border:1px solid #E5E7EB;border-radius:10px;margin-bottom:8px'>
            <div style='font-weight:bold;color:#6366F1'>{i}</div>
            <div>
                <div style='font-weight:600;color:#111827;margin-bottom:4px'>{title}</div>
                <div style='color:#6B7280;font-size:0.9rem'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
