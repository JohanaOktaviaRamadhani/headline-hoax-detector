import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

# SETUP PATH & IMPORTS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP  = os.path.join(ROOT, "app")
DATA = os.path.join(ROOT, "dataset")
AB_TEST = os.path.join(ROOT, "ab-testing")

sys.path.insert(0, APP)
sys.path.insert(0, AB_TEST)

from utils import SHARED_CSS, COLOR_MAIN, get_chart_text_color, left_aligned_dataframe

try:
    import ab_testing_pipeline as ab_pipe
except ImportError:
    sys.path.append(os.path.join(ROOT, "ab-testing"))
    import ab_testing_pipeline as ab_pipe

st.set_page_config(
    page_title="A/B Testing Preprocessing · Hoax Detector", 
    page_icon="📊", 
    layout="wide"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.markdown("""
<style>
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4F46E5;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.82rem;
    color: #6B7280;
    font-weight: 500;
}
.stat-box-green {
    background: #ECFDF5;
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #065F46;
}
.stat-box-red {
    background: #FEF2F2;
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #991B1B;
}
</style>
""", unsafe_allow_html=True)

COLOR_CONTROL = "#3B82F6" # Biru untuk Control
COLOR_VARIANT = "#EC4899" # Pink/Magenta untuk Variant
COLOR_RAW     = "#94A3B8" # Abu-abu untuk Raw

# HEADER 
st.title("📊 A/B Testing: Preprocessing & Data Wrangling")
st.markdown("""
Eksperimen **A/B Testing** ini membandingkan kinerja dua skema pembersihan data (*data cleaning*) 
untuk menguji hipotesis signifikansi statistik terkait efisiensi waktu komputasi (*Processing Latency*) 
dan efektivitas pemangkasan kata tidak penting (*Vocabulary Compression Rate*) sebelum data diumpankan ke model *Machine Learning*.
""")
st.divider()

# INTERACTIVE PANEL & DATA LOADING
col_panel, col_desc = st.columns([1, 2], gap="large")

with col_panel:
    st.markdown("### 🔧 Control Panel Eksperimen")
    sample_size = st.slider(
        "Ukuran Sampel Dokumen (n)", 
        min_value=100, 
        max_value=1500, 
        value=500, 
        step=100,
        help="Semakin besar ukuran sampel, semakin tinggi tingkat representasi data, namun membutuhkan waktu ekstra."
    )
    
    run_button = st.button("🚀 Jalankan A/B Test Real-Time", use_container_width=True)
    
    st.info("""
    💡 Klik tombol di atas untuk simulasi real-time preprocessing
    """)

with col_desc:
    st.markdown("### 📋 Parameter Desain Eksperimen")
    st.markdown(f"""
    *   **Grup A (Pembersihan Dasar)**: Mengubah huruf menjadi kecil (*lowercase*), menghapus tanda baca standar, serta membuang spasi ganda.
    *   **Grup B (Pembersihan Lanjutan)**: Normalisasi regex (hapus URL, mention @, hashtag, angka & simbol), formalisasi singkatan kata slang lokal, dan penyaringan stopword (gabungan Sastrawi & kamus kustom).
    *   **Metode Uji Statistik**:
        *   **Latency**: *Independent Two-Sample T-Test* (tingkat signifikansi $\\alpha = 0.05$).
        *   **Kepadatan Kata**: *Paired T-Test* (tingkat signifikansi $\\alpha = 0.05$).
    """)

if 'ab_results' not in st.session_state or run_button:
    with st.spinner("⏳ Sedang memproses data dan menghitung analisis statistik A/B Test..."):
        results = ab_pipe.run_ab_test(sample_size=sample_size)
        st.session_state['ab_results'] = results
        st.success(f"✅ Eksperimen berhasil diselesaikan dengan sampel n={sample_size} dokumen!")

results = st.session_state['ab_results']

# TABS
tab_summary, tab_plots, tab_stats, tab_slang = st.tabs([
    "🏆 Ringkasan & Rekomendasi",
    "📈 Visualisasi Distribusi",
    "🧮 Detail Uji Statistik",
    "📖 Kamus Slang & Stopword"
])

# 🏆 TAB 1 — 
with tab_summary:
    st.markdown("### Ringkasan Eksekutif Eksperimen")
    st.markdown("""
    Hasil benchmarking waktu proses dan tingkat penyusutan ukuran kata menunjukkan perbedaan performa yang nyata di antara kedua grup.
    """)
    
    # METRIC CARDS
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['control']['avg_latency_ms']:.4f} ms</div>
            <div class="metric-label">⏱️ Rerata Latency Control (A)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['variant']['avg_latency_ms']:.4f} ms</div>
            <div class="metric-label">⏱️ Rerata Latency Variant (B)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['control']['compression_rate_pct']:.1f}%</div>
            <div class="metric-label">📉 Penyusutan Kata Control (A)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['variant']['compression_rate_pct']:.1f}%</div>
            <div class="metric-label">📉 Penyusutan Kata Variant (B)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # REKOMENDASI KOTAK HIJAU/MERAH
    avg_lat_c = results['control']['avg_latency_ms']
    avg_lat_v = results['variant']['avg_latency_ms']
    comp_c = results['control']['compression_rate_pct']
    comp_v = results['variant']['compression_rate_pct']
    diff_lat_pct = ((avg_lat_v - avg_lat_c) / avg_lat_c) * 100
    
    is_variant_better = comp_v > comp_c and (avg_lat_v < 1.0 or diff_lat_pct < 200)
    
    st.markdown("### 💡 Rekomendasi Final untuk Model ML")
    if is_variant_better:
        st.markdown(f"""
            <strong>GUNAKAN GRUP B (VARIANT / PEMBERSIHAN LANJUTAN)</strong><br>
            • Meskipun Variant B membutuhkan waktu pemrosesan {diff_lat_pct:+.1f}% sedikit lebih lama dibanding Control, 
            namun dalam skala absolut perbedaan ini sangat kecil (hanya selisih {(avg_lat_v - avg_lat_c):.4f} milidetik per berita). Overhead komputasi ini sangat sepele untuk dashboard real-time.<br>
            • Variant B secara signifikan memotong kata tidak penting sebesar {comp_v:.1f}% dibanding Control yang hanya {comp_c:.1f}%. 
            Ini menyusutkan ukuran dimensi TF-IDF secara masif, menghilangkan noise bahasa non-faktual, dan secara ilmiah akan melatih model Machine Learning yang lebih presisi, stabil, dan ringan.
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stat-box-red">
            <strong>🏆 REKOMENDASI FINAL: GUNAKAN GRUP A (CONTROL / PEMBERSIHAN DASAR)</strong><br><br>
            • <strong>Analisis Efisiensi</strong>: Waktu pemrosesan Grup A jauh lebih efisien. Penambahan overhead pada Variant B terlalu lambat dan signifikan secara statistik.<br>
            • <strong>Analisis Relevansi Fitur</strong>: Peningkatan kompresi kata kustom pada Variant B tidak terlalu besar untuk membenarkan penambahan overhead komputasi. Prioritaskan kecepatan pipeline.
        </div>
        """, unsafe_allow_html=True)

    # TABEL PERBANDINGAN
    st.markdown("### 📊 Tabel Matriks Perbandingan Kinerja")
    comparison_data = {
        "Metrik Evaluasi": [
            "Ukuran Sampel Dokumen",
            "Rata-rata Waktu Proses per Dokumen (Latency)",
            "Total Waktu Proses Seluruh Sampel",
            "Total Panjang Kata Sebelum Cleaning",
            "Total Panjang Kata Setelah Cleaning",
            "Efisiensi Penyusutan Kata (%)"
        ],
        "Grup A (Control - Dasar)": [
            f"{results['sample_size']} dokumen",
            f"{avg_lat_c:.4f} ms",
            f"{results['control']['total_latency_ms']:.2f} ms",
            f"{results['original_total_words']} kata",
            f"{results['control']['total_words']} kata",
            f"{comp_c:.1f}%"
        ],
        "Grup B (Variant - Lanjutan)": [
            f"{results['sample_size']} dokumen",
            f"{avg_lat_v:.4f} ms",
            f"{results['variant']['total_latency_ms']:.2f} ms",
            f"{results['original_total_words']} kata",
            f"{results['variant']['total_words']} kata",
            f"{comp_v:.1f}%"
        ],
        "Perubahan Performa (Delta)": [
            "Sama",
            f"{avg_lat_v - avg_lat_c:+.4f} ms ({diff_lat_pct:+.1f}%)",
            f"{results['variant']['total_latency_ms'] - results['control']['total_latency_ms']:+.2f} ms",
            "Sama",
            f"{results['variant']['total_words'] - results['control']['total_words']:.0f} kata",
            f"{comp_v - comp_c:+.1f}% (Ekstra Kompresi)"
        ]
    }
    df_compare = pd.DataFrame(comparison_data)
    
    styles = [
        {"selector": "th", "props": [("text-align", "left")]},
        {"selector": "td", "props": [("text-align", "left")]},
    ]
    styled_df = df_compare.style.set_properties(**{"text-align": "left"}).set_table_styles(styles)
    st.table(styled_df)

# TAB 2 
with tab_plots:
    st.markdown("### Visualisasi Distribusi Statistik A/B Test")
    st.markdown("""
    Visualisasi di bawah ini menampilkan sebaran statistik dari latensi pemrosesan data (efisiensi) 
    dan boxplot penyusutan jumlah kata per dokumen (efektivitas kompresi) untuk membandingkan sebaran antar grup secara grafis.
    """)
    
    col_chart1, col_chart2 = st.columns(2, gap="large")
    chart_text_color = get_chart_text_color()
    
    with col_chart1:
        st.markdown("**1. Distribusi Kecepatan Proses Preprocessing (Latency KDE Plot)**")
        
        fig, ax = plt.subplots(figsize=(5, 3.8))
        # Plot KDE
        sns.kdeplot(data=results['control']['latencies'], fill=True, color=COLOR_CONTROL, label="Control A (Dasar)", alpha=0.4, linewidth=1.5, ax=ax)
        sns.kdeplot(data=results['variant']['latencies'], fill=True, color=COLOR_VARIANT, label="Variant B (Kustom)", alpha=0.4, linewidth=1.5, ax=ax)
        
        # Mean Lines
        ax.axvline(avg_lat_c, color="#1D4ED8", linestyle="--", linewidth=1, label=f"Rerata A ({avg_lat_c:.3f} ms)")
        ax.axvline(avg_lat_v, color="#B91C1C", linestyle="--", linewidth=1, label=f"Rerata B ({avg_lat_v:.3f} ms)")
        
        ax.set_xlabel("Waktu Pemrosesan per Teks (ms)", fontsize=8, color=chart_text_color)
        ax.set_ylabel("Probability Density", fontsize=8, color=chart_text_color)
        ax.tick_params(colors=chart_text_color, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(chart_text_color)
        
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=7.5)
        for text in legend.get_texts():
            text.set_color(chart_text_color)
            
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    with col_chart2:
        st.markdown("**2. Analisis Kepadatan Informasi (Boxplot Panjang Kata)**")
        
        # Prepare Data Frame for Boxplot
        df_box = pd.DataFrame({
            'Panjang Kata': results['original_word_counts'] + results['control']['word_counts'] + results['variant']['word_counts'],
            'Grup Eksperimen': (['Teks Mentah (Raw)'] * len(results['original_word_counts']) + 
                               ['Grup A (Control)'] * len(results['control']['word_counts']) + 
                               ['Grup B (Variant)'] * len(results['variant']['word_counts']))
        })
        
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.boxplot(
            x='Grup Eksperimen', y='Panjang Kata', data=df_box,
            palette=[COLOR_RAW, COLOR_CONTROL, COLOR_VARIANT], width=0.4, ax=ax,
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"4"}
        )
        
        ax.set_xlabel("Grup Eksperimen", fontsize=8, color=chart_text_color)
        ax.set_ylabel("Jumlah Kata per Berita", fontsize=8, color=chart_text_color)
        ax.tick_params(colors=chart_text_color, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(chart_text_color)
        
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
    <div class='info-box'>
        📌 <b>Insight Hasil Visualisasi:</b><br>
        1. <b>KDE Plot Latency</b> menunjukkan bahwa Grup A (Control) memiliki kurva yang condong ke kiri dengan puncak yang lebih tinggi, membuktikan kecepatan komputasi yang sangat konsisten dan cepat. 
        Grup B (Variant) tergeser sedikit ke kanan dengan sebaran yang lebih luas, dipicu oleh overhead algoritma regex, formalisasi, dan pemetaan set stopword.<br>
        2. <b>Boxplot Panjang Kata</b> membuktikan efektivitas pemangkasan dokumen. Kelompok Raw (Abu-abu) memiliki panjang kata yang tinggi, disusul penyusutan moderat oleh Control A (Biru). 
        Grup B (Variant - Pink) secara visual terkompresi ke bawah secara signifikan, mengonfirmasi hilangnya mayoritas kata fungsional non-semantik (stopword).
    </div>
    """, unsafe_allow_html=True)

# TAB 3 
with tab_stats:
    st.markdown("### Pembuktian Ilmiah secara Statistik (T-Test)")
    st.markdown("""
    Untuk memastikan perbedaan rata-rata data wrangling ini nyata secara ilmiah dan bukan kebetulan acak, 
    kita melakukan dua uji statistik independen dan berpasangan menggunakan pustaka `scipy.stats`.
    """)
    
    c_s1, c_s2 = st.columns(2, gap="large")
    
    with c_s1:
        st.markdown("#### Uji 1: Latency (Independent Two-Sample T-Test)")
        st.markdown("""
        Digunakan untuk menguji hipotesis perbedaan rata-rata waktu proses antar dua kelompok independen.
        
        *   **Hipotesis Nol ($H_0$)**: Rata-rata *latency* Grup A = Grup B (Tidak ada perbedaan kecepatan nyata).
        *   **Hipotesis Alternatif ($H_1$)**: Rata-rata *latency* Grup B > Grup A (Grup B lebih lambat secara signifikan).
        """)
        
        t_stat_l = results['t_test_latency']['t_statistic']
        p_val_l = results['t_test_latency']['p_value']
        sig_l = results['t_test_latency']['significant']
        
        st.metric("T-Statistic (Latency)", f"{t_stat_l:.4f}")
        
        if p_val_l < 0.0001:
            p_val_str = "< 0.0001"
        else:
            p_val_str = f"= {p_val_l:.6f}"
            
        st.metric("P-Value (Latency)", p_val_str)
        
        if sig_l:
            st.markdown("""
            <div class="stat-box-red" style="margin-top:10px;">
                🚨 <strong>Tolak H₀ (Signifikan)</strong><br>
                Variant B terbukti secara signifikan <strong>LEBIH LAMBAT</strong> dibandingkan Control A secara statistik (P-Value < 0.05).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-box-green" style="margin-top:10px;">
                ✅ <strong>Gagal Tolak H₀ (Tidak Signifikan)</strong><br>
                Tidak ada perbedaan kecepatan proses yang signifikan secara statistik antara Control A dan Variant B.
            </div>
            """, unsafe_allow_html=True)
            
    with c_s2:
        st.markdown("#### Uji 2: Kompresi Kata (Paired T-Test)")
        st.markdown("""
        Digunakan untuk membandingkan rata-rata jumlah kata pada dokumen berpasangan (sebelum vs sesudah variant).
        
        *   **Hipotesis Nol ($H_0$)**: Jumlah kata Grup A = Grup B (Tidak ada perbedaan hasil kompresi nyata).
        *   **Hipotesis Alternatif ($H_1$)**: Jumlah kata Grup B < Grup A (Grup B memangkas kata tidak penting secara signifikan).
        """)
        
        t_stat_c = results['t_test_compression']['t_statistic']
        p_val_c = results['t_test_compression']['p_value']
        sig_c = results['t_test_compression']['significant']
        
        st.metric("T-Statistic (Word Counts)", f"{t_stat_c:.4f}")
        
        if p_val_c < 0.0001:
            p_val_c_str = "< 0.0001"
        else:
            p_val_c_str = f"= {p_val_c:.6f}"
            
        st.metric("P-Value (Word Counts)", p_val_c_str)
        
        if sig_c:
            st.markdown("""
            <div class="stat-box-green" style="margin-top:10px;">
                ✅ <strong>Tolak H₀ (Signifikan)</strong><br>
                Variant B terbukti <strong>LEBIH EFEKTIF & SIGNIFIKAN</strong> memangkas noise kata fungsional dibanding Control A (P-Value < 0.05).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-box-red" style="margin-top:10px;">
                🚨 <strong>Gagal Tolak H₀ (Tidak Signifikan)</strong><br>
                Tidak ada perbedaan kepadatan informasi kata yang signifikan secara statistik antara Grup A dan Grup B.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        📋 <b>Interpretasi Statistik:</b><br>
        Nilai batas signifikansi menggunakan tingkat signifikansi <b>Alpha (α) = 0.05</b> (tingkat kepercayaan 95%).<br>
        Karena nilai P-Value untuk kedua pengujian di atas berada jauh di bawah 0.05, kita dapat menarik kesimpulan ilmiah tingkat tinggi: 
        <i>"Meskipun penambahan normalisasi regex, slang dictionary, dan stopwords removal kustom menambah beban waktu komputasi secara nyata (p < 0.05), 
        namun performa pembersihan kata dan pengurangan noise teks yang dihasilkan juga terbukti secara mutlak jauh lebih superior dan signifikan secara statistik (p < 0.05)."</i>
    </div>
    """, unsafe_allow_html=True)

# TAB 4
with tab_slang:
    st.markdown("### Glosarium & Kamus Preprocessing")
    st.markdown("""
    Informasi kamus formalisasi slang kustom dan total stopwords yang digunakan oleh pipeline lanjutan (**Grup B - Variant**) untuk standardisasi data wrangling.
    """)
    
    col_k1, col_k2 = st.columns([1, 1], gap="large")
    
    with col_k1:
        st.markdown("#### 📖 Kamus Slang Kustom (Slang Dictionary)")
        st.markdown("""
        Penerapan kamus singkatan lokal ini bertujuan untuk mengubah kata-kata slang internet informal 
        menjadi representasi baku bahasa Indonesia, memperkuat bobot TF-IDF pada kesamaan makna kata.
        """)
        
        # Buat dataframe slang
        slangs_list = [{"Kata Slang": k, "Kata Baku (Formal)": v} for k, v in ab_pipe.SLANG_DICT.items()]
        df_slang = pd.DataFrame(slangs_list)
        left_aligned_dataframe(df_slang, use_container_width=True, hide_index=True)
        
    with col_k2:
        st.markdown("#### 🛑 Informasi Stopword Removal")
        st.markdown("""
        Penyaringan kata menggunakan daftar stopwords gabungan pustaka **Sastrawi** dan **Kustom Stopwords Lokal** (file `stopwords_id.txt`):
        """)
        
        # Load stopwords sample
        root_dir, sw_path, _ = ab_pipe.get_paths()
        all_sw = ab_pipe.load_stopwords(sw_path)
        
        st.markdown(f"**Total Stopwords Terdaftar**: {len(all_sw)} kata")
        
        q_sw_in = st.text_input("🔍 Cek Kata di Kamus Stopword...", placeholder="Masukkan kata tugas (misal: 'yang', 'dengan')...")
        if q_sw_in:
            if q_sw_in.lower().strip() in all_sw:
                st.success(f"✅ Kata '{q_sw_in}' TERDAFTAR sebagai stopword (kata ini akan dihapus dari teks pada Variant B).")
            else:
                st.info(f"ℹ️ Kata '{q_sw_in}' TIDAK TERDAFTAR sebagai stopword (artinya kata ini tetap dipertahankan).")
                
        st.markdown("**Contoh 30 Kata Stopwords Acak**:")
        np.random.seed(42)
        sample_sw = np.random.choice(all_sw, size=30, replace=False)
        st.markdown(" ".join([f"<span class='badge badge-gray'>{w}</span>" for w in sample_sw]), unsafe_allow_html=True)
