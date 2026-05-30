import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
APP  = os.path.join(ROOT, "app")
DATA = os.path.join(ROOT, "dataset")

sys.path.append(APP)
from utils import SHARED_CSS, COLOR_MAIN, get_chart_text_color

COLOR_FAKTA = "#10B981" 
COLOR_HOAX  = "#EF4444"  

st.set_page_config(page_title="Hoax & Fake Dataset · Detector", page_icon="📰", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# LOADERS
@st.cache_data
def load_data():
    path = os.path.join(DATA, "final", "final-data-hoax.csv")
    df = pd.read_csv(path)
    
    target_col = None
    for col in ["full_text", "text", "konten", "content"]:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = df.columns[0]
        st.warning(f"⚠️ Kolom 'clean_text' tidak ditemukan. Menggunakan kolom alternatif: '{target_col}'")
    
    df["clean_text"] = df[target_col]
    if "label_text" not in df.columns:
        label_col = "label" if "label" in df.columns else df.columns[-1] 
        df["label_text"] = df[label_col].map({1: "Fakta/Asli", 0: "Hoaks/Fake"})
        
    if "text_length" not in df.columns:
        df["text_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))
        
    return df

@st.cache_data
def load_words(filename):
    path = os.path.join(DATA, "raw", filename)
    try:
        with open(path, encoding="utf-8") as f:
            return [w.strip() for w in f.readlines() if w.strip()]
    except FileNotFoundError:
        return []

# LOAD DATA & KAMUS
df = load_data()
stop_words = load_words("stopwords_id.txt")

# HEADER
st.title("📰 Hoax & Fake News Dataset")
st.markdown("""
Dataset ini berisi kumpulan artikel berita yang digunakan untuk mendeteksi tingkat keaslian informasi 
(*Hoax vs Genuine News*). Analisis dilakukan melalui pipeline NLP secara end-to-end, mulai dari 
pembersihan teks (*cleaning*), normalisasi kata baku, penghapusan kata umum (*stopword*), 
hingga ekstraksi fitur struktural untuk membedakan karakteristik manipulasi bahasa pada berita hoaks.
""")

# TABS
tab_info, tab_kamus, tab_prep, tab_feat, tab_eda, tab_dict = st.tabs([
    "📋 Info Dataset",
    "📖 Kamus Stopword",
    "⚙️ Preprocessing",
    "🔧 Feature Engineering",
    "📈 EDA ",
    "📚 Data Dictionary",
])

# 🛠️ TAB 1 — INFO DATASET
with tab_info:
    st.markdown("### Ringkasan Eksekutif Dataset")
    st.markdown("""
    Sebelum masuk ke tahap pemodelan, mari kita lihat gambaran umum data yang kita gunakan. 
    Ringkasan di bawah ini menampilkan metrik penting dataset berita, mulai dari ukuran total, 
    efek dari proses pembersihan data (*duplikat & missing values*), hingga keseimbangan proporsi 
    antara berita fakta dan hoaks untuk memastikan kualitas serta konsistensi data.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📁 Nama File** : final_data_hoax.csv  

        **📊 Jumlah Kolom** : 7 Kolom utama  
        """)
    with col2:
        total_rows = len(df)
        st.markdown(f"""
        **📝 Raw Awal** : 1116 baris
                    
        **✅ Total Final** : {total_rows} baris  
        """)
    with col3:
        st.markdown("""
        **🗑️ Status Data** : Cleaned & Formatted  

        **🏷️ Kategori Kelas** : Fakta & Hoaks  
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Grid Grafik Distribusi + Ringkasan Text
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("**Distribusi Kelas Berita**")
        fakta_cnt = len(df[df["label_text"] == "Fakta/Asli"])
        hoax_cnt = len(df[df["label_text"] == "Hoaks/Fake"])
        chart_text_color = get_chart_text_color()
        
        fig, ax = plt.subplots(figsize=(3, 3))
        wedges, _ = ax.pie(
            [fakta_cnt, hoax_cnt],
            colors=[COLOR_FAKTA, COLOR_HOAX],
            startangle=90,
            radius=0.85,
            wedgeprops=dict(width=0.35, edgecolor="white")
        )
        ax.text(0, 0, f"{total_rows}", ha="center", va="center", fontsize=12, fontweight="bold", color=chart_text_color)
        legend = ax.legend(wedges, [f"Fakta ({fakta_cnt})", f"Hoaks ({hoax_cnt})"], loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(chart_text_color)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)
        plt.close()

    with col_right:
        st.markdown("**Ringkasan Statistik Tekstual**")
        st.markdown(f"""
        - **Total Berita**: {total_rows} artikel  
        - **Berita Fakta**: {fakta_cnt} ({fakta_cnt/total_rows*100:.1f}%)  
        - **Berita Hoaks**: {hoax_cnt} ({hoax_cnt/total_rows*100:.1f}%)  
        - **Rata-rata Panjang**: {df['text_length'].mean():.0f} kata per berita  
        - **Median Panjang**: {df['text_length'].median():.0f} kata  
        """)

    st.markdown("**Preview Berita Terproses**")
    preview = df[["clean_text", "label_text", "text_length"]].head(5).copy()
    preview["clean_text"] = preview["clean_text"].apply(lambda x: str(x)[:70] + "..." if len(str(x)) > 70 else str(x))
    st.dataframe(preview, use_container_width=True, hide_index=True)


# 🛠️ TAB 2 — KAMUS KATA (Stopwords)
with tab_kamus:
    st.markdown("### Kamus Stopwords")
    st.markdown("""
    Tahap pembersihan teks ini memanfaatkan daftar **Stopwords** untuk menghapus kata-kata umum yang sering muncul 
    namun tidak memiliki makna semantik yang kuat dalam membedakan berita hoaks dan fakta. 
    Penghapusan kata seperti *dan*, *yang*, atau *di* ini berfungsi untuk mengurangi dimensi data (*dimensionality reduction*) 
    sekaligus mengoptimalkan akurasi bobot teks pada tahap ekstraksi fitur **TF-IDF**.
    """)
    
    q_sw = st.text_input("🔍 Cari Kata...", placeholder="Masukkan kata yang ingin kamu cek di dalam daftar stopword...")
    if q_sw:
        filt_sw = [w for w in stop_words if q_sw.lower() in w.lower()]
    else:
        filt_sw = stop_words
        
    st.markdown(
        f"<small style='color:#6B7280; display:block; margin-bottom:8px;'>Menampilkan {min(len(filt_sw), 200)} dari {len(filt_sw)} kata ditemukan</small>",
        unsafe_allow_html=True
    )
    if filt_sw:
        st.markdown(" ".join([f"<span class='badge badge-gray'>{w}</span>" for w in filt_sw[:200]]), unsafe_allow_html=True)
    else:
        st.info("ℹ️ Kata yang kamu cari tidak terdaftar sebagai stopword (artinya kata ini dipertahankan dalam teks).")


# 🛠️ TAB 3 — PREPROCESSING PIPELINE
with tab_prep:
    st.markdown("### Pipeline Preprocessing Teks")
    st.markdown("""
    Proses pembersihan data dilakukan secara bertahap untuk mentransformasi dokumen berita mentah 
    menjadi teks bersih siap pakai, sekaligus memfilter artikel berdasarkan panjang kata dan validitas bahasa.
    """)

    def render_chips(items, color="#F3F4F6", text_color="#111827"):
        return " ".join([
            f"<span style='background:{color}; color:{text_color}; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block; font-size:0.8rem; font-weight:500;'>{item}</span>"
            for item in items
        ])

    steps = [
        ("Drop Unused Columns", 
         f"Menghapus kolom kosong bawaan file mentah yang tidak digunakan: {render_chips(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'], '#FEE2E2', '#991B1B')}"),
    
        ("Early Duplicate Removal", 
         "Mendeteksi dan menghapus 1 baris data duplikat awal untuk menjaga validitas sampel."),
        
        ("Column Renaming", 
         f"Menstandardisasi nama kolom utama menjadi huruf kecil: {render_chips(['Headline ➔ headline', 'Body ➔ article'], '#E0F2FE', '#0369A1')}"),
        
        ("Regex Text Normalization", 
         "Menerapkan fungsi `normalize()` pada teks *headline* dan *article* (mengubah ke huruf kecil, menghapus URL, *mention* @, hashtag #, angka, tanda baca, serta merapikan spasi ganda)."),
        
        ("Feature Creation (full_text)", 
         f"Menggabungkan teks judul dan isi berita menjadi satu kesatuan fitur baru: {render_chips(['headline', '+', 'article', '➔', 'full_text'], '#ECFDF5', '#065F46')}"),
        
        ("Secondary Duplicate Removal", 
         "Menghapus kembali **2 data duplikat baru** yang terdeteksi setelah kolom judul dan isi digabungkan."),
        
        ("Formalisasi Kata Slang", 
         f"Mengubah singkatan atau kata tidak baku internet menjadi kata formal: {render_chips(['yg ➔ yang', 'tdk ➔ tidak', 'dgn ➔ dengan', 'hoax ➔ hoaks', 'bkn ➔ bukan', 'tp ➔ tapi', 'klo ➔ kalau', 'ga/gak ➔ tidak', 'udah ➔ sudah'], '#FEEFDF', '#9A3412')}"),
        
        ("Stopword Removal", 
         "Membuang kata-kata umum menggunakan Sastrawi & kustom stopword, namun tetap memproteksi kata negasi dan kata kunci sentimen penting agar konteks berita tidak hilang."),
        
        ("Article Length Filtering", 
         "Menghitung jumlah kata (`article_len`) pada *full_text* lalu membuang artikel ekstrem yang terlalu pendek (di bawah 10 kata) atau terlalu panjang (di atas 1000 kata)."),
         
        ("Language Detection", 
         f"Mendeteksi bahasa menggunakan fungsi `detect_lang` dan hanya mempertahankan artikel dengan tag bahasa Indonesia {render_chips(['lang == id'], '#FAF5FF', '#6B21A8')} untuk menghindari *noise* bahasa asing.")
    ]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-card' style='display: flex; gap: 16px; padding: 14px; border: 1px solid #E5E7EB; border-radius: 10px; margin-bottom: 10px; background-color: #FAFAFA;'>
            <div class='step-num' style='background: #10B981; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;'>{i}</div>
            <div>
                <div style='font-weight:600; color:#111827; margin-bottom:4px; font-size: 1rem;'>{title}</div>
                <div style='color:#4B5563; font-size:0.88rem; line-height: 1.4;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

# 🛠️ TAB 4 — FEATURE ENGINEERING
with tab_feat:
    st.markdown("### Fitur Engineering")
    st.markdown("""
    Berikut adalah daftar kolom dan fitur hasil *preprocessing* serta *feature engineering* yang digunakan 
    sebagai input utama dalam sistem pemodelan *machine learning* untuk mendeteksi berita hoaks.
    """)
    
    # Menyesuaikan dengan data kolom terbaru kamu
    features = [
        ("full_text", 
         "String · Preprocessing", 
         "Gabungan dari kolom headline dan article yang telah dinormalisasi serta dilakukan formalisasi kata-kata slang."),
        
        ("label (encoded)", 
         "Integer · Preprocessing / Output", 
         "Representasi numerik akhir dari target menggunakan LabelEncoder untuk kebutuhan model machine learning (0: Hoax, 1: Valid)."),
        
        ("article_len", 
         "Integer · Feature Engineering", 
         "Jumlah kata (word count) yang dihitung dari kolom full_text untuk memfilter panjang artikel (antara 10 - 1000 kata)."),
        
        ("headline_len", 
         "Integer · Feature Engineering", 
         "Jumlah kata (word count) yang dihitung secara spesifik dari kolom teks judul (headline).")
    ]
    for name, desc, explain in features:
        st.markdown(f"""
        <div style='padding:12px 14px; border:1px solid #E5E7EB; border-radius:10px; margin-bottom:10px; background-color: #FAFAFA;'>
            <div style='font-family:monospace; font-weight:600; color:#10B981; font-size:1.05rem; margin-bottom:4px'>{name}</div>
            <div style='font-size:0.85rem; color:#4B5563; font-weight:500; margin-bottom:4px'>{desc}</div>
            <div style='font-size:0.85rem; color:#6B7280; line-height:1.4;'>{explain}</div>
        </div>
        """, unsafe_allow_html=True)


# 🛠️ TAB 5 — EDA (EXPLORATORY DATA ANALYSIS)
with tab_eda:
    st.markdown("### Eksplorasi Data Analisis (EDA)")
    st.markdown("""
    Secara keseluruhan, visualisasi ini dibuat untuk memudahkan kita memahami karakteristik unik dari masing-masing data. 
    Melalui grafik di atas, terlihat jelas adanya perbedaan pola antara berita asli dan hoaks—terutama dari segi panjang teksnya.             
    Perbedaan karakteristik seperti inilah yang nantinya membantu kita (dan model komputer) untuk membedakan mana berita yang valid dan mana yang hoaks dengan lebih mudah.
    """) 

    st.markdown("### 📊 Distribusi Label Berita")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        vc = df["label"].value_counts()
        label_map = {0: "Hoax", 1: "Valid", "hoax": "Hoax", "valid": "Valid"}
        labels_text = [label_map[idx] for idx in vc.index]

        fig, ax = plt.subplots(figsize=(4, 3.2))
        bars = ax.bar(labels_text, vc.values,
                    color=[COLOR_HOAX if l == "Hoax" else COLOR_FAKTA for l in labels_text],
                    width=0.5, edgecolor="white")

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("none")

        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown("""
        <div class='info-box'>
        Dataset mencakup total <b>1.043 entri berita</b> setelah melalui tahap pembersihan data duplikat, penyaringan teks bahasa Indonesia, dan pembatasan panjang artikel.<br><br>
        Distribusi target menunjukkan bahwa kelas <b>Hoax (61.7%)</b> lebih dominan dibandingkan dengan kelas <b>Valid (38.3%)</b>. 
        Ketidakseimbangan kelas (class imbalance) ini perlu diwaspadai saat masuk ke tahap pemodelan (modeling). <br><br>
        Model machine learning berisiko mengalami bias yang cenderung menebak kelas mayoritas (Hoax). 
        Oleh karena itu, sangat disarankan untuk berfokus pada metrik evaluasi yang kebal terhadap ketidakseimbangan data seperti <b>F1-Score</b> atau menggunakan teknik penyeimbangan data (seperti pembobotan kelas / class weights).
        </div>
        """, unsafe_allow_html=True)

    # 2. VISUALISASI PANJANG TEKS
    st.markdown("### 📏 Analisis Komparasi Panjang Berita")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(df["text_length"], bins=25, color="#3B82F6", alpha=0.8)
        ax.axvline(df["text_length"].mean(), color="red", linestyle="--")
        ax.set_title("Distribusi Panjang Kata (All News)", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        data_bp = [
            df[df["label_text"]=="Fakta/Asli"]["text_length"],
            df[df["label_text"]=="Hoaks/Fake"]["text_length"]
        ]
        ax.boxplot(data_bp)
        ax.set_xticklabels(["Fakta","Hoaks"])
        ax.set_title("Perbandingan Panjang Berita Per Kelas", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    st.markdown("""
    <div class='info-box'>
        <b>Insight Panjang Teks:</b><br>
        Distribusi panjang teks pada dataset ini bersifat <i>right-skewed</i>, dengan sebagian besar dokumen berada pada rentang 100–500 kata, serta nilai tengah atau rata-rata berada di kisaran 320 kata. 
        Hal ini menunjukkan bahwa mayoritas artikel memiliki panjang moderat, namun terdapat sejumlah dokumen yang sangat panjang hingga mendekati 1.000 kata sebagai <i>outlier</i>. 
        <br><br>
        Analisis per kelas menunjukkan bahwa teks pada kelas <b>Fakta</b> memiliki median yang lebih tinggi dan struktur yang lebih konsisten, 
        mengindikasikan bahwa berita asli umumnya membutuhkan penjelasan lebih panjang untuk menyertakan data pendukung atau konfirmasi sumber. 
        Sebaliknya, kelas <b>Hoaks</b> cenderung memiliki sebaran yang lebih singkat (berada di bawah 250 kata), namun memiliki variasi ekstrim hingga ke batas atas. 
        Perbedaan distribusi yang kontras ini dapat menjadi sinyal tambahan yang kuat bagi model dalam membedakan kredibilitas artikel berdasarkan kompleksitas narasi dan panjang teksnya.
    </div>
    """, unsafe_allow_html=True)

    # 3. DUA GAMBAR TOP KATA BERDAMPINGAN
    st.markdown("### 💬 Perbandingan Kata Dominan di Tiap Label")

    top_words_data = {
        'word': [
            # VALID / Non-Clickbait (Urutan Frekuensi Naik)
            'hoaks', 'mengatakan', 'kata', 'referensi', 'com', 
            'informasi', 'pihak', 'indonesia', 'media', 'klarifikasi',
            
            # HOAX / Clickbait (Urutan Frekuensi Naik)
            'akun', 'penjelasan', 'sosial', 'berita', 'indonesia', 
            'narasi', 'foto', 'media', 'hoaks', 'sumber'
        ],
        'count': [
            # Valid counts (Urutan Naik)
            427, 429, 451, 452, 473, 
            482, 515, 521, 648, 929,
            
            # Hoax counts (Urutan Naik)
            540, 566, 578, 592, 660, 
            667, 726, 791, 933, 981
        ],
        'label': ['Non-Clickbait'] * 10 + ['Clickbait'] * 10
    }
    df_top_words = pd.DataFrame(top_words_data)

    unique_labels = ["Non-Clickbait", "Clickbait"]
    cols = st.columns(2)

    for i, label_name in enumerate(unique_labels):
        with cols[i]:
            df_top = df_top_words[df_top_words["label"] == label_name]
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            bar_color = COLOR_HOAX if label_name == "Non-Clickbait" else COLOR_FAKTA
            
            ax.barh(df_top["word"], df_top["count"], color=bar_color)
            ax.set_title(f"Top Kata - {label_name.upper()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Frekuensi", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)            
        
    st.markdown("""
    <div class='info-box'>
    Analisis frekuensi kata pada data valid menunjukkan bahwa kata <b>"klarifikasi"</b> muncul secara dominan jauh melampaui kata-kata lainnya. Hal ini menandakan bahwa korpus data ini sangat berpusat pada narasi pelurusan informasi atau pernyataan resmi.
    Diikuti oleh kata <b>"media"</b> dan <b>"indonesia"</b> di posisi teratas, grafik ini mencerminkan bahwa penyebaran atau verifikasi informasi yang dianalisis erat kaitannya dengan lanskap media nasional. 
    Munculnya kata spesifik seperti <b>"com"</b> juga mengindikasikan kuatnya pengaruh tautan portal berita digital atau situs web dalam dataset.

    Menariknya, kata <b>"hoaks"</b>, <b>"mengatakan"</b>, dan <b>"referensi"</b> ikut masuk ke dalam jajaran 10 besar dengan frekuensi yang relatif berimbang di kisaran 400 hingga 450. 
    Kehadiran kata-kata tersebut mengindikasikan bahwa dokumen di dalam dataset ini banyak memuat kutipan pernyataan, proses verifikasi, serta ketergantungan yang tinggi pada sumber rujukan dalam menentukan validitas suatu informasi.
    </div>
    """, unsafe_allow_html=True)


# 🛠️ TAB 6 — DATA DICTIONARY
with tab_dict:
    st.markdown("### Data Dictionary")
    st.markdown("""
    Data dictionary adalah dokumentasi yang menjelaskan setiap kolom dalam dataset berita, 
    termasuk tipe data, deskripsi, dan perannya dalam pemodelan. 
    Dokumentasi glosarium kolom ini membantu melacak jenis data numerik maupun objek string 
    yang digunakan dari tahap teks mentah, preprocessing, hingga sistem pemodelan deteksi hoaks, 
    sehingga memudahkan analisis dan pengembangan model selanjutnya.
    """)
    
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(ROOT, "dataset", "data dictionary", "Data_Dictionary_Hoax.csv")

    @st.cache_data
    def load_dict(p):
        return pd.read_csv(p)

    try:
        df_dict = load_dict(path)
        st.dataframe(df_dict, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.error("❌ File Data_Dictionary_Hoax.csv tidak ditemukan di folder dataset.")
        st.write("Path yang dicoba:", path)
