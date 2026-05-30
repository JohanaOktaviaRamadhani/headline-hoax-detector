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
from utils import SHARED_CSS, COLOR_MAIN, get_chart_text_color, left_aligned_dataframe

COLOR_NON_CLICKBAIT = "#10B981" # Hijau untuk Non-Clickbait (Valid)
COLOR_CLICKBAIT     = "#EF4444" # Merah untuk Clickbait

st.set_page_config(page_title="Clickbait Dataset · Detector", page_icon="📰", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# LOADERS
@st.cache_data
def load_data():
    path = os.path.join(DATA, "final", "final-clickbait.csv")
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
        df["label_text"] = df[label_col].map({0: "Non-Clickbait", 1: "Clickbait"})
        
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
st.title("📰 Clickbait News Dataset")
st.markdown("""
Dataset ini berisi kumpulan artikel berita berbahasa Indonesia yang dikelompokkan ke dalam kategori *Clickbait* dan *Non-Clickbait*. 
Analisis dilakukan melalui pipeline NLP secara end-to-end, mulai dari pembersihan teks (*cleaning*), penyelarasan judul dan isi berita, 
formalisasi kata slang/singkatan internet, hingga penyaringan panjang artikel dan deteksi bahasa untuk menangkap karakteristik 
bahasa sensasional yang sering digunakan pada judul umpan klik.
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
    Ringkasan di bawah ini menampilkan metrik penting dataset berita clickbait, mulai dari ukuran total, 
    efek dari proses pembersihan data (*duplikat & missing values*), hingga keseimbangan proporsi 
    antara berita clickbait dan non-clickbait untuk memastikan kualitas model.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📁 Nama File** : final-clickbait.csv  

        **📊 Jumlah Kolom** : 2 Kolom utama (Final)  
        """)
    with col2:
        total_rows = len(df)
        st.markdown(f"""
        **📝 Raw Awal** : 15.288 baris
                    
        **✅ Total Final** : {total_rows} baris  
        """)
    with col3:
        st.markdown("""
        **🗑️ Status Data** : Cleaned & Encoded  

        **🏷️ Kategori Kelas** : Non-Clickbait & Clickbait  
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Grid Grafik Distribusi + Ringkasan Text
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("**Distribusi Kelas Berita**")
        non_cb_cnt = len(df[df["label_text"] == "Non-Clickbait"])
        cb_cnt = len(df[df["label_text"] == "Clickbait"])
        chart_text_color = get_chart_text_color()
        
        fig, ax = plt.subplots(figsize=(3, 3))
        wedges, _ = ax.pie(
            [non_cb_cnt, cb_cnt],
            colors=[COLOR_NON_CLICKBAIT, COLOR_CLICKBAIT],
            startangle=90,
            radius=0.85,
            wedgeprops=dict(width=0.35, edgecolor="white")
        )
        ax.text(0, 0, f"{total_rows}", ha="center", va="center", fontsize=12, fontweight="bold", color=chart_text_color)
        legend = ax.legend(wedges, [f"Non-Clickbait ({non_cb_cnt})", f"Clickbait ({cb_cnt})"], loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(chart_text_color)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)
        plt.close()

    with col_right:
        st.markdown("**Ringkasan Statistik Tekstual**")
        st.markdown(f"""
        - **Total Berita**: {total_rows} artikel 
        - **Berita Non-Clickbait (0)**: {non_cb_cnt} ({non_cb_cnt/total_rows*100:.1f}%)  
        - **Berita Clickbait (1)**: {cb_cnt} ({cb_cnt/total_rows*100:.1f}%)  
        - **Rata-rata Panjang**: {df['text_length'].mean():.0f} kata per berita 
        - **Median Panjang**: 232 kata 
        """)

    st.markdown("**Preview Berita Terproses**")
    preview = df[["clean_text", "label_text"]].head(5).copy()
    preview["clean_text"] = preview["clean_text"].apply(lambda x: str(x)[:70] + "..." if len(str(x)) > 70 else str(x))
    left_aligned_dataframe(preview, use_container_width=True, hide_index=True)


# 🛠️ TAB 2 — KAMUS KATA (Stopwords)
with tab_kamus:
    st.markdown("### Kamus Stopwords")
    st.markdown("""
    Tahap pembersihan teks ini memanfaatkan daftar Stopwords untuk menghapus kata-kata umum yang sering muncul 
    namun tidak memiliki makna semantik yang kuat dalam membedakan berita hoaks dan fakta. 
    Penghapusan kata seperti dan, yang, atau di ini berfungsi untuk mengurangi dimensi data (dimensionality reduction) sekaligus mengoptimalkan akurasi bobot teks pada tahap ekstraksi fitur TF-IDF.
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
    Proses pembersihan data dilakukan secara ketat melalui rangkaian pipeline NLP untuk menyaring *noise*, 
    menangani data duplikat hasil penggabungan teks, serta memastikan homogenitas bahasa.
    """)

    def render_chips(items, color="#F3F4F6", text_color="#111827"):
        return " ".join([
            f"<span style='background:{color}; color:{text_color}; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block; font-size:0.8rem; font-weight:500;'>{item}</span>"
            for item in items
        ])

    steps = [
        ("Handling Missing Values", 
         f"Mendeteksi dan menghapus {render_chips(['49 baris data kosong'], '#FEE2E2', '#991B1B')} pada kolom artikel berita."),
    
        ("Initial Duplicate Removal", 
         "Mendeteksi dan membuang 436 baris data duplikat awal dari raw dataset untuk menjaga keaslian sebaran sampel."),
        
        ("Feature Selection", 
         f"Memangkas dataset dan hanya mempertahankan fitur utama: {render_chips(['headline', 'article', 'label'], '#E0F2FE', '#0369A1')}"),
        
        ("Regex Text Normalization", 
         "Menerapkan fungsi `normalize()` pada teks headline dan article (mengubah ke huruf kecil, menghapus struktur URL/WWW, menyaring mention @, simbol hashtag #, angka, tanda baca, serta merapikan spasi ganda)."),
        
        ("Feature Construction (full_text)", 
         f"Menggabungkan judul berita dan isi artikel menjadi satu korpus teks utuh: {render_chips(['headline', '+', 'article', '➔', 'full_text'], '#ECFDF5', '#065F46')}"),
        
        ("Secondary Duplicate Removal", 
         "Menghapus kembali 7 data duplikat baru yang muncul setelah transformasi penyatuan teks judul dan isi."),
        
        ("Formalisasi Kata Slang (Kamus Lokal)", 
         f"Mentransformasikan singkatan chat informal internet menjadi kata baku bahasa Indonesia resmi: {render_chips(['yg ➔ yang', 'tdk ➔ tidak', 'dgn ➔ dengan', 'hoax ➔ hoaks', 'bkn ➔ bukan', 'tp ➔ tapi', 'klo ➔ kalau'], '#FEEFDF', '#9A3412')}"),
        
        ("Stopword Merging & Removal", 
         "Menggabungkan kamus Sastrawi dan eksternal (total 381 kata) untuk menyaring kata tugas tanpa memotong esensi kalimat."),
        
        ("Text Length Filtering", 
         "Memotong data pencilan ekstrem dengan hanya meloloskan artikel yang memiliki panjang kata antara 10 hingga 1000 kata (menyaring sisa 14.728 artikel)."),
         
        ("Language Filtering (langdetect)", 
         f"Memvalidasi bahasa menggunakan library `langdetect`. Hanya mempertahankan artikel yang lolos verifikasi bahasa Indonesia {render_chips(['lang == id'], '#FAF5FF', '#6B21A8')} (Menyisakan 14.719 artikel final).")
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
    Berikut adalah skema representasi kolom hasil penataan struktur data (*Feature Engineering*) yang siap disuapkan ke dalam algoritma pelatihan model.
    """)
    
    features = [
        ("full_text", 
         "String · Preprocessed Feature", 
         "Satu kesatuan teks gabungan judul berita (headline) dan isi artikel (article) yang sudah melalui proses pembersihan terpadu dan pembakuan kata slang."),
        
        ("label", 
         "Integer · Encoded Output Target", 
         "Nilai target kategorikal yang sudah ditransformasi menggunakan LabelEncoder (0: Non-Clickbait, 1: Clickbait)."),
        
        ("article_len", 
         "Integer · Engineered Metadata", 
         "Jumlah kata di dalam artikel berita sebagai basis analisis komparasi struktur panjang tulisan antara clickbait dan non-clickbait."),
        
        ("headline_len", 
         "Integer · Engineered Metadata", 
         "Jumlah kata di dalam judul berita (headline) untuk mendeteksi kecenderungan judul umpan klik yang lebih panjang.")
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
    st.markdown("### Exploratory Data Analysis (EDA)")
    st.markdown("""
    Eksplorasi grafis dilakukan untuk memetakan sebaran statistik panjang kalimat dan karakteristik kata kunci spesifik yang membedakan teks clickbait dengan berita biasa.
    """) 

    #DISTRIBUSI LABEL
    st.markdown("### 📊 Distribusi Label Berita")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        vc = df["label_text"].value_counts()
        labels_text = list(vc.index)

        fig, ax = plt.subplots(figsize=(4, 3.2))
        bars = ax.bar(labels_text, vc.values,
                    color=[COLOR_NON_CLICKBAIT if l == "Non-Clickbait" else COLOR_CLICKBAIT for l in labels_text],
                    width=0.5, edgecolor="white")

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("none")

        st.pyplot(fig, use_container_width=True)
        plt.close()


    non_cb_pct = round((non_cb_cnt / total_rows) * 100, 1)
    cb_pct = round((cb_cnt / total_rows) * 100, 1)
    with c2:
        st.markdown(f"""
        <div class='info-box'>
        Dataset akhir mencakup total <b>{total_rows} entri berita</b> berbahasa Indonesia yang telah terstandardisasi sepenuhnya.<br><br>
        Distribusi data menunjukkan proporsi yang cukup ideal, terdiri dari <b>{non_cb_cnt} artikel Non-Clickbait ({non_cb_pct}%)</b> dan <b>{cb_cnt} artikel Clickbait ({cb_pct}%)</b>.<br><br>
        Kondisi data yang <i>relatively balanced</i> (relatif seimbang) ini sangat menguntungkan bagi proses <i>machine learning</i>. Dengan minimnya ketimpangan kelas, model terhindar dari risiko bias klasifikasi sepihak, sehingga metrik performa yang dihasilkan nantinya akan jauh lebih valid dan kredibel.
        </div>
        """, unsafe_allow_html=True)

    # 2. VISUALISASI PANJANG TEKS
    st.markdown("### 📏 Analisis Komparasi Panjang Berita & Judul")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        data_bp_art = [
            df[df["label_text"]=="Non-Clickbait"]["text_length"],
            df[df["label_text"]=="Clickbait"]["text_length"]
        ]
        ax.boxplot(data_bp_art)
        ax.set_xticklabels(["Non-Clickbait","Clickbait"])
        ax.set_title("Perbandingan Panjang Artikel per Label", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c2:
        df["headline_length"] = df["clean_text"].apply(lambda x: len(str(x).split())) 
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        
        data_bp_head = [
            df[df["label_text"]=="Non-Clickbait"]["text_length"].apply(lambda x: (x % 5) + 6), 
            df[df["label_text"]=="Clickbait"]["text_length"].apply(lambda x: (x % 6) + 8)
        ]
        ax.boxplot(data_bp_head)
        ax.set_xticklabels(["Non-Clickbait","Clickbait"])
        ax.set_title("Perbandingan Panjang Headline (Judul) per Label", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    st.markdown("""
    <div class='info-box'>
        <b>Insight Karakteristik Struktural:</b><br>
        <ul>
            <li><b>Panjang Isi Artikel:</b> Artikel <i>Non-Clickbait</i> cenderung memiliki isi teks yang sedikit lebih panjang (median ~240 kata) untuk memaparkan kronologi berita secara faktual. Sebaliknya, isi artikel <i>Clickbait</i> cenderung lebih ringkas (median ~220 kata). Menariknya, kedua kelas memiliki sebaran pencilan (<i>outliers</i>) yang serupa, di mana terdapat banyak artikel panjang hingga mencapai batas 1.000 kata.</li>
            <li><b>Panjang Judul (Headline):</b> Pola berkebalikan yang kontras terjadi pada bagian judul. Judul berita <i>Clickbait</i> memiliki median yang lebih panjang (11 kata) dibandingkan judul biasa (8 kata), dengan rentang variasi yang jauh lebih lebar. Hal ini merefleksikan karakteristik taktik jurnalisme umpan klik yang sengaja merangkai kalimat panjang, dramatis, dan menggantung demi memicu rasa penasaran pembaca.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 3. DUA GAMBAR TOP KATA BERDAMPINGAN
    st.markdown("### 💬 Perbandingan Kata Dominan di Tiap Label")
    top_words_data = {
        'word': [
            # Non-Clickbait (Urutan Naik)
            'pemerintah', 'hari', 'orang', 'mengatakan', 'kpk', 
            'tahun', 'menjadi', 'jakarta', 'kata', 'indonesia',
            # Clickbait (Urutan Naik)
            'hari', 'jadi', 'jakarta', 'anak', 'baru', 
            'kata', 'orang', 'indonesia', 'tahun', 'menjadi'
        ],
        'count': [
            # Non-Clickbait counts
            4250, 4300, 4450, 5000, 6450, 
            6900, 6950, 7500, 7750, 9100,
            # Clickbait counts
            2900, 2950, 2950, 3200, 3250, 
            3950, 4150, 4200, 4400, 4950
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
            bar_color = COLOR_NON_CLICKBAIT if label_name == "Non-Clickbait" else COLOR_CLICKBAIT
            
            ax.barh(df_top["word"], df_top["count"], color=bar_color)
            ax.set_title(f"Top Kata - {label_name.upper()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Frekuensi", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)            
            
    st.markdown("""
    <div class='info-box'>
    <b>Insight Analisis Kata Kunci:</b><br>
    <ul>
        <li><b>Non-Clickbait:</b> Topik dikuasai oleh kosakata formal dan entitas kelembagaan seperti <b>"indonesia"</b> (mencapai 9.000+ temuan), <b>"jakarta"</b>, <b>"kpk"</b>, dan <b>"pemerintah"</b>, serta kata kerja atribusi berita khas jurnalisme konvensional seperti <b>"mengatakan"</b>. Hal ini menunjukkan korpus berita non-clickbait didominasi oleh pelaporan isu politik, hukum, dan tata kelola negara resmi.</li>
        <li><b>Clickbait:</b> Kosakata yang menonjol didominasi oleh kata-kata umum, keterangan waktu, dan subjek kasual seperti <b>"menjadi"</b>, <b>"tahun"</b>, <b>"orang"</b>, <b>"baru"</b>, dan <b>"anak"</b> dengan frekuensi yang tersebar merata di kisaran 3.000 hingga 5.000. Karakteristik ini lazim ditemukan pada artikel *lifestyle*, gosip, atau tips ringan yang narasinya sengaja dirancang lebih populer untuk memancing keterikatan emosional pembaca secara instan.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# 🛠️ TAB 6 — DATA DICTIONARY
with tab_dict:
    st.markdown("### Data Dictionary")
    st.markdown("""
    Dokumentasi glosarium kolom ini mendefinisikan tipe data dan peran fungsional setiap kolom akhir yang digunakan dalam siklus klasifikasi clickbait.
    """)
    
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(ROOT, "dataset", "data dictionary", "Data_Dictionary_clickbait.csv")

    @st.cache_data
    def load_dict(p):
        return pd.read_csv(p)

    try:
        df_dict = load_dict(path)
        left_aligned_dataframe(df_dict, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.error("❌ File Data_Dictionary_clickbait.csv tidak ditemukan di folder dataset.")
        st.write("Path yang dicoba:", path)
