import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys, os

# PATH SETUP (robust)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
APP  = os.path.join(ROOT, "app")
DATA = os.path.join(ROOT, "dataset")

sys.path.append(APP)
from utils import SHARED_CSS, COLOR_MAIN, COLOR_MENDUKUNG, COLOR_MEMBANTAH

st.set_page_config(page_title="Stance Dataset · Hoax Detector", page_icon="🗣️", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# LOADERS
@st.cache_data
def load_data():
    path = os.path.join(DATA, "final", "final_data_stance.csv")
    df = pd.read_csv(path)
    if "stance" not in df.columns:
        df["stance"] = df["label"].map({1: "Mendukung", 0: "Membantah"})
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


# LOAD DATA
df = load_data()
pos_words  = load_words("positif-indonesia.txt")
neg_words  = load_words("negatif-indonesia.txt")
stop_words = load_words("stopwords_id.txt")

# HEADER
st.title("🗣️ Stance Dataset")
st.markdown("""
Dataset stance ini berasal dari platform Kompasiana dan digunakan untuk menganalisis 
posisi opini suatu artikel terhadap suatu topik.
Analisis dilakukan melalui pipeline NLP secara end-to-end, mulai dari data preprocessing, 
feature engineering, hingga eksplorasi pola teks untuk menghasilkan insight yang relevan 
terhadap perbedaan stance (mendukung vs membantah).
""")

# TABS 
tab_info, tab_kamus, tab_prep, tab_feat, tab_eda, tab_dict = st.tabs([
    "📋 Info Dataset",
    "📖 Kamus Kata",
    "⚙️ Preprocessing",
    "🔧 Feature Engineering",
    "📈 EDA",
    "📚 Data Dictionary",
])

# TAB 1 — INFO DATASET
with tab_info:
    st.markdown("### Informasi Dataset")
    st.markdown("""
    Dataset ini merupakan hasil akhir dari pipeline preprocessing yang telah melalui proses 
    filtering dan pembersihan data untuk memastikan kualitas dan konsistensi. 
    Ringkasan berikut menampilkan ukuran dataset, perubahan jumlah data setelah cleaning, 
    serta komposisi label yang menjadi dasar dalam analisis stance.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📁 Nama File**  
        dataset_stance_final.csv  

        **📊 Jumlah Kolom**  
        8 kolom  
        """)

    with col2:
        st.markdown("""
        **📝 Raw Awal**  
        402 baris  

        **✅ Total Final**  
        300 baris  
        """)

    with col3:
        st.markdown("""
        **🗑️ Dibuang (unknown)**  
        102 baris (25.4%)   

        **🏷️ Kelas**  
        Mendukung & Membantah  
        """)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TOP: CHART + SUMMARY ──
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("**Distribusi Stance**")

        mendukung = len(df[df["stance"] == "Mendukung"])
        membantah = len(df[df["stance"] == "Membantah"])
        total = len(df)

        sizes = [mendukung, membantah]
        colors = [COLOR_MENDUKUNG, COLOR_MEMBANTAH]

        fig, ax = plt.subplots(figsize=(3, 3))  
        wedges, _ = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            radius=0.85,
            wedgeprops=dict(width=0.35, edgecolor="white")
        )
        ax.text(
            0, 0,
            f"{total}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )
        ax.legend(
            wedges,
            [
                f"Mendukung ({mendukung})",
                f"Membantah ({membantah})"
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            fontsize=9
        )
        ax.set(aspect="equal")
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=False) 
        plt.close()

    with col_right:
        st.markdown("**Ringkasan Statistik**")
        st.markdown(f"""
        - **Total data**: {total} baris  
        - **Mendukung**: {mendukung} ({mendukung/total*100:.1f}%)  
        - **Membantah**: {membantah} ({membantah/total*100:.1f}%)  
        - **Rata-rata kata**: {df['text_length'].mean():.0f} kata  
        - **Median kata**: {df['text_length'].median():.0f} kata  
        - **Rata-rata kata positif**: {df['pos_count'].mean():.1f}  
        - **Rata-rata kata negatif**: {df['neg_count'].mean():.1f}  
        """)

    # ── PREVIEW DATA FINAL ──
    st.markdown("**Preview Data Final**")

    preview = df[[
        "clean_text", "stance",
        "text_length", "pos_count",
        "neg_count", "sentiment_score"
    ]].head(5).copy()

    preview["clean_text"] = preview["clean_text"].apply(
        lambda x: x[:55] + "..." if len(x) > 55 else x
    )

    st.dataframe(preview, use_container_width=True, hide_index=True)

# TAB 2 — KAMUS KATA
with tab_kamus:
    st.markdown("### Kamus Kata")
    st.markdown("""
    Tiga kamus kata digunakan dalam proses preprocessing dan feature engineering
    untuk menganalisis karakteristik teks secara lebih terstruktur.
    Kamus positif dan negatif digunakan untuk mengidentifikasi kecenderungan sentimen 
    dalam teks, sedangkan stopword digunakan untuk menghapus kata-kata umum yang tidak 
    memiliki makna signifikan dalam analisis.
    Penggunaan kamus ini membantu meningkatkan kualitas representasi teks serta 
    mendukung pembentukan fitur seperti pos_count, neg_count, dan sentiment_score.
    """)
    
    t1, t2, t3 = st.tabs([
        f"Positif ({len(pos_words)} kata)",
        f"Negatif ({len(neg_words)} kata)",
        f"Stopword ({len(stop_words)} kata)",
    ])

    def word_tab(words, badge_cls, key):
        q = st.text_input("🔍 Cari kata...", key=key, placeholder="Masukkan kata yang ingin kamu cari...")
        filtered = [w for w in words if q.lower() in w.lower()] if q else words
        st.markdown(f"<small style='color:#6B7280'>Menampilkan {min(len(filtered),200)} dari {len(words)} kata</small>",
                    unsafe_allow_html=True)
        st.markdown(" ".join([f"<span class='badge {badge_cls}'>{w}</span>" for w in filtered[:200]]),
                    unsafe_allow_html=True)

    with t1: word_tab(pos_words,  "badge-green", "kp")
    with t2: word_tab(neg_words,  "badge-red",   "kn")
    with t3: word_tab(stop_words, "badge-gray",  "ks")

# TAB 3 — PREPROCESSING
with tab_prep:
    st.markdown("### Pipeline Preprocessing Teks")
    st.markdown("""
    Pipeline preprocessing dilakukan secara end-to-end untuk mentransformasi data mentah 
    menjadi teks bersih yang siap digunakan dalam proses analisis dan pemodelan. 
    Setiap tahapan dirancang untuk memastikan kualitas data tetap terjaga serta 
    memaksimalkan informasi yang dapat diekstraksi dari teks.
    """)

    def render_chips(items, color="#F3F4F6", text_color="#111827"):
        return " ".join([
            f"<span style='background:{color};color:{text_color};padding:4px 8px;border-radius:6px;margin:2px;display:inline-block;font-size:0.8rem'>{item}</span>"
            for item in items
        ])

    # pipeline steps
    steps = [
        ("Gathering Data",
         "Mengumpulkan 402 artikel dari Kompasiana."),
        ("Assessing Data",
         "Cek duplikat, missing value, dan distribusi label."),
        ("Drop Kolom Tidak Relevan",
         render_chips(["no", "_id", "topic", "nnps"])),
        ("Gabungkan Title + Content",
         render_chips(["title", "+", "content", "→", "text"], "#E0F2FE", "#0369A1")),
        ("Hapus Label Unknown",
         "Hapus 102 baris unknown → tersisa <b>300 baris</b>."),
        ("Mapping Label",
         render_chips(["for → Mendukung", "against → Membantah"], "#FEF3C7", "#92400E")),
        ("Cleaning Teks",
         "Lowercase, hapus URL, mention, hashtag, angka, simbol, dan normalisasi spasi."),
        ("Formalisasi Slang",
         render_chips([
             "yg→yang","tdk→tidak","dgn→dengan","hoax→hoaks",
             "bkn→bukan","tp→tapi","klo→kalau","ga/gak→tidak","udah→sudah"
         ], "#ECFDF5", "#065F46")),
        ("Hapus Stopword",
         "Sastrawi + custom stopword dengan proteksi kata negasi & sentimen penting."),
        ("Stemming",
         "Mengubah kata ke bentuk dasar menggunakan <b>Sastrawi Stemmer</b>."),
    ]

    #  render step cards
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-num'>{i}</div>
            <div>
                <div style='font-weight:600;color:#111827;margin-bottom:4px'>{title}</div>
                <div style='color:#6B7280;font-size:0.9rem'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# TAB 4 — FEATURE ENGINEERING
with tab_feat:
    st.markdown("### Feature Engineering")
    st.markdown("""
    Feature engineering dilakukan untuk mentransformasikan teks bersih menjadi 
    fitur numerik yang dapat diproses oleh model machine learning. 
    Fitur yang dihasilkan mencerminkan karakteristik struktural dan sentimen teks, 
    seperti panjang dokumen, intensitas kata positif dan negatif, serta skor sentimen. 
    Representasi ini membantu model dalam menangkap pola yang relevan untuk 
    membedakan stance secara lebih akurat.
    """)

    features = [
        ("text_length",
        "Jumlah kata dalam teks",
        f"Teks berkisar dari {df['text_length'].min()} hingga {df['text_length'].max()} kata, dengan rata-rata sekitar {df['text_length'].mean():.0f} kata per dokumen."),

        ("pos_count",
        "Jumlah kata positif",
        f"Rata-rata terdapat {df['pos_count'].mean():.0f} kata positif per teks, dengan variasi dari {df['pos_count'].min()} hingga {df['pos_count'].max()} kata."),

        ("neg_count",
        "Jumlah kata negatif",
        f"Rata-rata terdapat {df['neg_count'].mean():.0f} kata negatif, dengan rentang {df['neg_count'].min()} sampai {df['neg_count'].max()} kata."),

        ("sentiment_score",
        "Selisih sentimen (positif − negatif)",
        f"Skor sentimen berkisar dari {df['sentiment_score'].min()} hingga {df['sentiment_score'].max()}, dengan kecenderungan rata-rata {df['sentiment_score'].mean():.1f}."),

        ("label",
        "Representasi numerik stance",
        "0 menunjukkan Membantah, sedangkan 1 menunjukkan Mendukung."),
    ]

    for name, desc, explain in features:
        st.markdown(f"""
        <div style='padding:10px 12px;border:1px solid #E5E7EB;border-radius:10px;margin-bottom:8px'>
            <div style='font-family:monospace;font-weight:600;color:#4F46E5;margin-bottom:3px'>{name}</div>
            <div style='font-size:0.85rem;color:#6B7280;margin-bottom:2px'>{desc}</div>
            <div style='font-size:0.82rem;color:#9CA3AF'>{explain}</div>
        </div>
        """, unsafe_allow_html=True)

    # st.markdown("#### Correlation Matrix")
    # st.markdown("""
    # Matriks korelasi menunjukkan hubungan antar fitur numerik. 
    # Nilai mendekati 1 berarti hubungan positif kuat, mendekati -1 berarti negatif kuat, 
    # dan mendekati 0 berarti tidak ada hubungan linear yang signifikan.
    # """)

    # cols = ["text_length", "pos_count", "neg_count", "sentiment_score", "label"]
    # corr = df[cols].corr()
    # fig, ax = plt.subplots(figsize=(4, 3), dpi=120)  
    # sns.heatmap(
    #     corr,
    #     annot=True,
    #     fmt=".1f",  
    #     cmap="coolwarm",
    #     annot_kws={"size":7},
    #     cbar=False,
    #     square=True, 
    #     ax=ax
    # )
    # ax.tick_params(labelsize=7)
    # plt.tight_layout()  
    # st.pyplot(fig, use_container_width=False)  

# TAB 5 — EDA
with tab_eda:
    st.markdown("### Exploratory Data Analysis (EDA)")
    st.markdown("""
    Secara keseluruhan, dataset menunjukkan bahwa perbedaan stance tidak hanya dipengaruhi 
    oleh sentimen, tetapi juga oleh struktur teks dan konteks kata. 
    Hal ini mengindikasikan bahwa pendekatan berbasis fitur sederhana perlu dikombinasikan 
    dengan representasi teks yang lebih kaya untuk menghasilkan model yang optimal.
    """)

    # DISTRIBUSI STANCE
    st.markdown("### 📊 Distribusi Stance")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        vc = df["stance"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3.2))
        bars = ax.bar(vc.index, vc.values,
                      color=[COLOR_MENDUKUNG, COLOR_MEMBANTAH],
                      width=0.5, edgecolor="white")

        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title("Distribusi Stance", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_facecolor("none")

        st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
       st.markdown("""
        <div class='info-box'>
        Distribusi data menunjukkan bahwa kelas <b>Mendukung</b> (59%) lebih dominan dibanding 
        <b>Membantah</b> (41%). Meskipun terdapat ketidakseimbangan, selisih ini masih dalam batas 
        yang relatif moderat sehingga model masih dapat belajar pola dari kedua kelas.

        Namun demikian, kondisi ini tetap berpotensi menyebabkan bias ke kelas mayoritas, 
        terutama pada model yang sensitif terhadap distribusi data. Oleh karena itu, pada tahap 
        modeling perlu dipertimbangkan strategi seperti penggunaan metrik evaluasi yang tepat 
        (AUC/F1-score) atau teknik penyeimbangan data.
        </div>
        """, unsafe_allow_html=True)

    # PANJANG TEKS
    st.markdown("### 📏 Panjang Teks")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(df["text_length"], bins=30, color=COLOR_MAIN, alpha=0.75)
        ax.axvline(df["text_length"].mean(), linestyle="--")
        ax.set_title("Distribusi Panjang Teks", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        data_bp = [
            df[df["stance"]=="Mendukung"]["text_length"],
            df[df["stance"]=="Membantah"]["text_length"]
        ]
        ax.boxplot(data_bp)
        ax.set_xticklabels(["Mendukung","Membantah"])
        ax.set_title("Per Stance", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown("""
    <div class='info-box'>
    Distribusi panjang teks bersifat <b>right-skewed</b>, dengan sebagian besar dokumen 
    berada pada rentang <b>200–600 kata</b>, serta rata-rata sekitar <b>433 kata</b>. 
    Hal ini menunjukkan bahwa mayoritas artikel memiliki panjang moderat, namun terdapat 
    sejumlah dokumen yang sangat panjang sebagai outlier.
    Analisis per kelas menunjukkan bahwa teks pada kelas <b>Membantah</b> cenderung memiliki 
    variasi panjang yang lebih besar, yang mengindikasikan bahwa argumen bantahan seringkali 
    lebih eksploratif atau membutuhkan penjelasan lebih panjang.
    Perbedaan distribusi ini dapat menjadi sinyal tambahan bagi model, terutama jika panjang 
    teks berasosiasi dengan kompleksitas argumen.
    </div>
    """, unsafe_allow_html=True)

    # TOP KATA
    st.markdown("### 💬 Top Kata")
    all_w = " ".join(df["clean_text"]).split()
    top20 = Counter(all_w).most_common(20)
    df_top = pd.DataFrame(top20, columns=["kata","frekuensi"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(df_top["kata"][::-1], df_top["frekuensi"][::-1])
    ax.set_title("Top 20 Kata", fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown("""
    <div class='info-box'>
    Analisis frekuensi kata menunjukkan bahwa kata <b>"ahok"</b> muncul dominan pada kedua kelas, 
    menandakan bahwa topik utama dalam dataset berpusat pada entitas tersebut.

    Pada kelas <b>Membantah</b>, kata seperti <b>"yusril"</b> muncul lebih sering, yang menunjukkan 
    adanya konteks perbandingan atau narasi kontra terhadap tokoh tertentu. Selain itu, 
    kata-kata seperti <b>"tidak"</b> dan <b>"bukan"</b> muncul di kedua kelas, mencerminkan adanya 
    banyak pernyataan negasi dalam teks.

    Temuan ini mengindikasikan bahwa perbedaan stance tidak hanya bergantung pada kata kunci utama, 
    tetapi juga pada konteks kalimat dan struktur argumen, sehingga pendekatan berbasis 
    bag-of-words saja mungkin belum cukup untuk menangkap keseluruhan pola.
    </div>
    """, unsafe_allow_html=True)

    # SENTIMEN
    st.markdown("### 😊 Analisis Sentimen")
    c1, c2, c3 = st.columns(3, gap="large")
    for col, feat, title in [
        (c1, "pos_count", "Positif"),
        (c2, "neg_count", "Negatif"),
        (c3, "sentiment_score", "Score"),
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(3.2, 3))
            data_bp = [
                df[df["stance"]=="Mendukung"][feat],
                df[df["stance"]=="Membantah"][feat]
            ]
            ax.boxplot(data_bp)
            ax.set_xticklabels(["M","B"], fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("""
    <div class='info-box'>
    Analisis sentimen menunjukkan bahwa <b>jumlah kata negatif</b> secara konsisten lebih tinggi 
    pada kelas <b>Membantah</b>, menjadikannya sebagai indikator yang cukup kuat dalam membedakan 
    kedua kelas.
    Sebaliknya, distribusi <b>kata positif</b> relatif mirip antara kedua kelas, sehingga fitur ini 
    tidak memberikan kontribusi signifikan sebagai pembeda utama.

    Sementara itu, <b>sentiment_score</b> menunjukkan bahwa kelas <b>Mendukung</b> cenderung memiliki 
    nilai yang lebih positif, namun terdapat overlap yang cukup besar antar kelas. Hal ini 
    menunjukkan bahwa meskipun sentimen berperan, stance tidak sepenuhnya ditentukan oleh 
    polaritas sentimen.

    Dengan demikian, fitur berbasis sentimen lebih tepat digunakan sebagai <i>supporting feature</i>, 
    bukan sebagai penentu utama dalam klasifikasi stance.
    </div>
    """, unsafe_allow_html=True)

# TAB 6 — DATA DICTIONARY
with tab_dict:
    st.markdown("### Data Dictionary")
    st.markdown("""
    Data dictionary adalah dokumentasi yang menjelaskan setiap kolom dalam dataset, 
    termasuk tipe data, deskripsi, dan perannya dalam pipeline data.
    Dokumentasi ini membantu memahami struktur data dari tahap raw data, 
    preprocessing, hingga feature engineering, sehingga memudahkan analisis 
    dan pengembangan model selanjutnya.
    """)

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(ROOT, "dataset", "data dictionary", "Data_Dictionary_Stance.csv")

    @st.cache_data
    def load_dict(p):
        return pd.read_csv(p)

    try:
        df_dict = load_dict(path)
        st.dataframe(df_dict, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.error("❌ File Data_Dictionary_Stance.csv tidak ditemukan di folder dataset.")
        st.write("Path yang dicoba:", path)