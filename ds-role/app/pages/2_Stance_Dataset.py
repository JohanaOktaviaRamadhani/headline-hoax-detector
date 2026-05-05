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
st.markdown("## 🗣️ Stance Dataset")
st.markdown(
    "<div class='info-box'>"
    "Analisis dataset stance dari Kompasiana — end-to-end pipeline NLP + insight eksploratif."
    "</div>",
    unsafe_allow_html=True,
)

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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Nama File", "dataset_stance_final.csv")
    c2.metric("🌐 Sumber", "Kompasiana")
    c3.metric("📝 Raw Awal", "402 baris")
    c4.metric("✅ Total Final", "300 baris")

    c5, c6, c7 = st.columns(3)
    c5.metric("🗑️ Dibuang (unknown)", "102 baris (25.4%)")
    c6.metric("📊 Jumlah Kolom", "8 kolom")
    c7.metric("🏷️ Kelas", "Mendukung & Membantah")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-title'>Distribusi Stance</div>", unsafe_allow_html=True)
        mendukung = len(df[df["stance"] == "Mendukung"])
        membantah = len(df[df["stance"] == "Membantah"])
        total = len(df)
        fig, ax = plt.subplots(figsize=(4, 4))
        sizes  = [mendukung, membantah]
        colors = [COLOR_MENDUKUNG, COLOR_MEMBANTAH]
        labels = [f"Mendukung\n{mendukung} ({mendukung/total*100:.1f}%)",
                  f"Membantah\n{membantah} ({membantah/total*100:.1f}%)"]
        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
        ax.text(0, 0, f"{total}\ndata", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1F2937")
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False, fontsize=9)
        ax.set_title("Distribusi Stance", fontsize=12, fontweight="bold", pad=10)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("<div class='section-title'>Ringkasan Statistik</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:12px'>
        <div class='dict-row'><span class='dict-col'>Data akhir</span><span class='dict-desc'><b>300 baris</b></span></div>
        <div class='dict-row'><span class='dict-col'>Mendukung</span><span class='dict-desc'><b style='color:{COLOR_MENDUKUNG}'>{mendukung} ({mendukung/total*100:.1f}%)</b></span></div>
        <div class='dict-row'><span class='dict-col'>Membantah</span><span class='dict-desc'><b style='color:{COLOR_MEMBANTAH}'>{membantah} ({membantah/total*100:.1f}%)</b></span></div>
        <div class='dict-row'><span class='dict-col'>Rata-rata kata</span><span class='dict-desc'><b>{df['text_length'].mean():.0f} kata / dokumen</b></span></div>
        <div class='dict-row'><span class='dict-col'>Median kata</span><span class='dict-desc'><b>{df['text_length'].median():.0f} kata</b></span></div>
        <div class='dict-row'><span class='dict-col'>Rata-rata pos_count</span><span class='dict-desc'><b>{df['pos_count'].mean():.1f} kata positif</b></span></div>
        <div class='dict-row'><span class='dict-col'>Rata-rata neg_count</span><span class='dict-desc'><b>{df['neg_count'].mean():.1f} kata negatif</b></span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><div class='section-title'>Preview Data</div>", unsafe_allow_html=True)
        preview = df[["clean_text","stance","text_length","pos_count","neg_count","sentiment_score"]].head(5).copy()
        preview["clean_text"] = preview["clean_text"].apply(lambda x: x[:55]+"..." if len(x)>55 else x)
        st.dataframe(preview, use_container_width=True, hide_index=True)


# TAB 2 — KAMUS KATA
with tab_kamus:
    st.markdown(
        "<div class='info-box'>Tiga kamus kata yang digunakan dalam preprocessing dan feature engineering.</div>",
        unsafe_allow_html=True,
    )
    t1, t2, t3 = st.tabs([
        f"✅ Positif ({len(pos_words)} kata)",
        f"❌ Negatif ({len(neg_words)} kata)",
        f"🚫 Stopword ({len(stop_words)} kata)",
    ])

    def word_tab(words, badge_cls, key):
        q = st.text_input("🔍 Cari kata...", key=key, placeholder="ketik kata...")
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
    st.markdown(
        "<div class='info-box'>Pipeline preprocessing teks secara <b>end-to-end</b> dari data mentah hingga teks bersih siap dimodelkan.</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("Gathering Data",              "Mengumpulkan 402 artikel dari Kompasiana."),
        ("Assessing Data",              "Cek duplikat, missing value, dan distribusi label."),
        ("Drop Kolom Tidak Relevan",    "Hapus kolom <code>no</code>, <code>_id</code>, <code>topic</code>, <code>nnps</code>."),
        ("Gabungkan Title + Content",   "Kolom <code>title + content</code> → kolom <code>text</code>."),
        ("Hapus Label Unknown",         "Hapus 102 baris unknown → tersisa <b>300 baris</b>."),
        ("Mapping Label",               "<code>for</code> → <b>Mendukung</b> · <code>against</code> → <b>Membantah</b>."),
        ("Cleaning Teks",               "Lowercase, hapus URL, mention, hashtag, angka, simbol, normalisasi spasi."),
        ("Formalisasi Slang",           "10 kata slang: <code>yg→yang</code>, <code>tdk→tidak</code>, <code>dgn→dengan</code>, <code>hoax→hoaks</code>, <code>bkn→bukan</code>, <code>tp→tapi</code>, <code>klo→kalau</code>, <code>ga/gak→tidak</code>, <code>udah→sudah</code>."),
        ("Hapus Stopword",              "Sastrawi + custom stopword — dengan proteksi kata negasi & sentimen penting."),
        ("Stemming",                    "Ubah ke bentuk dasar menggunakan <b>Sastrawi Stemmer</b>."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-num'>{i}</div>
            <div>
                <div style='font-weight:600;color:#111827;margin-bottom:3px'>{title}</div>
                <div style='color:#6B7280;font-size:0.88rem'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Slang Dictionary</div>", unsafe_allow_html=True)
    slang = {"yg":"yang","tdk":"tidak","dgn":"dengan","hoax":"hoaks",
             "bkn":"bukan","tp":"tapi","klo":"kalau","ga":"tidak","gak":"tidak","udah":"sudah"}
    st.dataframe(pd.DataFrame(list(slang.items()), columns=["Slang","Formal"]),
                 use_container_width=True, hide_index=True)


# TAB 4 — FEATURE ENGINEERING
with tab_feat:
    st.markdown(
        "<div class='info-box'>Fitur numerik diturunkan dari teks bersih untuk memperkaya representasi data.</div>",
        unsafe_allow_html=True,
    )
    features = [
        ("text_length",     "int", "Jumlah kata dalam <code>clean_text</code>.",                       f"Min {df['text_length'].min()} · Max {df['text_length'].max()} · Mean {df['text_length'].mean():.1f}"),
        ("pos_count",       "int", "Jumlah kata yang cocok dengan kamus kata positif.",                f"Min {df['pos_count'].min()} · Max {df['pos_count'].max()} · Mean {df['pos_count'].mean():.1f}"),
        ("neg_count",       "int", "Jumlah kata yang cocok dengan kamus kata negatif.",                f"Min {df['neg_count'].min()} · Max {df['neg_count'].max()} · Mean {df['neg_count'].mean():.1f}"),
        ("sentiment_score", "int", "Selisih <code>pos_count − neg_count</code>.",                     f"Min {df['sentiment_score'].min()} · Max {df['sentiment_score'].max()} · Mean {df['sentiment_score'].mean():.1f}"),
        ("label",          "int",  "Encoding numerik stance via <b>LabelEncoder</b>.",                "0 = Membantah · 1 = Mendukung"),
    ]
    for fname, ftype, fdesc, fstat in features:
        st.markdown(f"""
        <div class='step-card'>
            <div>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>
                    <span style='font-family:monospace;font-weight:700;color:#4F46E5;font-size:0.95rem'>{fname}</span>
                    <span class='badge badge-blue'>{ftype}</span>
                </div>
                <div style='color:#374151;font-size:0.88rem;margin-bottom:3px'>{fdesc}</div>
                <div style='color:#9CA3AF;font-size:0.82rem'>📊 {fstat}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df[["pos_count","neg_count","sentiment_score","label","text_length"]].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#E5E7EB", ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation Heatmap — Fitur Numerik", fontsize=11, fontweight="bold", pad=10)
    fig.patch.set_facecolor("none")
    st.pyplot(fig, use_container_width=True)
    plt.close()


# TAB 5 — EDA
with tab_eda:
    eda1, eda2, eda3, eda4 = st.tabs([
        "📊 Distribusi", "📏 Panjang Teks", "💬 Top Kata", "😊 Sentimen"
    ])

    # ── EDA 1: Distribusi ──
    with eda1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='section-title'>Bar Chart Distribusi Stance</div>", unsafe_allow_html=True)
            vc = df["stance"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3.5))
            bars = ax.bar(vc.index, vc.values,
                          color=[COLOR_MENDUKUNG, COLOR_MEMBANTAH], width=0.5, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        str(int(bar.get_height())), ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#1F2937")
            ax.set_ylabel("Jumlah Dokumen", fontsize=9)
            ax.set_title("Distribusi Stance", fontsize=11, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_ylim(0, max(vc.values)*1.15)
            fig.patch.set_facecolor("none")
            st.pyplot(fig, use_container_width=True); plt.close()
        with c2:
            st.markdown("<div class='section-title'>Insight</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-box'>
            Kelas <b>Mendukung</b> lebih dominan (<b>177 dokumen, 59%</b>) dibanding
            <b>Membantah</b> (<b>123 dokumen, 41%</b>).<br><br>
            Ketidakseimbangan ini masih dalam batas wajar namun perlu diperhatikan
            agar model tidak bias terhadap kelas mayoritas.
            </div>
            <br>
            """, unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame({"Stance":["Mendukung","Membantah","Total"],
                              "Jumlah":[177,123,300],
                              "Persentase":["59.0%","41.0%","100%"]}),
                use_container_width=True, hide_index=True,
            )

    # ── EDA 2: Panjang Teks ──
    with eda2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='section-title'>Distribusi Panjang Teks</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df["text_length"], bins=30, color=COLOR_MAIN, alpha=0.75, edgecolor="white")
            ax.axvline(df["text_length"].mean(), color="#F43F5E", linestyle="--", lw=1.5,
                       label=f"Mean ({df['text_length'].mean():.0f})")
            ax.axvline(df["text_length"].median(), color="#10B981", linestyle="--", lw=1.5,
                       label=f"Median ({df['text_length'].median():.0f})")
            ax.set_xlabel("Jumlah Kata", fontsize=9); ax.set_ylabel("Frekuensi", fontsize=9)
            ax.set_title("Distribusi Panjang Teks", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
            fig.patch.set_facecolor("none")
            st.pyplot(fig, use_container_width=True); plt.close()

        with c2:
            st.markdown("<div class='section-title'>Boxplot per Stance</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            data_bp = [df[df["stance"]=="Mendukung"]["text_length"],
                       df[df["stance"]=="Membantah"]["text_length"]]
            bp = ax.boxplot(data_bp, patch_artist=True, widths=0.4,
                            medianprops=dict(color="white", linewidth=2))
            for patch, c in zip(bp["boxes"], [COLOR_MENDUKUNG, COLOR_MEMBANTAH]):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_xticklabels(["Mendukung","Membantah"])
            ax.set_ylabel("Jumlah Kata", fontsize=9)
            ax.set_title("Panjang Teks per Stance", fontsize=11, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            fig.patch.set_facecolor("none")
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class='info-box'>
        Distribusi <b>right-skewed</b> — rata-rata <b>433 kata</b>, median <b>352 kata</b>.
        Mayoritas dokumen di rentang <b>200–600 kata</b>.
        Kelas <b>Membantah</b> cenderung sedikit lebih panjang dengan variasi lebih besar.
        </div>""", unsafe_allow_html=True)

    # ── EDA 3: Top Kata ──
    with eda3:
        st.markdown("<div class='section-title'>Top 20 Kata Paling Sering Muncul</div>", unsafe_allow_html=True)
        all_w = " ".join(df["clean_text"]).split()
        top20 = Counter(all_w).most_common(20)
        df_top = pd.DataFrame(top20, columns=["kata","frekuensi"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_bar = [COLOR_MAIN if i < 5 else "#A5B4FC" for i in range(20)]
        ax.barh(df_top["kata"][::-1], df_top["frekuensi"][::-1],
                color=colors_bar[::-1], edgecolor="white")
        ax.set_xlabel("Frekuensi", fontsize=9)
        ax.set_title("Top 20 Kata Paling Sering Muncul", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<br><div class='section-title'>Top 10 Kata per Stance</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        for col, stance, color in [(c1,"Mendukung",COLOR_MENDUKUNG),(c2,"Membantah",COLOR_MEMBANTAH)]:
            with col:
                w = " ".join(df[df["stance"]==stance]["clean_text"]).split()
                df_s = pd.DataFrame(Counter(w).most_common(10), columns=["kata","frekuensi"])
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                ax.barh(df_s["kata"][::-1], df_s["frekuensi"][::-1],
                        color=color, alpha=0.75, edgecolor="white")
                ax.set_title(f"Top 10 — {stance}", fontsize=10, fontweight="bold")
                ax.spines[["top","right"]].set_visible(False)
                fig.patch.set_facecolor("none")
                st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class='info-box'>
        Kata <b>"ahok"</b> dominan di kedua kelas. Pada <b>Membantah</b>, kata <b>"yusril"</b>
        lebih menonjol sebagai perbandingan tokoh. Kata <b>"tidak"</b> & <b>"bukan"</b>
        hadir di keduanya, menandakan banyak pernyataan kontradiktif.
        </div>""", unsafe_allow_html=True)

    # ── EDA 4: Sentimen ──
    with eda4:
        c1, c2, c3 = st.columns(3, gap="large")
        for col, feat, label in [
            (c1, "pos_count",      "Kata Positif"),
            (c2, "neg_count",      "Kata Negatif"),
            (c3, "sentiment_score","Sentiment Score"),
        ]:
            with col:
                st.markdown(f"<div class='section-title'>{label}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.8, 3.2))
                data_bp = [df[df["stance"]=="Mendukung"][feat],
                           df[df["stance"]=="Membantah"][feat]]
                bp = ax.boxplot(data_bp, patch_artist=True, widths=0.4,
                                medianprops=dict(color="white", linewidth=2))
                for patch, c in zip(bp["boxes"],[COLOR_MENDUKUNG,COLOR_MEMBANTAH]):
                    patch.set_facecolor(c); patch.set_alpha(0.7)
                ax.set_xticklabels(["Mendukung","Membantah"], fontsize=8)
                ax.set_title(label, fontsize=10, fontweight="bold")
                ax.spines[["top","right"]].set_visible(False)
                fig.patch.set_facecolor("none")
                st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class='info-box'>
        📌 <b>Kata Negatif</b>: Kelas <b>Membantah</b> konsisten lebih tinggi → indikator pembeda kuat.<br>
        📌 <b>Kata Positif</b>: Distribusi mirip antar kelas → bukan pembeda utama.<br>
        📌 <b>Sentiment Score</b>: <b>Mendukung</b> cenderung lebih positif, namun overlap besar.
        Fitur ini bersifat suplementatif.
        </div>""", unsafe_allow_html=True)


# TAB 6 — DATA DICTIONARY
with tab_dict:
    st.markdown(
        "<div class='info-box'>Penjelasan lengkap setiap kolom pada dataset final yang digunakan untuk pemodelan.</div>",
        unsafe_allow_html=True,
    )
    columns = [
        ("title",          "str", "Judul artikel berita dari Kompasiana.",                    "—"),
        ("content",        "str", "Isi artikel berita (konten utama).",                       "—"),
        ("text",           "str", "Gabungan title + content sebagai input teks awal.",        "—"),
        ("clean_text",     "str", "Teks setelah seluruh tahap preprocessing.",                "—"),
        ("stance",         "str", "Label kategori sikap artikel.",                            "Mendukung / Membantah"),
        ("label",          "int", "Encoding numerik dari kolom stance.",                      "1 = Mendukung · 0 = Membantah"),
        ("text_length",    "int", "Jumlah kata dalam clean_text.",                            f"Range: {df['text_length'].min()}–{df['text_length'].max()}"),
        ("pos_count",      "int", "Jumlah kata positif ditemukan dalam clean_text.",          f"Range: {df['pos_count'].min()}–{df['pos_count'].max()}"),
        ("neg_count",      "int", "Jumlah kata negatif ditemukan dalam clean_text.",          f"Range: {df['neg_count'].min()}–{df['neg_count'].max()}"),
        ("sentiment_score","int", "Selisih pos_count − neg_count.",                           f"Range: {df['sentiment_score'].min()}–{df['sentiment_score'].max()}"),
    ]
    for cname, ctype, cdesc, cextra in columns:
        st.markdown(f"""
        <div class='step-card'>
            <div style='width:100%'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>
                    <span style='font-family:monospace;font-weight:700;color:#4F46E5;font-size:0.95rem'>{cname}</span>
                    <span class='badge badge-blue'>{ctype}</span>
                </div>
                <div style='color:#374151;font-size:0.88rem;margin-bottom:2px'>{cdesc}</div>
                <div style='color:#9CA3AF;font-size:0.80rem'>📌 {cextra}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Statistik Deskriptif</div>", unsafe_allow_html=True)
    st.dataframe(
        df[["text_length","pos_count","neg_count","sentiment_score","label"]].describe().round(2),
        use_container_width=True,
    )
