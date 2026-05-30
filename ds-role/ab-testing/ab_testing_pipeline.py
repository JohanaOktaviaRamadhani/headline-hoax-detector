import os
import re
import time
import string
import numpy as np
import pandas as pd
from scipy import stats

def get_paths():
    """Mendapatkan path absolut yang fleksibel ke dataset dan stopword."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    raw_dir = os.path.join(root_dir, "dataset", "raw")
    final_dir = os.path.join(root_dir, "dataset", "final")
    
    stopwords_path = os.path.join(raw_dir, "stopwords_id.txt")
    dataset_path = os.path.join(final_dir, "final-clickbait.csv")
    
    return root_dir, stopwords_path, dataset_path

# DEFINISI KAMUS SLANG & STOPWORDS
SLANG_DICT = {
    "yg": "yang",
    "tdk": "tidak",
    "dgn": "dengan",
    "hoax": "hoaks",
    "bkn": "bukan",
    "tp": "tapi",
    "klo": "kalau",
    "ga": "tidak",
    "gak": "tidak",
    "udah": "sudah",
    "dpt": "dapat",
    "bgt": "banget",
    "bisaa": "bisa",
    "pake": "pakai",
    "krn": "karena",
    "jd": "jadi",
    "nyari": "mencari"
}

def load_stopwords(stopwords_path):
    """Memuat stopword dari pustaka Sastrawi dan file lokal stopwords_id.txt."""
    stopwords_list = []
    
    # Load Sastrawi Stopwords jika terpasang
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        factory = StopWordRemoverFactory()
        stopwords_list.extend(factory.get_stop_words())
    except ImportError:
        stopwords_list.extend([
            "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
            "ia", "seperti", "jika", "sehingga", "kembali", "dan", "tidak", "ini", "karena",
            "kepada", "oleh", "saat", "harus", "sementara", "setelah", "belum", "kami", "mereka"
        ])
        
    # Load custom stopword lokal
    if os.path.exists(stopwords_path):
        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                lokal = [line.strip() for line in f if line.strip()]
                stopwords_list.extend(lokal)
        except Exception as e:
            print(f"[WARNING] Gagal membaca stopwords_id.txt: {e}")
            
    return list(set(stopwords_list))

# METODE PREPROCESSING A (CONTROL) & B (VARIANT)

def preprocess_control(text):
    """
    Grup A (Control): Preprocessing Dasar
    - Mengubah ke huruf kecil (lowercase)
    - Menghapus tanda baca standar
    - Merapikan spasi ganda
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_variant(text, stopwords_set):
    """
    Grup B (Variant): Preprocessing Lanjutan (Kustom)
    - Mengubah ke huruf kecil (lowercase)
    - Menghapus URL, Mention @, Hashtag #, angka & simbol
    - Formalisasi kata slang lokal
    - Stopword removal kustom
    """
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = text.split()
    formalized_words = [SLANG_DICT.get(w, w) for w in words]
    
    cleaned_words = [w for w in formalized_words if w not in stopwords_set]
    
    return " ".join(cleaned_words)

# EKSEKUSI PIPELINE A/B TESTING
def run_ab_test(sample_size=1000, random_state=42):
    """Menjalankan A/B Testing secara komprehensif."""
    root_dir, stopwords_path, dataset_path = get_paths()
    stopwords_set = set(load_stopwords(stopwords_path))
    
    print("=" * 60)
    print("[RUNNING] PIPELINE A/B TESTING PREPROCESSING")
    print("=" * 60)
    print(f"Path Dataset: {dataset_path}")
    print(f"Jumlah Stopwords: {len(stopwords_set)}")
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"[ERROR] File dataset tidak ditemukan di {dataset_path}!")
        print("Membuat data dummy untuk kebutuhan demonstrasi/testing...")
        df = pd.DataFrame({
            'headline': [
                "Ini adalah contoh berita hoax yg sangat viral di facebook hari ini!",
                "KPK menangkap pejabat yg diduga menerima suap miliaran rupiah.",
                "Info kesehatan terbaru: air rebusan serai bisa menyembuhkan penyakit kronis.",
                "Viral anak ojol di bekasi dapet beasiswa kuliah ke luar negeri bgt keren klo bener.",
                "Pemerintah resmi menaikkan tarif tol jakarta cikampek mulai tdk bersahabat nih."
            ] * 200,
            'article': [
                "Isi artikel berita lengkap mengenai penyebaran hoax yg terus meningkat di media sosial kita."
            ] * 1000
        })
    else:
        df = pd.read_csv(dataset_path)
        
    # Ambil sampel teks acak untuk pengujian
    target_col = 'full_text'
    if target_col not in df.columns:
        if 'headline' in df.columns and 'article' in df.columns:
            df['full_text'] = df['headline'].astype(str) + " " + df['article'].astype(str)
        elif 'headline' in df.columns:
            df['full_text'] = df['headline']
        else:
            df['full_text'] = df[df.columns[0]]
            
    # Hapus missing values dan batasi sampel
    text_samples = df['full_text'].dropna().sample(n=min(sample_size, len(df)), random_state=random_state).tolist()
    print(f"Ukuran Sampel Eksperimen: {len(text_samples)} dokumen berita")
    
    # Penampung hasil benchmarking
    latency_control = []
    latency_variant = []
    
    words_original = []
    words_control = []
    words_variant = []
    
    # 1. Jalankan Benchmark untuk Grup A (Control)
    for text in text_samples:
        orig_len = len(str(text).split())
        words_original.append(orig_len)
        
        start_time = time.perf_counter()
        cleaned = preprocess_control(text)
        duration = (time.perf_counter() - start_time) * 1000  
        
        latency_control.append(duration)
        words_control.append(len(cleaned.split()))
        
    # 2. Jalankan Benchmark untuk Grup B (Variant)
    for text in text_samples:
        start_time = time.perf_counter()
        cleaned = preprocess_variant(text, stopwords_set)
        duration = (time.perf_counter() - start_time) * 1000 
        
        latency_variant.append(duration)
        words_variant.append(len(cleaned.split()))
        
    # KALKULASI METRIK & EVALUASI    
    # Latency Metrics
    avg_lat_c = np.mean(latency_control)
    avg_lat_v = np.mean(latency_variant)
    total_lat_c = np.sum(latency_control)
    total_lat_v = np.sum(latency_variant)
    
    # Persentase kata yang berhasil dikompresi
    total_words_orig = sum(words_original)
    total_words_c = sum(words_control)
    total_words_v = sum(words_variant)
    
    comp_rate_c = ((total_words_orig - total_words_c) / total_words_orig) * 100
    comp_rate_v = ((total_words_orig - total_words_v) / total_words_orig) * 100
    
    # UJI SIGNIFIKANSI STATISTIK (T-TEST)
    t_stat_lat, p_val_lat = stats.ttest_ind(latency_control, latency_variant, equal_var=False)
    
    t_stat_comp, p_val_comp = stats.ttest_rel(words_control, words_variant)
    
    results = {
        'sample_size': len(text_samples),
        'control': {
            'avg_latency_ms': avg_lat_c,
            'total_latency_ms': total_lat_c,
            'total_words': total_words_c,
            'compression_rate_pct': comp_rate_c,
            'latencies': latency_control,
            'word_counts': words_control
        },
        'variant': {
            'avg_latency_ms': avg_lat_v,
            'total_latency_ms': total_lat_v,
            'total_words': total_words_v,
            'compression_rate_pct': comp_rate_v,
            'latencies': latency_variant,
            'word_counts': words_variant
        },
        't_test_latency': {
            't_statistic': t_stat_lat,
            'p_value': p_val_lat,
            'significant': bool(p_val_lat < 0.05)
        },
        't_test_compression': {
            't_statistic': t_stat_comp,
            'p_value': p_val_comp,
            'significant': bool(p_val_comp < 0.05)
        },
        'original_total_words': total_words_orig,
        'original_word_counts': words_original
    }
    
    # DISPLAY RINGKASAN EKSPERIMEN
    print("\n" + "=" * 50)
    print("[SUMMARY] HASIL A/B TESTING PREPROCESSING")
    print("=" * 50)
    print(f"Metrik 1: Kecepatan Proses (Latency - Rata-rata per Teks)")
    print(f"  - Grup A (Control - Dasar): {avg_lat_c:.4f} ms")
    print(f"  - Grup B (Variant - Kustom): {avg_lat_v:.4f} ms")
    diff_lat_pct = ((avg_lat_v - avg_lat_c) / avg_lat_c) * 100
    print(f"  -> Variant B {diff_lat_pct:+.1f}% lebih lambat secara komputasi.")
    print(f"  -> Uji Statistik (T-Test) Latency:")
    print(f"     - T-Statistic = {t_stat_lat:.4f}")
    print(f"     - P-Value     = {p_val_lat:.8f}")
    print(f"     - Signifikan? = {p_val_lat < 0.05} (pada alpha = 0.05)")
    
    print("\nMetrik 2: Pemangkasan Kata (Vocabulary Compression Rate)")
    print(f"  - Total Kata Sebelum Cleaning: {total_words_orig} kata")
    print(f"  - Total Kata Grup A (Control): {total_words_c} kata (Terkompresi {comp_rate_c:.1f}%)")
    print(f"  - Total Kata Grup B (Variant): {total_words_v} kata (Terkompresi {comp_rate_v:.1f}%)")
    print(f"  -> Variant B menghemat {total_words_c - total_words_v} kata ekstra dibanding Control A ({comp_rate_v - comp_rate_c:+.1f}% lebih kompresif).")
    print(f"  -> Uji Statistik (T-Test) Ukuran Kata:")
    print(f"     - T-Statistic = {t_stat_comp:.4f}")
    print(f"     - P-Value     = {p_val_comp:.8f}")
    print(f"     - Signifikan? = {p_val_comp < 0.05} (pada alpha = 0.05)")
    
    print("\n" + "=" * 50)
    print("[DECISION] KESIMPULAN REKOMENDASI UNTUK MODEL MACHINE LEARNING:")
    print("=" * 50)
    
    if p_val_lat < 0.05 and avg_lat_v > avg_lat_c:
        lat_sig_desc = "secara signifikan LEBIH LAMBAT"
    else:
        lat_sig_desc = "tidak berbeda secara signifikan dalam hal kecepatan"
        
    if p_val_comp < 0.05 and comp_rate_v > comp_rate_c:
        comp_sig_desc = "secara signifikan LEBIH EFISIEN memangkas kata tidak penting"
    else:
        comp_sig_desc = "tidak memberikan perbedaan kepadatan informasi yang signifikan"
        
    print(f"1. Grup B (Variant) {lat_sig_desc} dibanding Grup A.")
    print(f"2. Grup B (Variant) {comp_sig_desc} dibanding Grup A.")
    
    if comp_rate_v > comp_rate_c and (avg_lat_v < 1.0 or diff_lat_pct < 200):
        print("\n[RECOMMENDATION] REKOMENDASI FINAL: GUNAKAN GRUP B (VARIANT)!")
        print("Alasan: Meskipun sedikit lebih lambat, perbedaan kecepatannya sangat kecil (di bawah 1 ms per teks) ")
        print("dan performa kompresi kata jauh lebih tinggi. Hal ini akan mengurangi ukuran dimensi TF-IDF secara masif ")
        print("sehingga membuat waktu pelatihan model ML menjadi jauh lebih cepat dan akurat (minim noise).")
    else:
        print("\n[RECOMMENDATION] REKOMENDASI FINAL: GUNAKAN GRUP A (CONTROL)!")
        print("Alasan: Kecepatan komputasi merupakan prioritas utama dan performa kompresi kata kustom ")
        print("tidak memberikan signifikansi yang cukup besar untuk membenarkan penambahan overhead komputasi.")
    print("=" * 60)
    
    # Simpan hasil eksperimen ke CSV
    output_dir = os.path.join(root_dir, "ab-testing", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Summary
    summary_df = pd.DataFrame({
        'Metric': ['Sample Size', 'Control Avg Latency (ms)', 'Variant Avg Latency (ms)', 
                   'Control Compression Pct', 'Variant Compression Pct', 
                   'T-Stat Latency', 'P-Val Latency', 'T-Stat Compression', 'P-Val Compression'],
        'Value': [len(text_samples), avg_lat_c, avg_lat_v, comp_rate_c, comp_rate_v,
                  t_stat_lat, p_val_lat, t_stat_comp, p_val_comp]
    })
    summary_df.to_csv(os.path.join(output_dir, "ab_summary.csv"), index=False)
    
    # Save raw data for plotting
    raw_results_df = pd.DataFrame({
        'Original_Word_Count': words_original,
        'Control_Word_Count': words_control,
        'Variant_Word_Count': words_variant,
        'Control_Latency_ms': latency_control,
        'Variant_Latency_ms': latency_variant
    })
    raw_results_df.to_csv(os.path.join(output_dir, "ab_raw_experiment_data.csv"), index=False)
    print(f"\n[INFO] Hasil eksperimen berhasil disimpan di folder: {output_dir}")
    
    return results

if __name__ == "__main__":
    run_ab_test(sample_size=1000)
