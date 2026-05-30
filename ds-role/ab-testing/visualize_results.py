import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    results_dir = os.path.join(current_dir, "results")
    plots_dir = os.path.join(current_dir, "plots")
    return results_dir, plots_dir

def generate_plots():
    results_dir, plots_dir = get_paths()
    raw_data_path = os.path.join(results_dir, "ab_raw_experiment_data.csv")
    
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Data eksperimen mentah tidak ditemukan di {raw_data_path}!")
        print("Harap jalankan 'python ab_testing_pipeline.py' terlebih dahulu.")
        return False
        
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(raw_data_path)
    
    # Set style matplotlib & seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.titlesize': 14,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # DISTRIBUSI LATENSI 
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot KDE untuk Control & Variant
    sns.kdeplot(data=df['Control_Latency_ms'], fill=True, color="#3B82F6", label="Grup A (Control - Dasar)", alpha=0.5, linewidth=2, ax=ax)
    sns.kdeplot(data=df['Variant_Latency_ms'], fill=True, color="#EF4444", label="Grup B (Variant - Kustom)", alpha=0.5, linewidth=2, ax=ax)
    
    # Tambahkan garis rata-rata
    mean_c = df['Control_Latency_ms'].mean()
    mean_v = df['Variant_Latency_ms'].mean()
    ax.axvline(mean_c, color="#1D4ED8", linestyle="--", linewidth=1.5, label=f"Rerata Control ({mean_c:.3f} ms)")
    ax.axvline(mean_v, color="#B91C1C", linestyle="--", linewidth=1.5, label=f"Rerata Variant ({mean_v:.3f} ms)")
    
    ax.set_title("Distribusi Kecepatan Proses Preprocessing (Latency)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Waktu Pemrosesan per Teks (milidetik)", labelpad=10)
    ax.set_ylabel("Kerapatan Probabilitas (Density)", labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=9.5)
    
    # Despine
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    plot_lat_path = os.path.join(plots_dir, "latency_comparison.png")
    plt.savefig(plot_lat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot Latency disimpan di: {plot_lat_path}")
    
    # PERBANDINGAN PANJANG TEKS 
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    melt_df = df[['Original_Word_Count', 'Control_Word_Count', 'Variant_Word_Count']].melt(
        var_name='Kategori', value_name='Panjang_Kata'
    )
    kategori_map = {
        'Original_Word_Count': 'Teks Mentah (Raw)',
        'Control_Word_Count': 'Grup A\n(Control - Dasar)',
        'Variant_Word_Count': 'Grup B\n(Variant - Lanjutan)'
    }
    melt_df['Kategori'] = melt_df['Kategori'].map(kategori_map)
    
    palette_colors = ["#94A3B8", "#60A5FA", "#34D399"]
    sns.boxplot(
        x='Kategori', y='Panjang_Kata', data=melt_df,
        palette=palette_colors, width=0.5, ax=ax,
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
    )
    
    ax.set_title("Analisis Komparasi Kepadatan Informasi (Panjang Kata)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Tahap / Grup Eksperimen", labelpad=10)
    ax.set_ylabel("Jumlah Kata per Dokumen", labelpad=10)
    
    # Despine
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    plot_comp_path = os.path.join(plots_dir, "compression_comparison.png")
    plt.savefig(plot_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot Kompresi Kata disimpan di: {plot_comp_path}")
    
    return True

if __name__ == "__main__":
    generate_plots()
