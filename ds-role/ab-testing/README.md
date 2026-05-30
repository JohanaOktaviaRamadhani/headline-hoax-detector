# 📊 Dokumentasi A/B Testing: Preprocessing & Data Wrangling

Eksperimen A/B Testing ini dirancang khusus untuk peran **Data Scientist** pada proyek **Headline Hoax Detector**. Eksperimen ini membandingkan keandalan dan efisiensi waktu dari dua skema pembersihan data (*data cleaning*) yang berbeda sebelum data disalurkan ke model *Machine Learning*.

---

## 1. Latar Belakang & Tujuan Eksperimen

Proses pembersihan teks (*preprocessing*) dalam proyek NLP memegang peranan krusial dalam membentuk kualitas data yang diumpankan ke model ML. Namun, preprocessing lanjutan (seperti formalisasi kata slang lokal dan pembersihan stopwords kustom) menambahkan beban komputasi (*overhead*). 

Eksperimen ini bertujuan untuk **membuktikan secara statistik** apakah penambahan pembersihan teks tingkat lanjut (Variant B) memberikan nilai kompresi kata (pembuangan *noise*) yang signifikan dibandingkan pembersihan standar (Control A), dan apakah biaya performa waktu (*latency*) yang dibayar sepadan.

---

## 2. Metodologi Eksperimen (Experimental Design)

Eksperimen dilakukan dengan membagi korpus teks berita dari dataset utama (`final-clickbait.csv`) ke dalam dua skema preprocessing secara paralel:

| Parameter | Grup A (Control) | Grup B (Variant) |
| :--- | :--- | :--- |
| **Deskripsi** | Preprocessing Dasar (*Baseline*) | Preprocessing Lanjutan (Kustom) |
| **Metode Pembersihan** | • Mengubah ke huruf kecil (*lowercase*)<br>• Menghapus tanda baca standar | • Mengubah ke huruf kecil (*lowercase*)<br>• Menghapus URL, Mention (@), Hashtag (#), angka & simbol dengan regex<br>• Formalisasi kata slang lokal (kamus slang)<br>• Stopword removal kustom (Sastrawi + kustom) |

### Variabel Eksperimen:
*   **Variabel Bebas (Independent Variable)**: Jenis metode preprocessing yang digunakan (Dasar vs Lanjutan).
*   **Variabel Terikat (Dependent Variables)**:
    1.  *Processing Latency* (Waktu eksekusi pembersihan teks dalam milidetik).
    2.  *Vocabulary Compression Rate* (Persentase pengurangan kata setelah pembersihan).

---

## 3. Hipotesis Statistik

### Eksperimen 1: Kecepatan Komputasi (Processing Latency)
*   **Hipotesis Nol ($H_0$)**: Tidak ada perbedaan rata-rata waktu proses (*latency*) yang signifikan secara statistik antara Grup A dan Grup B ($μ_A = μ_B$).
*   **Hipotesis Alternatif ($H_1$)**: Grup B membutuhkan waktu proses rata-rata yang secara signifikan lebih lambat dibanding Grup A ($μ_B > μ_A$).
*   **Uji Statistik**: *Independent Two-Sample T-Test* (karena latensi antar grup bersifat independen).

### Eksperimen 2: Kepadatan Informasi (Compression Rate)
*   **Hipotesis Nol ($H_0$)**: Tidak ada perbedaan rata-rata jumlah kata tersisa per artikel yang signifikan secara statistik antara Grup A dan Grup B ($μ_A = μ_B$).
*   **Hipotesis Alternatif ($H_1$)**: Grup B memangkas jumlah kata tidak penting secara signifikan lebih banyak dibandingkan Grup A ($μ_B < μ_A$).
*   **Uji Statistik**: *Paired/Related T-Test* (karena menguji dokumen yang sama sebelum dan sesudah diproses).

---

## 4. Cara Menjalankan Eksperimen

1.  Buka terminal/PowerShell di direktori utama proyek (`headline-hoax-detector`).
2.  Jalankan pipeline pengujian untuk memproses sampel data dan melakukan kalkulasi statistik:
    ```bash
    python ab-testing/ab_testing_pipeline.py
    ```
3.  Jalankan script visualisasi untuk membuat grafik perbandingan:
    ```bash
    python ab-testing/visualize_results.py
    ```
4.  Hasil plot grafis akan disimpan di folder `ab-testing/plots/` dan data mentah tersimpan di `ab-testing/results/`.

---

## 5. Interpretasi Hasil Uji Statistik

Hasil uji statistik dievaluasi dengan menggunakan nilai batas signifikansi **Alpha ($\alpha$) = 0.05** (tingkat kepercayaan 95%):

*   **Jika $P\text{-Value} < 0.05$**: Kita **Menolak Hipotesis Nol ($H_0$)**. Ini berarti perbedaan performa/kompresi antara Grup A dan Grup B adalah **nyata dan signifikan secara ilmiah**, bukan karena faktor kebetulan.
*   **Jika $P\text{-Value} \ge 0.05$**: Kita **Gagal Menolak Hipotesis Nol ($H_0$)**. Perbedaan rata-rata yang tipis kemungkinan terjadi karena variasi acak sampling.

Rekomendasi final eksperimen ini secara otomatis dihitung di konsol terminal untuk membantu Anda memutuskan pilihan preprocessing terbaik bagi model *Machine Learning* Anda berikutnya!
