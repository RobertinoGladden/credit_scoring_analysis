# Laporan Proyek Machine Learning - Predictive Analytics
Nama Lengkap: Robertino Gladden Narendra

Cohort ID   : MC012D5Y2381

## Project Overview

Analisis risiko kredit merupakan salah satu aspek penting dalam industri keuangan, khususnya bagi bank dan lembaga pemberi pinjaman. Kemampuan untuk memprediksi apakah seorang peminjam akan gagal bayar (default) atau berhasil melunasi pinjaman (non-default) dapat membantu mengurangi kerugian finansial dan meningkatkan efisiensi pengambilan keputusan. Proyek ini menggunakan dataset risiko kredit untuk membangun model machine learning yang dapat memprediksi status pinjaman (gagal bayar atau lunas) berdasarkan fitur seperti usia, pendapatan, jumlah pinjaman, dan riwayat kredit peminjam.

**Mengapa Masalah Ini Penting?**  
Prediksi yang akurat dapat membantu lembaga keuangan mengelola risiko, mengurangi tingkat gagal bayar, dan memberikan pinjaman kepada peminjam yang lebih layak. Menurut penelitian oleh [1], tingkat gagal bayar pada pinjaman konsumen dapat menyebabkan kerugian signifikan bagi bank jika tidak dikelola dengan baik. Oleh karena itu, pendekatan berbasis data dan machine learning menjadi solusi yang relevan untuk masalah ini.

**Referensi**:  
[1] A. Khashman, "Neural networks for credit risk evaluation: Investigation of different neural models and learning schemes," *Expert Systems with Applications*, vol. 37, no. 9, pp. 6233-6239, 2010. [Online]. Available: https://doi.org/10.1016/j.eswa.2010.02.101

## Business Understanding

### Problem Statements
1. **Tingkat Gagal Bayar yang Tinggi**: Lembaga keuangan sering menghadapi risiko gagal bayar dari peminjam, yang dapat menyebabkan kerugian finansial signifikan.
2. **Kurangnya Prediksi Akurat**: Tanpa model prediktif yang andal, sulit untuk mengidentifikasi peminjam yang berisiko tinggi sebelum memberikan pinjaman.
3. **Efisiensi Pengambilan Keputusan**: Proses penilaian kredit secara manual memakan waktu dan kurang efisien, terutama dengan volume aplikasi pinjaman yang besar.

### Goals
1. **Membangun Model Prediktif**: Mengembangkan model machine learning untuk memprediksi status pinjaman (gagal bayar atau lunas) dengan akurasi tinggi.
2. **Mengurangi Risiko Gagal Bayar**: Mengidentifikasi peminjam yang berisiko tinggi untuk membantu bank membuat keputusan pemberian pinjaman yang lebih baik.
3. **Meningkatkan Efisiensi**: Mengotomatiskan proses penilaian risiko kredit untuk mempercepat pengambilan keputusan.

### Solution Approach
1. **Random Forest Classifier**: Menggunakan algoritma Random Forest karena kemampuannya menangani dataset yang tidak seimbang dan fitur yang beragam.
2. **Logistic Regression**: Sebagai alternatif, model regresi logistik dapat digunakan untuk memberikan probabilitas gagal bayar, yang lebih mudah diinterpretasikan oleh pemangku kepentingan non-teknis.

## Data Understanding

Dataset yang digunakan adalah **Credit Risk Dataset** yang tersedia di [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). Dataset ini berisi 32.581 baris data dengan 12 kolom, termasuk informasi demografis peminjam, detail pinjaman, dan status pinjaman. Terdapat beberapa nilai yang hilang pada kolom `loan_int_rate` (3.116 missing) dan `person_emp_length` (895 missing), serta beberapa outlier (misalnya, usia > 100 tahun atau masa kerja > 50 tahun).

**Variabel pada Dataset**:
- `person_age`: Usia peminjam (numerik, tahun).
- `person_income`: Pendapatan tahunan peminjam (numerik, USD).
- `person_home_ownership`: Status kepemilikan rumah (kategorikal: RENT, OWN, MORTGAGE, OTHER).
- `person_emp_length`: Lama bekerja peminjam (numerik, tahun).
- `loan_intent`: Tujuan pinjaman (kategorikal: PERSONAL, EDUCATION, MEDICAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT, VENTURE).
- `loan_grade`: Tingkat kualitas pinjaman (kategorikal: A, B, C, D, E, F, G).
- `loan_amnt`: Jumlah pinjaman (numerik, USD).
- `loan_int_rate`: Suku bunga pinjaman (numerik, persen).
- `loan_status`: Status pinjaman (target, biner: 0 = Lunas, 1 = Gagal Bayar).
- `loan_percent_income`: Rasio pinjaman terhadap pendapatan (numerik).
- `cb_person_default_on_file`: Riwayat gagal bayar sebelumnya (kategorikal: Y, N).
- `cb_person_cred_hist_length`: Panjang riwayat kredit peminjam (numerik, tahun).

**Exploratory Data Analysis (EDA)**:
- Distribusi `loan_status` menunjukkan ketidakseimbangan, dengan sekitar 78% data berstatus lunas (`loan_status = 0`) dan 22% gagal bayar (`loan_status = 1`).
- Fitur seperti `loan_int_rate` dan `loan_percent_income` memiliki korelasi positif dengan gagal bayar, yang mengindikasikan bahwa suku bunga tinggi dan rasio pinjaman besar meningkatkan risiko.

## Data Preparation

Berikut adalah tahapan persiapan data yang dilakukan, sesuai urutan dalam notebook:

1. **Penanganan Nilai Hilang**:
   - Kolom `loan_int_rate` dan `person_emp_length` diisi dengan nilai median untuk menjaga distribusi data.
   - **Alasan**: Median lebih robust terhadap outlier dibandingkan mean.

2. **Penghapusan Outlier**:
   - Data dengan `person_age` > 100 tahun dan `person_emp_length` > 50 tahun dihapus karena dianggap tidak realistis.
   - **Alasan**: Outlier dapat memengaruhi performa model dan mengurangi generalisasi.

3. **Encoding Variabel Kategorikal**:
   - Kolom kategorikal (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`) diubah menjadi numerik menggunakan `LabelEncoder`.
   - **Alasan**: Model machine learning memerlukan input numerik.

4. **Pemisahan Fitur dan Target**:
   - Fitur (X) adalah semua kolom kecuali `loan_status`, dan target (y) adalah `loan_status`.
   - **Alasan**: Untuk memisahkan variabel independen dan dependen.

5. **Pembagian Data**:
   - Data dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split`.
   - **Alasan**: Untuk mengevaluasi performa model pada data yang belum dilihat.

6. **Skalasi Fitur**:
   - Fitur numerik diskalakan menggunakan `StandardScaler` untuk menstandarisasi skala.
   - **Alasan**: Random Forest tidak terlalu sensitif terhadap skala, tetapi skalasi memastikan konsistensi jika model lain digunakan.

## Modeling

Model utama yang digunakan adalah **RandomForestClassifier** dengan 100 pohon keputusan (`n_estimators=100`). Random Forest dipilih karena kemampuannya menangani dataset yang tidak seimbang, menangkap hubungan non-linear, dan memberikan performa yang robust.

**Solusi Alternatif**:  
Sebagai perbandingan, **Logistic Regression** juga dapat digunakan. Logistic Regression memberikan probabilitas gagal bayar yang lebih mudah diinterpretasikan, tetapi kurang fleksibel dalam menangani hubungan non-linear dibandingkan Random Forest.

**Kelebihan dan Kekurangan**:
- **Random Forest**:
  - **Kelebihan**: Robust terhadap outlier, menangani ketidakseimbangan kelas, dan memberikan akurasi tinggi.
  - **Kekurangan**: Kompleksitas komputasi lebih tinggi dan kurang interpretabel.
- **Logistic Regression**:
  - **Kelebihan**: Mudah diinterpretasikan, cepat dalam pelatihan, dan cocok untuk probabilitas risiko.
  - **Kekurangan**: Performa lebih rendah pada data dengan hubungan non-linear.

**Output**:  
Model menghasilkan prediksi status pinjaman (gagal bayar atau lunas) untuk data uji dan sampel baru. Contoh output untuk tiga sampel uji:
- Sampel 1: Prediksi Status Pinjaman = Lunas
- Sampel 2: Prediksi Status Pinjaman = Lunas
- Sampel 3: Prediksi Status Pinjaman = Lunas

## Evaluation

**Metrik Evaluasi**:
- **Accuracy**: Mengukur persentase prediksi yang benar.
- **Precision, Recall, F1-Score**: Mengukur performa model pada kelas minoritas (gagal bayar).
- **Confusion Matrix**: Menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.

**Hasil Evaluasi**:
- **Accuracy**: 93.38%, menunjukkan model cukup akurat secara keseluruhan.
- **Classification Report**:
  - Kelas 0 (Lunas): Precision = 0.93, Recall = 0.99, F1-Score = 0.96
  - Kelas 1 (Gagal Bayar): Precision = 0.97, Recall = 0.72, F1-Score = 0.83
- **Confusion Matrix**:
  ```
  [[5063   36]  # Lunas: 5063 benar, 36 salah
   [ 395 1021]] # Gagal Bayar: 1021 benar, 395 salah
  ```
- **Analisis**: Model sangat baik dalam memprediksi kelas lunas (recall 0.99), tetapi recall untuk gagal bayar (0.72) menunjukkan beberapa kasus gagal bayar tidak terdeteksi. Ini dapat ditingkatkan dengan menangani ketidakseimbangan kelas lebih lanjut (misalnya, oversampling).

**Kesimpulan**:  
Model Random Forest memberikan performa yang baik untuk memprediksi risiko kredit, dengan akurasi tinggi dan presisi yang kuat untuk kelas gagal bayar. Namun, recall untuk kelas gagal bayar dapat ditingkatkan untuk mengurangi false negatives, yang penting dalam konteks risiko kredit.

---