# Panduan Penggunaan Aplikasi

Dokumen ini menjelaskan cara menjalankan dan menggunakan aplikasi Streamlit **Credit Risk Intelligence Battle**.

## 1. Gambaran Umum

Aplikasi ini adalah dashboard interaktif untuk membandingkan hasil prediksi risiko kredit dari tiga tahap model:

1. **Tahap 1 - Manual FIS**: Fuzzy Inference System berbasis aturan manual.
2. **Tahap 2 - GA FIS**: FIS yang parameternya dituning menggunakan Genetic Algorithm.
3. **Tahap 3 - ANN FIS**: FIS yang dikombinasikan dengan pendekatan neural network.

Melalui aplikasi ini, pengguna dapat melihat ringkasan metrik, confusion matrix, distribusi skor, hasil ablation study, parameter membership function, hingga isi file hasil prediksi.

## 2. Persiapan

Pastikan struktur folder proyek sudah lengkap, terutama:

- `data/credit_risk_dataset.csv`
- isi folder `results/`
- file aplikasi `app-streamlit/app.py`

Jika ingin menjalankan aplikasi dari awal, instal dependensi terlebih dahulu:

```bash
pip install -r requirements.txt
```

## 3. Menjalankan Aplikasi

Jalankan perintah berikut dari root project:

```bash
streamlit run app-streamlit/app.py
```

Setelah itu browser akan terbuka otomatis, atau Anda bisa membuka alamat lokal yang ditampilkan Streamlit, biasanya:

```text
http://localhost:8501
```

## 4. Struktur Tampilan Aplikasi

Aplikasi memiliki beberapa bagian utama di sidebar dan tab utama.

### 4.1 Sidebar Controls

Di sidebar, Anda dapat mengatur mode prediksi:

- **Use saved predictions**
  - Aplikasi memakai kolom prediksi yang sudah tersimpan di file CSV hasil tiap tahap.
  - Ini adalah mode paling sederhana dan cocok untuk melihat hasil akhir apa adanya.

- **Use score threshold**
  - Aplikasi menghitung prediksi ulang berdasarkan kolom skor.
  - Anda dapat memilih kolom skor dan mengatur threshold untuk masing-masing tahap.

Jika mode threshold dipilih, setiap tahap akan memiliki kontrol berikut:

- **Score column**: pilih kolom skor yang dipakai untuk klasifikasi.
- **Threshold**: atur batas keputusan prediksi.

## 5. Penjelasan Tiap Tab

### 5.1 Tab Overview

Tab ini menampilkan ringkasan utama hasil aplikasi.

Fitur yang tersedia:

- **Summary**
  - Menampilkan metrik dari file `summary_tahap2.json` jika tersedia.
  - Jika file tidak tersedia, aplikasi tetap menampilkan metrik yang dihitung dari CSV hasil tahap.

- **Metrics from Result CSVs**
  - Menampilkan tabel metrik untuk setiap tahap:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
  - Juga menampilkan grafik perbandingan antar tahap.

- **Confusion Matrix and Score Distribution**
  - Pilih tahap yang ingin dianalisis.
  - Confusion matrix memperlihatkan jumlah prediksi benar dan salah.
  - Jika mode threshold aktif, aplikasi juga menampilkan distribusi skor beserta garis threshold.

### 5.2 Tab Ablation

Tab ini digunakan untuk melihat hasil **ablation study** dari tahap tuning.

Fitur yang tersedia:

- tabel hasil ablation dari `ablation_tahap2.json`
- pemilihan setting eksperimen
- grafik **loss history** bila tersedia
- scatter plot perbandingan **accuracy vs f1_score**

Tab ini berguna untuk membandingkan pengaruh tiap konfigurasi terhadap performa model.

### 5.3 Tab Membership Functions

Tab ini menampilkan parameter membership function yang tersimpan di file JSON.

Fitur yang tersedia:

- pilih tahap model
- pilih variabel input
- lihat tabel parameter triangular membership function `(a, b, c)`
- lihat visualisasi kurva membership function

Jika file parameter tidak tersedia, aplikasi akan menampilkan pemberitahuan bahwa data belum ditemukan.

### 5.4 Tab Data Explorer

Tab ini digunakan untuk memeriksa data hasil prediksi secara lebih detail.

Filter yang tersedia:

- **Loan status**: filter label asli (`0` atau `1`)
- **Prediction**: filter hasil prediksi (`0` atau `1`)
- **Score column**: pilih kolom skor yang ingin dianalisis
- **Score range**: filter berdasarkan rentang skor
- Filter tambahan untuk kolom kategorikal seperti:
  - `loan_intent`
  - `person_home_ownership`
  - `loan_grade`
  - `cb_person_default_on_file`

Di bagian bawah, aplikasi menampilkan tabel data yang sudah difilter.

### 5.5 Tab Artifacts

Tab ini menampilkan file gambar yang tersimpan di folder `results/plots/`.

Fitur yang tersedia:

- daftar gambar yang bisa dipilih
- preview gambar dalam aplikasi
- berguna untuk melihat hasil visualisasi yang sudah dibuat sebelumnya

## 6. Cara Membaca Hasil

Saat menggunakan aplikasi, perhatikan beberapa hal berikut:

- **Accuracy** menunjukkan persentase prediksi yang benar secara keseluruhan.
- **Precision** penting jika ingin mengetahui seberapa akurat prediksi kelas positif.
- **Recall** penting jika ingin menangkap sebanyak mungkin kasus kelas positif.
- **F1 Score** cocok untuk melihat keseimbangan precision dan recall.
- **Confusion matrix** membantu melihat jenis kesalahan prediksi.
- **Score distribution** membantu mengevaluasi apakah threshold yang digunakan terlalu ketat atau terlalu longgar.

## 7. Jika Data Tidak Muncul

Jika aplikasi menampilkan pesan bahwa file tidak ditemukan, periksa hal berikut:

1. Pastikan folder `results/` berisi file CSV dan JSON yang dibutuhkan.
2. Pastikan nama kolom pada file hasil sesuai dengan yang diharapkan aplikasi.
3. Pastikan file `credit_risk_dataset.csv` ada di folder `data/`.
4. Jalankan aplikasi dari root project agar path relatif terbaca dengan benar.

## 8. Ringkasan File Penting

File utama yang dipakai aplikasi:

- `results/hasil_tahap1_manual_fis.csv`
- `results/hasil_tahap2_ga_fis.csv`
- `results/hasil_tahap3_ann_fis.csv`
- `results/summary_tahap2.json`
- `results/ablation_tahap2.json`
- `results/mf_params_tahap1.json`
- `results/mf_params_tahap2.json`
- `results/mf_params_tahap3_ann.json`
- `results/plots/`

## 9. Alur Penggunaan yang Disarankan

Urutan penggunaan yang paling praktis:

1. Jalankan aplikasi dengan Streamlit.
2. Buka tab **Overview** untuk melihat perbandingan metrik utama.
3. Gunakan tab **Ablation** untuk melihat hasil eksperimen tuning.
4. Buka tab **Membership Functions** untuk memeriksa parameter fuzzy.
5. Gunakan tab **Data Explorer** jika ingin menelusuri baris data tertentu.
6. Cek tab **Artifacts** untuk melihat file visualisasi yang sudah disimpan.

## 10. Catatan

- Aplikasi akan tetap berjalan meski sebagian file hasil belum lengkap, tetapi beberapa tab mungkin menampilkan informasi kosong atau pesan peringatan.
- Jika Anda mengubah isi folder `results/`, refresh aplikasi agar perubahan terbaca.
- Mode threshold berguna jika Anda ingin membandingkan performa dengan batas keputusan yang berbeda.
