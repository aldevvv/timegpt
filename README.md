# **📈 Stock Price Forecasting Dashboard Using TimeGPT**

Live Preview : https://forecastingtimegpt.up.railway.app/

---

#**📘 Dokumentasi - Cara Kerja & Panduan Setup**

#**📌 Ringkasan**
Dashboard Ini Memungkinkan Pengguna Untuk :
- Mengunggah Data Harga Saham Dalam Format Excel/CSV
- Melihat Visualisasi Harga Penutupan (Close) & Volume Historis
- Melakukan Prediksi Harga Saham Di Masa Depan Menggunakan **TimeGPT (Nixtla)**
- Menganalisis Akurasi Hasil Prediksi Dengan Metrik Seperti :
  - **MAPE** (Mean Absolute Percentage Error)
  - **RMSE** (Root Mean Square Error)
  - **MAE** (Mean Absolute Error)
- Mengunduh Laporan Analisis Lengkap Dalam Format PDF

**Catatan :**

1. Silahkan Unduh Historical Data Yahoo Finance ( CleanedVersion ) Disini - **https://yahoofinance.up.railway.app/**


2. Kunjungi Repository Projectnya Jika Terjadi Error / Limit - **https://github.com/aldevvv/yahoofinance**

---

#**⚙️ Cara Kerja**

1. **Unggah Data**
   - Format File Yang Didukung ( `.csv`, `.xlsx` )
   - Kolom Minimal Yang Dibutuhkan :
     - `Date` - Tanggal Dalam Format Time Series
     - `Close` - Harga Penutupan Saham
     - `Volume` - (Opsional) Volume Perdagangan

2. **Praprocessing Data**
   - Data Diurutkan Berdasarkan Tanggal
   - Duplikasi & Data Kosong Dibersihkan
   - Data di-*resample* Sesuai Frekuensi (Harian, Mingguan, atau Bulanan)

3. **Forecasting**
   - Menggunakan Model TimeGPT dari Nixtla
   - Memungkinkan Pemilihan Horizon Prediksi (7, 14, 30 hari)
   - Interval Kepercayaan 80% dan 95% by Default

4. **Analisis Residual**
   - Menghitung Selisih Antara Nilai Aktual & Prediksi (`Residual`)
   - Menampilkan :
     - Grafik Residual Dari Waktu ke Waktu
     - Plot ACF & PACF ( Untuk Autokorelasi )
     - Histogram Distribusi Error

5. **Ekspor PDF**
   - Generate Laporan Otomatis Berisi :
     - Judul, Tanggal Forecasting & Horizon Prediksi
     - Grafik Harga Historis & Prediksi
     - Metrik Performa Forecasting
     - Visualisasi Residual
   - File PDF Dapat Diunduh Langsung Dari Dashboard

---

#**🛠️ Setup & Jalankan (Google Collab Notebook)**

**1. Instal Library & Dependensi (Jika Belum)**

**2. Run Tiap Cell Secara Berurutan**

**3. Jalankan Streamlit Pyngrok Untuk Akses UI Dashboard**

#**📝 Catatan**

**Pastikan Kolom Data Sesuai Dengan Format (Date, Close, dll)**

**Koneksi Internet Diperlukan Saat Load Model TimeGPT**

#**👤 Developer**
**Developed with Love by Mahasiswa Universitas Negeri Makassar (2025) untuk Project Kelompok 8.**

**📎 Website : https://aldev.web.id**
