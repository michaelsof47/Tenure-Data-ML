# Tenure Data Learning ML

Tenure Data Learning merupakan data learning hasil dari **data.csv** dan hasil dari data akan menjadi model machine learning dalam bentuk **Keras**

## Cara Menjalankan

Cukup gunakan **python show_data.py** untuk build data model. Namun, harus jalankan **python insert.py** untuk memasukkan data .csv ke postgreSQL

## Panduan

Berikut ini merupakan panduan dari **comment** pada **show_data.py**

1. Konfigurasi Database & Akses Database : Untuk mengkonfigurasikan database 
2. Cek Dataset.... : Ini untuk mengecek apakah kolom "biaya bulanan" ada data yang kosong atau tidak
3. SQL Analisis : Ini untuk analisis data query dengan SQL beserta visualisasi dari matplotlib
4. Data Quality Check : validasi data query apakah lolos pengecekkan
5. Featuring Engineering : Mengkonversi data query dari text menjadi angka yang mudah dimengerti oleh ML
6. Training : Melatih model data dari Featuring Engineering dengan metode LogisticRegression

## Note

Pastikan anda sudah menginstal **PostgreSQL** di komputer/Laptop lokal anda
