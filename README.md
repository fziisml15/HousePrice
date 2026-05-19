# Prediksi Harga Rumah Menggunakan XGBoost

Project machine learning untuk memprediksi harga rumah menggunakan dataset **Kaggle House Prices: Advanced Regression Techniques**. Project ini mencakup seluruh tahapan data science mulai dari data cleaning, exploratory data analysis (EDA), feature engineering, training model, evaluasi, hingga deployment menggunakan Streamlit.

---

## Gambaran Project

Tujuan project ini adalah memprediksi harga jual rumah (`SalePrice`) berdasarkan berbagai karakteristik rumah, seperti:

- Kualitas keseluruhan rumah (`OverallQual`)
- Luas area hunian (`GrLivArea`)
- Kapasitas garasi (`GarageCars`)
- Luas basement (`TotalBsmtSF`)
- Tahun pembangunan (`YearBuilt`)
- Lokasi lingkungan (`Neighborhood`)
- Dan berbagai fitur lainnya

Model yang digunakan adalah **XGBoost Regressor**, yang mampu memberikan performa prediksi yang sangat baik.

---

## Dataset

Sumber dataset:
- Kaggle – House Prices: Advanced Regression Techniques
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Informasi dataset:
- 1.460 data rumah
- 81 fitur
- Target: `SalePrice`

---

## Alur Pengerjaan

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Training Model dengan XGBoost
5. Evaluasi Model
6. Pembuatan Aplikasi Streamlit
7. Deployment

---

## Model yang Digunakan

Algoritma:
- XGBoost Regressor

Tahapan preprocessing:
- Penanganan missing values
- Penghapusan data duplikat
- One-Hot Encoding
- Transformasi log pada target (`SalePrice`)

---

## Hasil Model

| Metrik | Nilai |
|------|------:|
| RMSE (log scale) | 0.1327 |
| R² Score | 0.9057 |

Model mampu menjelaskan sekitar **90,57% variasi harga rumah**.

---

## Aplikasi Streamlit

Project ini dilengkapi dengan aplikasi web interaktif menggunakan Streamlit. Pengguna dapat memasukkan karakteristik rumah dan mendapatkan estimasi harga dalam:

- Dolar Amerika (USD)
- Rupiah Indonesia (IDR)

Fitur aplikasi:
- Antarmuka interaktif
- Prediksi real-time
- Konversi USD ke Rupiah
- Ringkasan input pengguna

---

## Struktur Project

HousePrice/
├── actual_vs_predicted.png
├── residual_distribution.png
├── feature_importance.csv
├── feature_names.pkl
├── house_price_xgboost.pkl
├── streamlit_app.py
├── requirements.txt
└── README.md