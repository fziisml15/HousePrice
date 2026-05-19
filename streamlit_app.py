import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ==========================================
# LOAD MODEL DAN FEATURE NAMES
# ==========================================
BASE_PATH = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_PATH, "house_price_xgboost.pkl")
FEATURE_PATH = os.path.join(BASE_PATH, "feature_names.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Load nama fitur
feature_names = joblib.load(FEATURE_PATH)

# ==========================================
# HEADER
# ==========================================
st.title("🏠 House Price Prediction")
st.write(
    """
    Aplikasi ini memprediksi harga rumah menggunakan model XGBoost.
    Masukkan karakteristik rumah di bawah ini, lalu klik tombol **Predict Price**.
    """
)

# ==========================================
# INPUT USER
# ==========================================
st.subheader("📋 Input Karakteristik Rumah")

overall_qual = st.slider(
    "Overall Quality (1 = Sangat Buruk, 10 = Sangat Baik)",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.number_input(
    "Ground Living Area (sq ft)",
    min_value=500,
    max_value=5000,
    value=1500
)

garage_cars = st.slider(
    "Garage Capacity (Jumlah Mobil)",
    min_value=0,
    max_value=5,
    value=2
)

total_bsmt_sf = st.number_input(
    "Total Basement Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=800
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)

# ==========================================
# PREDIKSI
# ==========================================
if st.button("🔮 Predict Price"):

    # Buat dataframe kosong sesuai seluruh fitur model
    X_input = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Isi fitur yang tersedia dari input user
    input_features = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "YearBuilt": year_built
    }

    for feature, value in input_features.items():
        if feature in X_input.columns:
            X_input.loc[0, feature] = value

    # Prediksi (hasil model masih dalam log scale)
    pred_log = model.predict(X_input)[0]

    # Kembalikan ke harga asli
    predicted_price = np.expm1(pred_log)

    # ==========================================
    # TAMPILKAN HASIL
    # ==========================================
    # Konversi USD ke Rupiah (kurs asumsi Rp16.500 per USD)
    usd_to_idr = 16500
    predicted_price_idr = predicted_price * usd_to_idr

    # Tampilkan hasil dalam USD dan Rupiah
    st.success(f"Dollar Estimated House Price (USD): ${predicted_price:,.2f}")
    st.success(f"Rupiah Estimated House Price (IDR): Rp {predicted_price_idr:,.0f}")

    st.subheader("📊 Detail Input")
    st.write({
        "Overall Quality": overall_qual,
        "Living Area (sq ft)": gr_liv_area,
        "Garage Capacity": garage_cars,
        "Basement Area (sq ft)": total_bsmt_sf,
        "Year Built": year_built
    })

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption(
    "Model: XGBoost Regressor | Dataset: Kaggle House Prices | "
    "R² Score ≈ 0.9057"
)