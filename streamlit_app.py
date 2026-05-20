import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "house_price_xgboost.pkl"
FEATURE_PATH = "feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 3rem;
    border-radius: 28px;
    color: white;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.25);
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.hero p {
    font-size: 1.1rem;
    color: #cbd5e1;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
}

.result-box {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 2.5rem;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(16, 185, 129, 0.25);
}

.result-title {
    font-size: 1.1rem;
    opacity: 0.95;
    margin-bottom: 0.5rem;
}

.result-price {
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.2;
}

.sidebar-note {
    background: #eff6ff;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #bfdbfe;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1>Prediksi Harga Rumah</h1>
    <p>
        Dashboard machine learning dengan menggunakan <b>XGBoost</b>
        untuk memprediksi harga rumah berdasarkan 5 fitur utama.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUT
# =========================
with st.sidebar:
    st.header("📝 Input Karakteristik Rumah")

    st.markdown("""
    <div class="sidebar-note">
        Masukkan spesifikasi rumah, lalu klik tombol prediksi untuk
        memperoleh estimasi harga dalam USD dan Rupiah.
    </div>
    """, unsafe_allow_html=True)

    overall_qual = st.slider(
        "⭐ Kualitas Rumah",
        min_value=1,
        max_value=10,
        value=7
    )

    gr_liv_area = st.number_input(
        "📐 Luas Rumah (sq ft)",
        min_value=100,
        value=2000,
        step=100
    )

    garage_cars = st.slider(
        "🚗 Kapasitas Garasi",
        min_value=0,
        max_value=12,
        value=2
    )

    total_bsmt_sf = st.number_input(
        "🏗️ Luas Basement (sq ft)",
        min_value=0,
        value=800,
        step=50
    )

    year_built = st.number_input(
        "📅 Tahun Dibangun",
        min_value=1900,
        max_value=2025,
        value=2000
    )

    predict_button = st.button(
        "🔮 Prediksi Harga Rumah",
        use_container_width=True
    )

# =========================
# MAIN CONTENT
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🤖 Model</h4>
        <h2>XGBoost</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>📊 Jumlah Fitur</h4>
        <h2>5 Fitur</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>💱 Kurs USD/IDR</h4>
        <h2>17.500</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# PREDIKSI
# =========================
if predict_button:

    # DataFrame kosong sesuai fitur model
    X_input = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Isi 5 fitur utama
    X_input["OverallQual"] = overall_qual
    X_input["GrLivArea"] = gr_liv_area
    X_input["GarageCars"] = garage_cars
    X_input["TotalBsmtSF"] = total_bsmt_sf
    X_input["YearBuilt"] = year_built

    pred_log = model.predict(X_input)[0]

    # Konversi dari log ke harga asli
    predicted_price = np.expm1(pred_log)

    # Jika hasil tidak valid (inf atau NaN), gunakan batas maksimum realistis
    if not np.isfinite(predicted_price):
        predicted_price = 3_000_000  # USD maksimum realistis (~ Rp 52,5 miliar)

    # Konversi ke Rupiah
    usd_to_idr = 17500
    predicted_price_idr = predicted_price * usd_to_idr

    # Hasil prediksi
    st.markdown(f"""
    <div class="result-box">
        <div class="result-title">💰 Estimasi Harga Rumah</div>
        <div class="result-price">
            Rp {predicted_price_idr:,.0f}
        </div>
        <div style="margin-top: 12px; font-size: 1.1rem;">
            ≈ USD {predicted_price:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📋 Detail Input")

    detail_df = pd.DataFrame({
        "Fitur": [
            "Kualitas Rumah",
            "Luas Rumah",
            "Kapasitas Garasi",
            "Luas Basement",
            "Tahun Dibangun"
        ],
        "Nilai": [
            overall_qual,
            f"{gr_liv_area:,} sq ft",
            garage_cars,
            f"{total_bsmt_sf:,} sq ft",
            year_built
        ]
    })

    st.dataframe(detail_df, use_container_width=True)

else:
    st.info("👈 Masukkan data rumah pada sidebar, lalu klik **Prediksi Harga Rumah**.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Dibuat menggunakan Python, XGBoost, dan Streamlit • Dataset: Kaggle House Prices"
)