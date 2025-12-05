import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="K-Means Clustering - Movie Dataset", layout="wide")

st.title("🎬 K-Means Clustering on Movie Dataset")
st.write("Aplikasi ini menampilkan hasil clustering K-Means berdasarkan dataset yang sudah dipreprocessing sesuai PDF.")

MODEL_DIR = Path("models_pdf_pipeline")

# Load dataset
DATA_PATH = "movies_clean_saved_for_app.csv"
try:
    df = pd.read_csv(DATA_PATH, index_col=0)
    st.success("Dataset berhasil dimuat!")
except:
    st.error(f"Dataset '{DATA_PATH}' tidak ditemukan. Pastikan file ada di folder project.")
    st.stop()

# Features dipakai untuk K-Means
cluster_features = ["profit_log", "budget_log", "votes_log", "rating"]

if not all(f in df.columns for f in cluster_features):
    st.error("Kolom clustering tidak lengkap dalam dataset. Pastikan preprocessing benar.")
    st.stop()

# Load model & scaler
try:
    kmeans = joblib.load(MODEL_DIR / "kmeans_pdf.joblib")
    scaler = joblib.load(MODEL_DIR / "kmeans_scaler.joblib")
    st.success("Model & scaler berhasil dimuat!")
except:
    st.error("Model K-Means atau scaler tidak ditemukan. Pastikan file ada di folder models_pdf_pipeline.")
    st.stop()

# Standardize
X = df[cluster_features].dropna()
X_scaled = scaler.transform(X)

# Predict clusters
clusters = kmeans.predict(X_scaled)
df["cluster"] = clusters

# Show cluster summary
st.subheader("📊 Cluster Summary")
cluster_summary = df.groupby("cluster")[cluster_features].mean().round(2)
st.dataframe(cluster_summary)

# PCA for visualization
st.subheader("🌀 PCA Visualization of Clusters")

pca = PCA(n_components=2)
pca_vals = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10,6))
scatter = ax.scatter(pca_vals[:,0], pca_vals[:,1], c=clusters, cmap='tab10')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

ax.set_title("K-Means Clusters (PCA 2D)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

st.pyplot(fig)

# Show full data
st.subheader("📄 Dataset dengan Label Cluster")
st.dataframe(df.head(30))

st.info("Aplikasi ini menggunakan preprocessing dan model K-Means dari pipeline PDF final project.")
