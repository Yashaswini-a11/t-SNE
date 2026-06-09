import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pathlib import Path

st.set_page_config(
    page_title="Fashion MNIST t-SNE",
    layout="wide"
)

try:
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

BASE_DIR = Path(__file__).parent

df = pd.read_csv(
    BASE_DIR / "fashion_tsne_output.csv"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Dataset Info",
        "t-SNE Visualization"
    ]
)

if page == "Dashboard":

    st.title("Fashion MNIST t-SNE Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

elif page == "Dataset Info":

    st.title("Dataset Information")

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(df.columns.tolist())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

elif page == "t-SNE Visualization":

    st.title("t-SNE Visualization")

    sample_size = st.slider(
        "Select Sample Size",
        500,
        min(3000, len(df)),
        1000,
        step=500
    )

    sample_df = df.sample(
        sample_size,
        random_state=42
    )

    X = sample_df.copy()

    X = X.select_dtypes(include=["number"])

    X = X.fillna(0)

    X = X.replace(
        [float("inf"), float("-inf")],
        0
    )

    st.write("Shape:", X.shape)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame(
        X_tsne,
        columns=["TSNE1", "TSNE2"]
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.scatterplot(
        data=tsne_df,
        x="TSNE1",
        y="TSNE2",
        ax=ax
    )

    st.pyplot(fig)