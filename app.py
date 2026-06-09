import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

st.set_page_config(
    page_title="Fashion MNIST t-SNE",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

from pathlib import Path

BASE_DIR = Path(__file__).parent

df = pd.read_csv(
    BASE_DIR / "fashion_tsne_output.csv"
)
st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "EDA",
        "t-SNE Visualization"
    ]
)

if page == "Dashboard":

    st.title(
        "Fashion MNIST t-SNE Dashboard"
    )

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Records",
        df.shape[0]
    )

    col2.metric(
        "Features",
        df.shape[1]
    )

    col3.metric(
        "Classes",
        df["label"].nunique()
    )

    st.dataframe(
        df.head()
    )

elif page == "EDA":

    st.title(
        "Fashion Category Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.countplot(
        x="label",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

elif page == "t-SNE Visualization":

    st.title(
        "t-SNE Projection"
    )

    sample_df = df.sample(
        3000,
        random_state=42
    )

    X = sample_df.drop(
        "label",
        axis=1
    )

    y = sample_df["label"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    X_tsne = tsne.fit_transform(
        X_scaled
    )

    tsne_df = pd.DataFrame(
        X_tsne,
        columns=[
            "TSNE1",
            "TSNE2"
        ]
    )

    tsne_df["Label"] = y

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.scatterplot(
        data=tsne_df,
        x="TSNE1",
        y="TSNE2",
        hue="Label",
        palette="tab10",
        ax=ax
    )

    st.pyplot(fig)