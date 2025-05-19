import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_data():
    files = {
        "Benin": "data/benin_clean.csv",
        "Togo": "data/togo_clean.csv",
        "Sierra Leone": "data/sierraleone_clean.csv"
    }

    dfs = []
    for country, path in files.items():
        df = pd.read_csv(path)
        df["country"] = country
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_ghi_comparison(df):
    st.subheader("📊 GHI Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x="country", y="GHI", data=df, ax=ax)
    st.pyplot(fig)
