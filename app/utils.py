import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where main.py lives

    data_path_benin = os.path.join(BASE_DIR, '..', 'data', 'benin_clean.csv')
    data_path_togo = os.path.join(BASE_DIR, '..', 'data', 'togo_clean.csv')
    data_path_sl = os.path.join(BASE_DIR, '..', 'data', 'sierraleone_clean.csv')

    df_benin = pd.read_csv(data_path_benin)
    df_togo = pd.read_csv(data_path_togo)
    df_sl = pd.read_csv(data_path_sl)

    df_benin["country"] = "Benin"
    df_togo["country"] = "Togo"
    df_sl["country"] = "Sierra Leone"

    df = pd.concat([df_benin, df_togo, df_sl], ignore_index=True)
    return df

def plot_ghi_comparison(df, metric):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="country", y=metric, ax=ax, palette="Set2")
    ax.set_title(f"{metric} Distribution by Country", fontsize=16)
    return fig
