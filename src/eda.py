import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from data_loader import load_all, clean_results, clean_qualifying
from feature_engineering import build_features


FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def save(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print("saved:", name)


def plot_grid_vs_finish(df: pd.DataFrame):
    sample = df[df["grid"].notna() & df["positionOrder"].notna()].sample(
        min(5000, len(df)), random_state=42
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(sample["grid"], sample["positionOrder"], alpha=0.15, s=12)

    ax.set_xlabel("Grid")
    ax.set_ylabel("Finish")
    ax.set_title("Grid vs Finish")

    corr = sample["grid"].corr(sample["positionOrder"])

    ax.text(
        0.05, 0.92, f"r = {corr:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    save("grid_vs_finish.png")


def plot_podium_rate_by_grid(df: pd.DataFrame):
    
    top20 = df[df["grid"].between(1, 20)]
    
    rates = top20.groupby("grid")["podium"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(rates["grid"], rates["podium"] * 100)
    ax.set_xlabel("Grid")
    ax.set_ylabel("Podium %")
    ax.set_title("Podium rate by grid")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    save("podium_rate_by_grid.png")


def plot_top_drivers(df: pd.DataFrame, drivers: pd.DataFrame, n: int = 15):
    wins = (
        df[df["win"] == 1]
        .groupby("driverId")["win"].sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    wins = wins.merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left",)

    wins["name"] = wins["forename"] + " " + wins["surname"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(wins["name"][::-1], wins["win"][::-1])

    ax.set_xlabel("Wins")
    ax.set_title(f"Top {n} drivers")

    save("top_drivers.png")


def plot_wins_per_year(df: pd.DataFrame):
    races_per_year = df.drop_duplicates("raceId").groupby("year").size()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(races_per_year.index, races_per_year.values, marker="o", markersize=3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Races")
    ax.set_title("Races per season")

    save("races_per_year.png")


def plot_feature_distributions(df: pd.DataFrame, feature_columns: list[str]):
    
    numeric = [c for c in feature_columns if df[c].nunique() > 5]

    cols = 3
    rows = (len(numeric) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric):
        axes[i].hist(df[col].dropna(), bins=40)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    save("feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, feature_columns: list[str]):
    cols = feature_columns + ["podium", "win"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Correlation heatmap")
    plt.tight_layout()
    save("correlation_heatmap.png")


def plot_class_balance(df: pd.DataFrame):
    counts = df["podium"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].bar(["0", "1"], counts.values)
    axes[0].set_title("Class balance")

    axes[1].pie(counts.values, labels=["Non-podium", "Podium"], autopct="%1.1f%%")
    axes[1].set_title("Percent split")

    save("class_balance.png")


def plot_constructor_dominance(df: pd.DataFrame, constructors: pd.DataFrame, n: int = 10):
    top_c = (
        df[df["win"] == 1]
        .groupby("constructorId")["win"].sum()
        .nlargest(n)
        .index
    )

    yearly = (
        df[(df["win"] == 1) & (df["constructorId"].isin(top_c))]
        .groupby(["year", "constructorId"])["win"].sum()
        .unstack(fill_value=0)
    )

    yearly.columns = yearly.columns.map(
        lambda cid: constructors.loc[
            constructors["constructorId"] == cid, "name"
        ].values[0] if cid in constructors["constructorId"].values else str(cid)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    yearly.plot.area(ax=ax, alpha=0.75)

    ax.set_xlabel("Year")
    ax.set_ylabel("Wins")
    ax.set_title(f"Top {n} constructors")

    save("constructor_dominance.png")


def run_eda():
    tables = load_all()
    df = clean_results(tables["results"], tables["races"])
    df_model, feature_columns = build_features()

    print("running EDA now")

    plot_grid_vs_finish(df)
    plot_podium_rate_by_grid(df)
    plot_top_drivers(df, tables["drivers"])
    plot_wins_per_year(df)
    plot_feature_distributions(df_model, feature_columns)
    plot_correlation_heatmap(df_model, feature_columns)
    plot_class_balance(df_model)
    plot_constructor_dominance(df, tables["constructors"])

    print("Done")


if __name__ == "__main__":
    run_eda()