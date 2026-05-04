import pandas as pd
import numpy as np
from data_loader import load_all, clean_results, clean_qualifying

def _rolling_rate(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_driver_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["driverId", "year", "round"]).copy()

    grp = df.groupby("driverId")

    df["driver_win_rate"] = grp["win"].transform(
        lambda s: _rolling_rate(s, window)
    )

    df["driver_podium_rate"] = grp["podium"].transform(
        lambda s: _rolling_rate(s, window)
    )

    df["driver_avg_pos"] = grp["positionOrder"].transform(
        lambda s: _rolling_rate(s, window)
    )

    df["driver_avg_pts"] = grp["points"].transform(
        lambda s: _rolling_rate(s, window)
    )

    return df


def add_constructor_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["constructorId", "year", "round"]).copy()

    grp = df.groupby("constructorId")

    df["constructor_win_rate"] = grp["win"].transform(
        lambda s: _rolling_rate(s, window)
    )

    df["constructor_avg_pts"] = grp["points"].transform(
        lambda s: _rolling_rate(s, window)
    )

    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    df["grid_front_row"] = (df["grid"] <= 2).astype(int)
    df["grid_top5"] = (df["grid"] <= 5).astype(int)

    df["grid_norm"] = df.groupby("raceId")["grid"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    return df


def add_circuit_history(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df = df.sort_values(["driverId", "circuitId", "year"]).copy()

    grp = df.groupby(["driverId", "circuitId"])

    df["driver_circuit_wins"] = grp["win"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    df["driver_circuit_podiums"] = grp["podium"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    return df


def add_qualifying_features(df: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    qual_agg = qualifying[["raceId", "driverId", "best_qual_sec", "position"]].copy()

    pole = (
        qual_agg.groupby("raceId")["best_qual_sec"]
        .min()
        .rename("pole_time_sec")
        .reset_index()
    )

    qual_agg = qual_agg.merge(pole, on="raceId", how="left")

    qual_agg["qual_gap_percent"] = ((qual_agg["best_qual_sec"] - qual_agg["pole_time_sec"]) / qual_agg["pole_time_sec"]) * 100

    df = df.merge(
        qual_agg[["raceId", "driverId", "best_qual_sec", "qual_gap_percent"]],
        on=["raceId", "driverId"],
        how="left"
    )

    return df


def add_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    max_round = df.groupby("year")["round"].transform("max")
    df["season_progress"] = df["round"] / max_round
    return df


def build_features(window: int = 5) -> pd.DataFrame:
    tables = load_all()

    print("Cleaning data:")
    df = clean_results(tables["results"], tables["races"])
    qual = clean_qualifying(tables["qualifying"])

    print("Building features: ")
    df = add_driver_form(df, window=window)
    df = add_constructor_form(df, window=window)
    df = add_grid_features(df)
    df = add_circuit_history(df, window=window)
    df = add_qualifying_features(df, qual)
    df = add_season_progress(df)

    FEATURE_COLUMNS = [
        "grid",
        "grid_front_row",
        "grid_top5",
        "grid_norm",
        "driver_win_rate",
        "driver_podium_rate",
        "driver_avg_pos",
        "driver_avg_pts",
        "constructor_win_rate",
        "constructor_avg_pts",
        "driver_circuit_wins",
        "driver_circuit_podiums",
        "qual_gap_percent",
        "season_progress",
    ]

    df_model = df.dropna(subset=["podium"] + ["grid", "driver_avg_pos"]).copy()

    print(f"Dataset: {df_model.shape[0]} rows, {len(FEATURE_COLUMNS)} features")
    print(f"Podium rate: {df_model['podium'].mean():.2%}")

    return df_model, FEATURE_COLUMNS


if __name__ == "__main__":
    df, features = build_features()
    print("Preview:")
    print(df[features].describe().round(3))
