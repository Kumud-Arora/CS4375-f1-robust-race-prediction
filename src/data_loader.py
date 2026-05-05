import pandas as pd
import numpy as np
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{filename}' not found in {DATA_DIR}. Download from Kaggle and put files in the data folder."
        )
    df = pd.read_csv(path)
    print(f"  Loaded {filename:40s} → {df.shape[0]:>6,} rows × {df.shape[1]} cols")
    return df

def load_all() -> dict[str, pd.DataFrame]:
    print("Loading data: ")
    tables = {
        "results":               load_csv("Race_Results.csv"),
        "races":                 load_csv("Race_Schedule.csv"),
        "drivers":               load_csv("Driver_Details.csv"),
        "constructors":          load_csv("Team_Details.csv"),
        "qualifying":            load_csv("Qualifying_Results.csv"),
        "driver_standings":      load_csv("Driver_Rankings.csv"),
        "constructor_standings": load_csv("Constructor_Rankings.csv"),
    }
    print(f"{len(tables)} tables loaded")
    return tables


def clean_results(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name"]], on="raceId", how="left"
    )

    for col in df.columns:
        df[col] = df[col].replace("\\N", np.nan)

    numeric_cols = [
        "positionOrder", "points", "laps", "milliseconds",
        "fastestLap", "fastestLapSpeed", "grid", "year", "round"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["podium"] = (df["positionOrder"] <= 3).astype(int)
    df["win"] = (df["positionOrder"] == 1).astype(int)

    print(f"  Results after cleaning: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def clean_qualifying(qualifying: pd.DataFrame) -> pd.DataFrame:
    df = qualifying.copy()
    df = df.replace("\\N", np.nan)

    def lap_to_seconds(t):
        if pd.isna(t):
            return np.nan
        try:
            parts = str(t).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(t)
        except:
            return np.nan

    for q in ["q1", "q2", "q3"]:
        if q in df.columns:
            df[f"{q}_sec"] = df[q].apply(lap_to_seconds)

    q_sec_cols = [c for c in ["q1_sec", "q2_sec", "q3_sec"] if c in df.columns]
    if q_sec_cols:
        df["best_qual_sec"] = df[q_sec_cols].min(axis=1)

    return df


if __name__ == "__main__":
    tables = load_all()
    results_clean = clean_results(tables["results"], tables["races"])
    qual_clean = clean_qualifying(tables["qualifying"])

    print("Sample:")
    print(results_clean[[
        "raceId", "driverId", "year", "grid",
        "positionOrder", "podium", "win"
    ]].head(10))