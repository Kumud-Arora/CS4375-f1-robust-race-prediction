import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    average_precision_score, log_loss
)
from feature_engineering import build_features
from gradient_boosting import GradientBoostingClassifier


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
LOG_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", dpi=150)
    plt.close()
    print("saved:", name)


def prepare_data():
    df, feature_cols = build_features()

    train_df = df[df["year"] < 2020]
    test_df = df[df["year"] >= 2020]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["podium"].values

    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["podium"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("train:", X_train.shape, "test:", X_test.shape)

    return X_train, X_test, y_train, y_test, feature_cols


def train_scratch(X, y):
    print("training scratch model...")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X, y)
    return model


def train_sklearn(X, y):
    print("training sklearn model...")

    model = SklearnGB(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X, y)
    return model


def evaluate(model, X, y, name="model"):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = model.decision_function(X)

    pred = (prob >= 0.5).astype(int)

    metrics = {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
        "logloss": log_loss(y, prob)
    }

    print("\n", name)
    for k, v in metrics.items():
        print(k, round(v, 4))

    return metrics


def plot_roc(m1, m2, X, y):
    for model, label in [(m1, "scratch"), (m2, "sklearn")]:
        p = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC")
    save_fig("roc.png")


def plot_pr(m1, m2, X, y):
    for model, label in [(m1, "scratch"), (m2, "sklearn")]:
        p = model.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=label)

    plt.legend()
    plt.title("PR curve")
    save_fig("pr.png")


def confusion(model, X, y, name):
    pred = model.predict(X)
    cm = confusion_matrix(y, pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name)
    save_fig(f"cm_{name}.png")


def main():
    X_train, X_test, y_train, y_test, features = prepare_data()

    scratch = train_scratch(X_train, y_train)
    sklearn = train_sklearn(X_train, y_train)

    evaluate(scratch, X_test, y_test, "scratch")
    evaluate(sklearn, X_test, y_test, "sklearn")

    plot_roc(scratch, sklearn, X_test, y_test)
    plot_pr(scratch, sklearn, X_test, y_test)

    confusion(scratch, X_test, y_test, "scratch")
    confusion(sklearn, X_test, y_test, "sklearn")

    print("done")


if __name__ == "__main__":
    main()