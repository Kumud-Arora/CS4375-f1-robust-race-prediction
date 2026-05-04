import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("saved", name)


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
    print("training scratch model")

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
    print("training sklearn model")

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
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, prob),
        "log_loss": log_loss(y, prob)
    }

    print("\n", name)
    for k, v in metrics.items():
        print(k, round(v, 4))

    return metrics


def plot_roc(m1, m2, X, y):
    fig, ax = plt.subplots()

    for model, label in [(m1, "scratch"), (m2, "sklearn")]:
        p = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    save_fig("roc.png")


def plot_pr(m1, m2, X, y):
    for model, label in [(m1, "scratch"), (m2, "sklearn")]:
        p = model.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    save_fig("pr.png")


def plot_confusion(model, X, y, name):
    pred = model.predict(X)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Podium", "Podium"],
                yticklabels=["No Podium", "Podium"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    save_fig(f"cm_{name}.png")


def plot_learning_curve(model, X_train, y_train, X_test, y_test):
    train_loss = model.staged_loss(X_train, y_train)
    test_loss  = model.staged_loss(X_test, y_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="train loss")
    ax.plot(test_loss,  label="test loss")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Log-loss")
    ax.set_title("Learning Curve")
    ax.legend()
    save_fig("learning_curve.png")
 
 
def plot_feature_importance(model, feature_cols):
    counts = np.zeros(len(feature_cols))
 
    def count_splits(node):
        if node is None or node.value is not None:
            return
        counts[node.feature_idx] += 1
        count_splits(node.left)
        count_splits(node.right)
 
    for tree in model._trees:
        count_splits(tree.root)
 
    importance = counts / (counts.sum() + 1e-9)
    order = np.argsort(importance)
 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_cols[i] for i in order], importance[order])
    ax.set_xlabel("Relative importance (split frequency)")
    ax.set_title("Feature Importance — Scratch GB")
    plt.tight_layout()
    save_fig("feature_importance.png")
 
 
def adversarial_robustness(model, X_test, y_test):
    """Test model stability by adding Gaussian noise to inputs."""
    print("\nRunning adversarial robustness test")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    aucs = []
    rng = np.random.default_rng(0)
    for sigma in noise_levels:
        X_noisy = X_test + rng.normal(0, sigma, size=X_test.shape)
        auc = roc_auc_score(y_test, model.predict_proba(X_noisy)[:, 1])
        aucs.append(auc)
        print("noise:", sigma, "auc:", round(auc, 4))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise_levels, aucs, marker="o")
    ax.axhline(aucs[0], linestyle="--", color="grey", label=f"clean AUC={aucs[0]:.3f}")
    ax.set_xlabel("Noise")
    ax.set_ylabel("AUC")
    ax.set_title("Adversarial Robustness")
    ax.legend()
    save_fig("adversarial_robustness.png")
    return dict(zip(noise_levels, aucs))
 
 
def decision_simulation(model, feature_cols):
    # simple simulation: pick best driver each race
    print("\nRunning decision simulation")
    df, _ = build_features()
    test_df = df[df["year"] >= 2020].copy()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Refit scaler on train portion
    train_df = df[df["year"] < 2020]
    scaler.fit(train_df[feature_cols].fillna(0).values)
    X = scaler.transform(test_df[feature_cols].fillna(0).values)
    test_df["pred_prob"] = model.predict_proba(X)[:, 1]
 
    results = []
    for race_id, grp in test_df.groupby("raceId"):
        best_idx = grp["pred_prob"].idxmax()
        results.append(int(grp.loc[best_idx, "podium"]))
 
    hit_rate = np.mean(results)
    baseline = test_df["podium"].mean()
    print(f"  Races: {len(results)}")
    print(f"  Model top-pick podium rate: {hit_rate:.2%}")
    print(f"  Random baseline:            {baseline:.2%}")
    print(f"  Lift: {hit_rate / baseline:.2f}x")
    return hit_rate

def main():
    X_train, X_test, y_train, y_test, features = prepare_data()
 
    # Experiment 1 — Scratch GB
    scratch = train_scratch(X_train, y_train)
    m1 = evaluate(scratch, X_test, y_test, "scratch_gb")
    log_experiment(1, {"algorithm": "scratch_gb", "n_estimators": 150, "lr": 0.05, "max_depth": 3, "subsample": 0.8}, m1)
 
    # Experiment 2 — Sklearn baseline
    sklearn_model = train_sklearn(X_train, y_train)
    m2 = evaluate(sklearn_model, X_test, y_test, "sklearn_gb")
    log_experiment(2, {"algorithm": "sklearn_gb", "n_estimators": 150, "lr": 0.05, "max_depth": 3, "subsample": 0.8}, m2)
 
    # Plots
    plot_roc(scratch, sklearn_model, X_test, y_test)
    plot_pr(scratch, sklearn_model, X_test, y_test)
    plot_confusion(scratch, X_test, y_test, "scratch")
    plot_confusion(sklearn_model, X_test, y_test, "sklearn")
    plot_learning_curve(scratch, X_train, y_train, X_test, y_test)
    plot_feature_importance(scratch, features)
 
    # Adversarial robustness
    adversarial_robustness(scratch, X_test, y_test)
 
    # Decision simulation
    decision_simulation(scratch, features)
 
    print("Done, check outputs folder for plots and experiment_log.csv")


if __name__ == "__main__":
    main()