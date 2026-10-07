"""Fit the frozen logistic baseline using train and val only."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from vision import FEATURE_COLUMNS, FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def read_split_features() -> dict[str, list[dict]]:
    result = {"train": [], "val": []}
    with (ROOT / "data" / "features.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FEATURE_COLUMNS:
            raise ValueError("features.csv columns do not match the frozen schema")
        for row in reader:
            if row["split"] in result:
                result[row["split"]].append(row)
    if not result["train"] or not result["val"]:
        raise ValueError("Both train and val must contain samples")
    train_groups = {row["group_id"] for row in result["train"]}
    val_groups = {row["group_id"] for row in result["val"]}
    if train_groups & val_groups:
        raise ValueError("Group leakage between train and val")
    return result


def split_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    if not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError("Nonfinite feature or missing class")
    return x, y


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float(np.mean(predicted[y_true == 1]))
        specificity = float(np.mean(~predicted[y_true == 0]))
        if sensitivity >= 0.75:
            candidate = (specificity, float(threshold), sensitivity)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError("No validation threshold reaches sensitivity 0.75")
    specificity, threshold, sensitivity = best
    return threshold, sensitivity, specificity


def main() -> None:
    rows = read_split_features()
    x_train, y_train = split_arrays(rows["train"])
    x_val, y_val = split_arrays(rows["val"])
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    classifier.fit(x_train_scaled, y_train)
    scores = classifier.predict_proba(x_val_scaled)[:, 1]
    threshold, sensitivity, specificity = choose_threshold(y_val, scores)
    coefficients = sorted(zip(FEATURE_NAMES, classifier.coef_[0]),
                          key=lambda pair: abs(pair[1]), reverse=True)
    importance = permutation_importance(classifier, x_val_scaled, y_val, n_repeats=10,
                                        scoring="roc_auc", random_state=42)
    permutations = sorted(zip(FEATURE_NAMES, importance.importances_mean,
                              importance.importances_std), key=lambda item: item[1], reverse=True)
    train_counts = dict(sorted(Counter(y_train.tolist()).items()))
    val_counts = dict(sorted(Counter(y_val.tolist()).items()))
    metrics = {
        "train": {"n": len(y_train), "class_counts": train_counts},
        "val": {"n": len(y_val), "class_counts": val_counts,
                "auc": float(roc_auc_score(y_val, scores)),
                "pr_auc": float(average_precision_score(y_val, scores)),
                "sensitivity": sensitivity, "specificity": specificity},
        "classification_threshold": threshold,
        "top_10_coefficients": [{"feature": name, "coefficient": float(value)}
                                for name, value in coefficients[:10]],
        "permutation_importance": [{"feature": name, "mean": float(mean), "std": float(std)}
                                   for name, mean, std in permutations],
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "classifier": classifier,
                 "feature_names": FEATURE_NAMES.copy()}, artifacts / "model.joblib")
    with (artifacts / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump({"classification": {"threshold": threshold,
                                      "selection": "Maximum val specificity subject to sensitivity >= 0.75",
                                      "val_sensitivity": sensitivity,
                                      "val_specificity": specificity}},
                  handle, ensure_ascii=False, indent=2)
    with (ROOT / "outputs" / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
