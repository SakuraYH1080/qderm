"""Frozen M4/M5 degradation, quality gate, and held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

from vision import (FEATURE_COLUMNS, FEATURE_NAMES, QUALITY_NAMES, analyze_image,
                    load_image, resize_image, segment_lesion)


ROOT = Path(__file__).resolve().parents[1]
DEG_TYPES = ("gauss_blur", "motion_blur", "under_exposure", "over_exposure",
             "glare", "occlusion")
METRIC_COLUMNS = ("set", "level", "deg_type", "gate", "n_input", "n_accepted",
                  "coverage", "reject_rate", "auc", "pr_auc", "sens", "spec",
                  "tn", "fp", "fn", "tp")


def degrade(image: np.ndarray, kind: str, level: int, isic_id: str) -> np.ndarray:
    """Apply one deterministic degradation to a 512-long-side BGR image in memory."""
    height, width = image.shape[:2]
    short = min(height, width)
    if kind == "gauss_blur":
        sigma = (1.5, 3.0)[level - 1] * max(height, width) / 512
        return cv2.GaussianBlur(image, (0, 0), sigma)
    if kind == "motion_blur":
        length = max(3, round(short * (0.01, 0.03)[level - 1]))
        length += int(length % 2 == 0)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1 / length
        return cv2.filter2D(image, -1, kernel)
    if kind in ("under_exposure", "over_exposure"):
        factor = ((0.6, 0.35) if kind == "under_exposure" else (1.4, 2.0))[level - 1]
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    seed = int.from_bytes(hashlib.sha256(f"42:{isic_id}:{kind}:{level}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    result = image.copy()
    if kind == "glare":
        radius = max(1, round(short * (0.08, 0.15)[level - 1]))
        for _ in range((1, 3)[level - 1]):
            center = (int(rng.integers(radius, max(radius + 1, width - radius))),
                      int(rng.integers(radius, max(radius + 1, height - radius))))
            cv2.circle(result, center, radius, (255, 255, 255), -1)
        return result
    if kind == "occlusion":
        # Each dark rectangle touches an image border; total target area is 8% or 20%.
        count = (1, 2)[level - 1]
        area = (0.08, 0.20)[level - 1] * height * width / count
        for _ in range(count):
            side = int(rng.integers(4))
            if side < 2:
                block_w = max(1, round(width * 0.28))
                block_h = min(height, max(1, round(area / block_w)))
                x0 = 0 if side == 0 else width - block_w
                y0 = int(rng.integers(0, height - block_h + 1))
            else:
                block_h = max(1, round(height * 0.28))
                block_w = min(width, max(1, round(area / block_h)))
                y0 = 0 if side == 2 else height - block_h
                x0 = int(rng.integers(0, width - block_w + 1))
            result[y0:y0 + block_h, x0:x0 + block_w] = 0
        return result
    raise ValueError(f"Unknown degradation: {kind}")


def load_inputs() -> tuple[dict, dict, dict]:
    with (ROOT / "data" / "features.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FEATURE_COLUMNS:
            raise ValueError("features.csv schema mismatch")
        features = {row["isic_id"]: row for row in reader}
    with (ROOT / "data" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if len(features) != len(manifest):
        raise ValueError("Manifest and features length mismatch")
    splits = {name: [] for name in ("train", "val", "test")}
    for item in manifest:
        row = features[item["isic_id"]]
        if item["split"] != row["split"] or item["group_id"] != row["group_id"] or item["label"] != row["label"]:
            raise ValueError("Manifest/features mismatch")
        splits[item["split"]].append(item)
    groups = [{item["group_id"] for item in splits[name]} for name in splits]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Group leakage across splits")
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model feature order mismatch")
    with (ROOT / "artifacts" / "thresholds.json").open(encoding="utf-8") as handle:
        thresholds = json.load(handle)
    if "classification" not in thresholds:
        raise ValueError("Missing M3 classification threshold")
    return splits, features, {"model": model, "thresholds": thresholds}


def original_record(item: dict, features: dict) -> dict:
    row = features[item["isic_id"]]
    return {"id": item["isic_id"], "group": item["group_id"], "label": int(item["label"]),
            "features": [float(row[name]) for name in FEATURE_NAMES],
            "quality": [float(row[name]) for name in QUALITY_NAMES]}


def degraded_records(items: list[dict], kind: str, level: int) -> list[dict]:
    records = []
    for index, item in enumerate(items, 1):
        image = resize_image(load_image(ROOT / item["path"]))
        altered = degrade(image, kind, level, item["isic_id"])
        values, quality, _, _, _ = analyze_image(altered)
        records.append({"id": item["isic_id"], "group": item["group_id"],
                        "label": int(item["label"]),
                        "features": [values[name] for name in FEATURE_NAMES],
                        "quality": [quality[name] for name in QUALITY_NAMES]})
        if index % 40 == 0:
            print(f"{kind} level={level} processed={index}/{len(items)}", flush=True)
    return records


def gate_bounds(train: list[dict], factor: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array([row["quality"] for row in train])
    low, high = np.quantile(matrix, (0.01, 0.99), axis=0)
    center = np.median(matrix, axis=0)
    return center - factor * (center - low), center + factor * (high - center)


def accepts(records: list[dict], bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    matrix = np.array([row["quality"] for row in records])
    return np.all((matrix >= bounds[0]) & (matrix <= bounds[1]), axis=1)


def choose_gate(train: list[dict], val_original: list[dict], heavy: list[dict]) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    # The normal interval is estimated from train alone. Validation selects a
    # single expansion factor, never independent per-metric test-tuned bounds.
    candidates = np.round(np.arange(0.5, 3.001, 0.025), 3)
    feasible = []
    for factor in candidates:
        bounds = gate_bounds(train, float(factor))
        original_coverage = float(accepts(val_original, bounds).mean())
        if original_coverage >= 0.9:
            heavy_rejection = float(1 - accepts(heavy, bounds).mean())
            feasible.append((heavy_rejection, original_coverage, float(factor), bounds))
    if not feasible:
        raise ValueError("No quality gate candidate satisfies 90% val original coverage")
    # Maximize heavy rejection; ties favor original coverage, then wider bounds.
    rejection, coverage, factor, bounds = max(feasible, key=lambda row: row[:3])
    result = {
        "selection": "Train originals: per-metric 1st/99th percentiles around median; val chooses one interval scale maximizing pooled level-2 rejection subject to original coverage >= 0.90. Ties favor original coverage then wider intervals.",
        "candidate_scale_min": 0.5, "candidate_scale_max": 3.0, "candidate_scale_step": 0.025,
        "selected_scale": factor,
        "metrics": {name: {"lower": float(bounds[0][i]), "upper": float(bounds[1][i])}
                    for i, name in enumerate(QUALITY_NAMES)},
        "val_original_coverage": coverage,
        "val_original_reject_rate": 1 - coverage,
        "val_heavy_n_input": len(heavy),
        "val_heavy_coverage": 1 - rejection,
        "val_heavy_reject_rate": rejection,
    }
    return result, bounds


def score_records(records: list[dict], model: dict) -> np.ndarray:
    values = np.array([row["features"] for row in records], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite degraded features")
    return model["classifier"].predict_proba(model["scaler"].transform(values))[:, 1]


def metric_row(records: list[dict], scores: np.ndarray, accepted: np.ndarray,
               split: str, level: int, kind: str, gate: str, threshold: float) -> dict:
    selected = accepted if gate == "on" else np.ones(len(records), dtype=bool)
    labels = np.array([row["label"] for row in records], dtype=int)[selected]
    probabilities = scores[selected]
    if len(labels):
        tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if tp + fn else None
        spec = float(tn / (tn + fp)) if tn + fp else None
        auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
        pr_auc = float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    else:
        tn = fp = fn = tp = 0
        sens = spec = auc = pr_auc = None
    n = len(records)
    accepted_n = int(selected.sum())
    return dict(zip(METRIC_COLUMNS, (split, level, kind, gate, n, accepted_n,
              accepted_n / n, 1 - accepted_n / n, auc, pr_auc, sens, spec,
              int(tn), int(fp), int(fn), int(tp))))


def group_bootstrap_auc(records: list[dict], scores: np.ndarray) -> dict:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(row["group"], []).append(index)
    group_ids = list(groups)
    labels = np.array([row["label"] for row in records])
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(500):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        indices = np.array([index for group in sampled for index in groups[group]], dtype=int)
        if len(np.unique(labels[indices])) == 2:
            aucs.append(roc_auc_score(labels[indices], scores[indices]))
    if not aucs:
        raise ValueError("No valid two-class bootstrap resamples")
    low, high = np.quantile(aucs, (0.025, 0.975))
    return {"method": "group_id bootstrap with replacement", "seed": 42,
            "attempted": 500, "valid": len(aucs), "lower": float(low), "upper": float(high)}


def segmentation_stability(items: list[dict]) -> dict:
    values = []
    for index, item in enumerate(items, 1):
        image = resize_image(load_image(ROOT / item["path"]))
        original, _, _ = segment_lesion(image)
        variants = (np.clip(image.astype(np.float32) * 0.95, 0, 255).astype(np.uint8),
                    np.clip(image.astype(np.float32) * 1.05, 0, 255).astype(np.uint8),
                    cv2.GaussianBlur(image, (0, 0), 1.5))
        for variant in variants:
            altered, _, _ = segment_lesion(variant)
            union = np.count_nonzero((original > 0) | (altered > 0))
            values.append(np.count_nonzero((original > 0) & (altered > 0)) / union)
        if index % 40 == 0:
            print(f"segmentation_stability processed={index}/{len(items)}", flush=True)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "n_images": len(items), "n_comparisons": len(values),
            "perturbations": ["brightness -5%", "brightness +5%", "Gaussian blur sigma=1.5 at 512px"],
            "note": "seg_iou_stability is mask agreement under perturbation; it is not Dice and not segmentation accuracy."}


def save_figures(rows: list[dict], originals: list[dict], original_scores: np.ndarray,
                 original_accepted: np.ndarray, model: dict, threshold: float) -> None:
    figures = ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    labels = np.array([row["label"] for row in originals])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(labels, original_scores)
    precision, recall, _ = precision_recall_curve(labels, original_scores)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(labels, original_scores):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="Test original ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(labels, original_scores):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test original PR", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "roc_pr.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, gate, mask in zip(axes, ("off", "on"), (np.ones(len(labels), dtype=bool), original_accepted)):
        matrix = confusion_matrix(labels[mask], original_scores[mask] >= threshold, labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Melanoma"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Test originals, gate {gate} (n={mask.sum()})")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png")
    plt.close(fig)

    coefficients = model["classifier"].coef_[0]
    order = np.argsort(coefficients)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(FEATURE_NAMES)[order], coefficients[order],
            color=["#bd4c50" if value > 0 else "#3a78a1" for value in coefficients[order]])
    ax.set(xlabel="Standardized logistic coefficient", title="Model feature contribution")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        for level, style in ((1, "o-"), (2, "s--")):
            series = [next(row["auc"] for row in rows if row["set"] == split and row["deg_type"] == kind
                           and row["level"] == level and row["gate"] == "off") for kind in DEG_TYPES]
            ax.plot(range(len(DEG_TYPES)), series, style, label=f"level {level}")
        ax.axhline(next(row["auc"] for row in rows if row["set"] == split and row["deg_type"] == "none"
                        and row["gate"] == "off"), color="gray", linestyle=":", label="original")
        ax.set_xticks(range(len(DEG_TYPES)), DEG_TYPES, rotation=45, ha="right")
        ax.set(xlabel="Degradation type", title=split, ylim=(0, 1))
        ax.legend()
    axes[0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(figures / "degradation_perf.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        subset = [row for row in rows if row["set"] == split and row["auc"] is not None]
        for gate, marker in (("off", "o"), ("on", "x")):
            chosen = [row for row in subset if row["gate"] == gate]
            ax.scatter([row["coverage"] for row in chosen], [row["auc"] for row in chosen],
                       marker=marker, alpha=0.75, label=f"gate {gate}")
        ax.set(xlabel="Coverage", title=split, xlim=(0, 1.04), ylim=(0, 1))
        ax.legend()
    axes[0].set_ylabel("AUC on accepted samples")
    fig.tight_layout()
    fig.savefig(figures / "coverage_perf.png")
    plt.close(fig)


def main() -> None:
    splits, features, inputs = load_inputs()
    model, thresholds = inputs["model"], inputs["thresholds"]
    class_threshold = float(thresholds["classification"]["threshold"])
    train = [original_record(item, features) for item in splits["train"]]
    all_records = {}
    for split in ("val", "test"):
        all_records[(split, "none", 0)] = [original_record(item, features) for item in splits[split]]
    # Generate validation degradations first. The gate must be frozen before
    # any test degradation is computed or observed.
    for kind in DEG_TYPES:
        for level in (1, 2):
            all_records[("val", kind, level)] = degraded_records(splits["val"], kind, level)
    heavy_val = [row for kind in DEG_TYPES for row in all_records[("val", kind, 2)]]
    gate_info, bounds = choose_gate(train, all_records[("val", "none", 0)], heavy_val)
    thresholds["quality_gate"] = gate_info
    with (ROOT / "artifacts" / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"gate_selected scale={gate_info['selected_scale']} val_original_coverage={gate_info['val_original_coverage']:.4f} val_heavy_reject_rate={gate_info['val_heavy_reject_rate']:.4f}", flush=True)
    for kind in DEG_TYPES:
        for level in (1, 2):
            all_records[("test", kind, level)] = degraded_records(splits["test"], kind, level)

    rows = []
    score_cache = {}
    for (split, kind, level), records in all_records.items():
        scores = score_records(records, model)
        score_cache[(split, kind, level)] = scores
        accepted = accepts(records, bounds)
        for gate in ("off", "on"):
            rows.append(metric_row(records, scores, accepted, split, level, kind, gate, class_threshold))
    with (ROOT / "outputs" / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    original = all_records[("test", "none", 0)]
    original_scores = score_cache[("test", "none", 0)]
    original_accepted = accepts(original, bounds)
    heavy_test = [row for kind in DEG_TYPES for row in all_records[("test", kind, 2)]]
    heavy_scores = np.concatenate([score_cache[("test", kind, 2)] for kind in DEG_TYPES])
    heavy_accepted = accepts(heavy_test, bounds)
    summary = {
        "test_original_gate_off": next(row for row in rows if row["set"] == "test" and row["deg_type"] == "none" and row["gate"] == "off"),
        "test_original_gate_on": next(row for row in rows if row["set"] == "test" and row["deg_type"] == "none" and row["gate"] == "on"),
        "test_original_auc_95ci": group_bootstrap_auc(original, original_scores),
        "test_heavy_pooled_gate_off": metric_row(heavy_test, heavy_scores, heavy_accepted, "test", 2, "pooled_heavy_summary_only", "off", class_threshold),
        "test_heavy_pooled_gate_on": metric_row(heavy_test, heavy_scores, heavy_accepted, "test", 2, "pooled_heavy_summary_only", "on", class_threshold),
        "seg_iou_stability": segmentation_stability(splits["test"]),
    }
    with (ROOT / "outputs" / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    save_figures(rows, original, original_scores, original_accepted, model, class_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
