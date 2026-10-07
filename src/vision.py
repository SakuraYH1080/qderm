"""Shared image quality, coarse segmentation, and 20 interpretable features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    "asym_major", "asym_minor", "circularity", "solidity", "eccentricity",
    "perimeter_norm", "bbox_aspect", "area_ratio", "border_roughness", "radial_cv",
    "lab_mean_L", "lab_mean_a", "lab_mean_b", "lab_std_L", "lab_std_a",
    "lab_std_b", "color_count", "dark_ratio", "center_edge_diff", "color_entropy",
]
QUALITY_NAMES = [
    "blur_lapvar", "exposure_mean", "exposure_clip_hi_ratio",
    "exposure_clip_lo_ratio", "contrast_std", "glare_ratio",
    "occlusion_border_ratio",
]
FEATURE_COLUMNS = ["isic_id", "split", "group_id", "label"] + FEATURE_NAMES + QUALITY_NAMES + ["seg_fallback"]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def resize_image(image: np.ndarray, long_side: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    if scale == 1:
        return image.copy()
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    image = resize_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light = lab[:, :, 0]
    dark = np.uint8(light <= 35)
    count, components = cv2.connectedComponents(dark, connectivity=8)
    border_labels = np.unique(np.concatenate((components[0], components[-1],
                                              components[:, 0], components[:, -1])))
    border_dark = np.isin(components, border_labels[border_labels != 0]) if count > 1 else np.zeros_like(dark, dtype=bool)
    return {
        "blur_lapvar": float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
        "exposure_mean": float(light.mean()),
        "exposure_clip_hi_ratio": float(np.mean(light >= 250)),
        "exposure_clip_lo_ratio": float(np.mean(light <= 20)),
        "contrast_std": float(light.std()),
        "glare_ratio": float(np.mean((light >= 240) & (hsv[:, :, 1] <= 40))),
        "occlusion_border_ratio": float(border_dark.mean()),
    }


def correct_illumination(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    kernel_size = max(3, round(min(light.shape) / 8))
    kernel_size += int(kernel_size % 2 == 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel)
    corrected = np.clip(light.astype(np.float32) * 128 / np.maximum(background, 1), 0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(corrected)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def segment_lesion(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Return mask, fallback flag, and raw largest-component area fraction."""
    image = resize_image(image)
    corrected = correct_illumination(image)
    a_channel = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)[:, :, 1]
    _, mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(components == largest) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        mask = filled
    else:
        mask = np.zeros_like(mask)
    raw_fraction = float(np.mean(mask > 0))
    fallback = int(raw_fraction < 0.02 or raw_fraction > 0.95)
    if fallback:
        height, width = mask.shape
        mask[:] = 0
        mask[round(height * 0.2):round(height * 0.8),
             round(width * 0.2):round(width * 0.8)] = 255
    return mask, fallback, raw_fraction


def _mirror_asymmetry(mask: np.ndarray, center: np.ndarray, direction: np.ndarray) -> float:
    # Reflection in a line through center with unit direction.
    dx, dy = direction
    reflection = np.array([[2 * dx * dx - 1, 2 * dx * dy],
                           [2 * dx * dy, 2 * dy * dy - 1]], dtype=np.float32)
    shift = center - reflection @ center
    transform = np.column_stack((reflection, shift)).astype(np.float32)
    mirrored = cv2.warpAffine(mask, transform, (mask.shape[1], mask.shape[0]),
                              flags=cv2.INTER_NEAREST, borderValue=0) > 0
    original = mask > 0
    union = np.count_nonzero(original | mirrored)
    return 1 - np.count_nonzero(original & mirrored) / union if union else 0.0


def extract_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    image = resize_image(image)
    if image.shape[:2] != mask.shape or not np.any(mask):
        raise ValueError("Mask must be nonempty and match resized image")
    binary = np.uint8(mask > 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    y, x = np.nonzero(binary)
    area = len(x)
    x0, y0, width, height = cv2.boundingRect(contour)
    diagonal = float(np.hypot(width, height))
    coords = np.column_stack((x.astype(float), y.astype(float)))
    center = coords.mean(axis=0)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, 1]
    minor = eigenvectors[:, 0]
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    boundary = contour[:, 0, :].astype(float)
    radii = np.linalg.norm(boundary - center, axis=1) / diagonal
    lab_pixels = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[binary > 0].astype(np.float32)
    means = lab_pixels.mean(axis=0)
    stds = lab_pixels.std(axis=0)
    ab = lab_pixels[:, 1:3]
    # Deterministic uniform subsample bounds the cost of k-means on large lesions.
    sample = ab[np.linspace(0, len(ab) - 1, min(len(ab), 4096), dtype=int)]
    cv2.setRNGSeed(42)
    _, assignments, _ = cv2.kmeans(sample, 3, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
                                   1, cv2.KMEANS_PP_CENTERS)
    cluster_shares = np.bincount(assignments.ravel(), minlength=3) / len(assignments)
    lesion_radius = np.sqrt(area / np.pi)
    distances = np.linalg.norm(coords - center, axis=1)
    inner = distances <= 0.5 * lesion_radius
    if not np.any(inner) or np.all(inner):
        inner = distances <= np.median(distances)
    delta_e = np.linalg.norm(lab_pixels[inner].mean(axis=0) - lab_pixels[~inner].mean(axis=0))
    histogram, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=16, range=[[0, 256], [0, 256]])
    probabilities = histogram.ravel() / len(ab)
    probabilities = probabilities[probabilities > 0]
    return {
        "asym_major": _mirror_asymmetry(binary, center, major),
        "asym_minor": _mirror_asymmetry(binary, center, minor),
        "circularity": float(4 * np.pi * area / perimeter**2) if perimeter else 0.0,
        "solidity": float(area / hull_area) if hull_area else 0.0,
        "eccentricity": float(np.sqrt(max(0, 1 - eigenvalues[0] / eigenvalues[1]))) if eigenvalues[1] else 0.0,
        "perimeter_norm": float(perimeter / diagonal),
        "bbox_aspect": float(min(width, height) / max(width, height)),
        "area_ratio": float(area / binary.size),
        "border_roughness": float(perimeter / hull_perimeter) if hull_perimeter else 0.0,
        "radial_cv": float(radii.std() / radii.mean()) if radii.mean() else 0.0,
        "lab_mean_L": float(means[0]), "lab_mean_a": float(means[1]), "lab_mean_b": float(means[2]),
        "lab_std_L": float(stds[0]), "lab_std_a": float(stds[1]), "lab_std_b": float(stds[2]),
        "color_count": float(np.count_nonzero(cluster_shares > 0.05)),
        "dark_ratio": float(np.mean(lab_pixels[:, 0] < np.percentile(lab_pixels[:, 0], 20))),
        "center_edge_diff": float(delta_e),
        "color_entropy": float(-(probabilities * np.log(probabilities)).sum() / np.log(256)),
    }


def analyze_image(image: np.ndarray) -> tuple[dict[str, float], dict[str, float], np.ndarray, int, float]:
    resized = resize_image(image)
    quality = quality_metrics(resized)
    mask, fallback, raw_fraction = segment_lesion(resized)
    features = extract_features(resized, mask)
    return features, quality, mask, fallback, raw_fraction


def save_overlay(image: np.ndarray, mask: np.ndarray, destination: Path, fallback: int, raw_fraction: float) -> None:
    image = resize_image(image)
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :, 2] = 255
    overlay[mask > 0] = cv2.addWeighted(image, 0.6, tint, 0.4, 0)[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.putText(overlay, f"fallback={fallback} raw_area={raw_fraction:.3f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(overlay, f"fallback={fallback} raw_area={raw_fraction:.3f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Could not write overlay: {destination}")


def build_features(manifest_path: Path = ROOT / "data" / "manifest.csv") -> None:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    rows = []
    overlay_candidates = {0: [], 1: []}
    raw_areas = []
    for index, entry in enumerate(manifest, 1):
        image = load_image(ROOT / entry["path"])
        features, quality, mask, fallback, raw_fraction = analyze_image(image)
        row = {"isic_id": entry["isic_id"], "split": entry["split"],
               "group_id": entry["group_id"], "label": int(entry["label"]),
               **features, **quality, "seg_fallback": fallback}
        assert list(row) == FEATURE_COLUMNS
        rows.append(row)
        raw_areas.append(raw_fraction)
        # Keep the first 10 examples of each result without keeping image arrays in memory.
        if len(overlay_candidates[fallback]) < 10:
            overlay_candidates[fallback].append((entry["isic_id"], entry["path"], fallback, raw_fraction))
        if index % 100 == 0:
            print(f"processed={index}/{len(manifest)}", flush=True)
    assert len(FEATURE_NAMES) == 20
    assert all(set(row) == set(FEATURE_COLUMNS) for row in rows)
    numeric = np.array([[row[name] for name in FEATURE_NAMES + QUALITY_NAMES + ["seg_fallback"]]
                        for row in rows], dtype=float)
    assert np.isfinite(numeric).all(), "Features contain NaN or infinity"
    assert len(rows) == len(manifest)
    output = ROOT / "data" / "features.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    overlay_dir = ROOT / "outputs" / "figures" / "seg_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    selected = overlay_candidates[0] + overlay_candidates[1]
    if len(selected) < 20:
        seen = {candidate[0] for candidate in selected}
        for entry, row, raw_fraction in zip(manifest, rows, raw_areas):
            if entry["isic_id"] not in seen:
                selected.append((entry["isic_id"], entry["path"], row["seg_fallback"], raw_fraction))
                seen.add(entry["isic_id"])
            if len(selected) >= 20:
                break
    for isic_id, path, fallback, raw_fraction in selected:
        image = load_image(ROOT / path)
        mask, _, _ = segment_lesion(image)
        save_overlay(image, mask, overlay_dir / f"{isic_id}_fallback{fallback}.png", fallback, raw_fraction)
    areas = np.array(raw_areas)
    print(f"seg_fallback={int(numeric[:, -1].sum())}/{len(rows)} ({numeric[:, -1].mean():.4%})")
    print("raw_mask_area_fraction_quantiles(0,5,25,50,75,95,100)%=" +
          ", ".join(f"{value:.4f}" for value in np.percentile(areas, [0, 5, 25, 50, 75, 95, 100])))
    print(f"overlays={len(selected)} features={len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    args = parser.parse_args()
    build_features(args.manifest)
