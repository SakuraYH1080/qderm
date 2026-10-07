"""Create a one-page visual screening report for a single dermoscopy image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import degrade
from vision import FEATURE_NAMES, QUALITY_NAMES, analyze_image, load_image, resize_image


ROOT = Path(__file__).resolve().parents[1]
QUALITY_LABELS = {
    "blur_lapvar": "清晰度/Laplacian方差",
    "exposure_mean": "亮度均值",
    "exposure_clip_hi_ratio": "高亮截断占比",
    "exposure_clip_lo_ratio": "低亮截断占比",
    "contrast_std": "对比度标准差",
    "glare_ratio": "眩光占比",
    "occlusion_border_ratio": "边框暗遮挡占比",
}


def configure_font() -> None:
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def predict(image: np.ndarray, model: dict, thresholds: dict) -> dict:
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError("模型特征顺序与当前流水线不一致")
    features, quality, mask, fallback, raw_fraction = analyze_image(image)
    values = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("特征含非有限值")
    scaled = model["scaler"].transform(values)
    standardized = scaled[0]
    coefficients = model["classifier"].coef_[0]
    contributions = coefficients * standardized
    score = float(model["classifier"].predict_proba(scaled)[0, 1])
    gate = thresholds["quality_gate"]["metrics"]
    violations = []
    for name in QUALITY_NAMES:
        value = quality[name]
        lower, upper = float(gate[name]["lower"]), float(gate[name]["upper"])
        if value < lower or value > upper:
            violations.append((name, value, lower, upper))
    return {
        "quality": quality, "mask": mask, "fallback": fallback,
        "raw_fraction": raw_fraction, "score": score,
        "contributions": contributions, "violations": violations,
        "accepted": not violations,
        "threshold": float(thresholds["classification"]["threshold"]),
    }


def draw_report(image: np.ndarray, result: dict, image_name: str, output: Path,
                degradation: str | None) -> None:
    configure_font()
    image = resize_image(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 230, 40), 3)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 10), facecolor="#f8fafc")
    grid = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.25, 1.7],
                           height_ratios=[1.05, 1.05, 0.78],
                           left=0.045, right=0.975, top=0.89, bottom=0.09,
                           wspace=0.18, hspace=0.29)
    fig.suptitle("Q-Derm  单图质控与筛查报告", x=0.045, y=0.965,
                 ha="left", fontsize=22, fontweight="bold", color="#17324d")
    subtitle = image_name + (f"  |  内存合成退化：{degradation}" if degradation else "  |  原图")
    fig.text(0.045, 0.922, subtitle, fontsize=11, color="#526477")

    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(rgb)
    ax_image.set_title("原图 / 本次输入", loc="left", fontweight="bold")
    ax_image.axis("off")

    ax_overlay = fig.add_subplot(grid[1, 0])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("病灶粗分割轮廓叠加", loc="left", fontweight="bold")
    ax_overlay.axis("off")

    ax_quality = fig.add_subplot(grid[:2, 1])
    ax_quality.axis("off")
    status = "质量合格 / 可输出筛查分" if result["accepted"] else "质量拒判 / 请重新拍摄"
    color = "#16734b" if result["accepted"] else "#b42332"
    ax_quality.text(0, 0.98, "质量判定", fontsize=15, weight="bold", va="top", transform=ax_quality.transAxes)
    ax_quality.text(0, 0.91, status, fontsize=13, weight="bold", color=color,
                    va="top", transform=ax_quality.transAxes)
    quality_lines = []
    for name in QUALITY_NAMES:
        value = result["quality"][name]
        marker = "●" if any(item[0] == name for item in result["violations"]) else "○"
        quality_lines.append(f"{marker} {QUALITY_LABELS[name]}：{value:.4f}")
    ax_quality.text(0, 0.82, "\n".join(quality_lines), fontsize=10.2, va="top",
                    linespacing=1.8, transform=ax_quality.transAxes)
    if result["violations"]:
        reasons = [f"{QUALITY_LABELS[name]} {value:.4f}，正常区间 [{lo:.4f}, {hi:.4f}]"
                   for name, value, lo, hi in result["violations"]]
        ax_quality.text(0, 0.37, "触发阈值：\n" + "\n".join(reasons), fontsize=8.5,
                        color="#9f2431", va="top", linespacing=1.25, transform=ax_quality.transAxes)
    else:
        ax_quality.text(0, 0.35, "触发阈值：无；七项质量量均在训练/验证冻结区间内。",
                        fontsize=9.6, va="top", transform=ax_quality.transAxes)
    seg_note = "分割回退：中心 ROI（不可视作真实病灶边界）" if result["fallback"] else "分割：最大连通域轮廓"
    ax_quality.text(0, 0.08, seg_note, fontsize=9.5, color="#526477", transform=ax_quality.transAxes)

    ax_bar = fig.add_subplot(grid[:2, 2])
    order = np.argsort(np.abs(result["contributions"]))[::-1]
    names = np.array(FEATURE_NAMES)[order]
    values = result["contributions"][order]
    positions = np.arange(len(names))
    ax_bar.barh(positions, values, color=np.where(values >= 0, "#bd4b58", "#3283aa"))
    ax_bar.set_yticks(positions, names, fontsize=8.6)
    ax_bar.invert_yaxis()
    ax_bar.axvline(0, color="#64748b", linewidth=0.8)
    ax_bar.set_title("20 特征对本图的贡献（按绝对值排序）", loc="left", fontweight="bold")
    ax_bar.set_xlabel("系数 × 标准化特征值；红色推高风险，蓝色降低风险")
    ax_bar.grid(axis="x", alpha=0.2)

    ax_risk = fig.add_subplot(grid[2, :])
    ax_risk.axis("off")
    if result["accepted"]:
        high = result["score"] >= result["threshold"]
        grade = "高于筛查阈值：建议由皮肤科专业人员进一步评估" if high else "低于筛查阈值：如有疑虑仍请咨询皮肤科专业人员"
        headline = f"风险分 {result['score']:.3f}  |  分类阈值 {result['threshold']:.3f}  |  {grade}"
    else:
        headline = "风险分与分级建议：未输出（质量门控拒判）；建议重新拍摄后再分析"
    ax_risk.text(0, 0.82, headline, fontsize=13.5, weight="bold", color=color,
                 va="top", transform=ax_risk.transAxes)
    ax_risk.text(0, 0.40, "贡献条形图为内部模型计算的 log-odds 分量；拒判图仅供失效分析，不作筛查解释。",
                 fontsize=10, color="#526477", transform=ax_risk.transAxes)
    ax_risk.text(0, 0.08, "本原型不构成临床诊断", fontsize=15, weight="bold",
                 color="#9f2431", transform=ax_risk.transAxes)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="单张图像路径")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "figures" / "single_image_report.png")
    parser.add_argument("--degrade", choices=["gauss_blur", "motion_blur", "under_exposure",
                                             "over_exposure", "glare", "occlusion"],
                        help="在内存中生成退化演示；不保存退化原图")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    with (ROOT / "artifacts" / "thresholds.json").open(encoding="utf-8") as handle:
        thresholds = json.load(handle)
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    image = resize_image(load_image(args.image))
    if args.degrade:
        image = degrade(image, args.degrade, args.level, args.image.stem)
    result = predict(image, model, thresholds)
    draw_report(image, result, args.image.name, args.output,
                f"{args.degrade} level={args.level}" if args.degrade else None)
    print(f"saved={args.output} accepted={result['accepted']} score={'%.6f' % result['score'] if result['accepted'] else 'withheld'}")
    for name, value, lower, upper in result["violations"]:
        print(f"rejected_by={name} value={value:.6f} bounds=[{lower:.6f},{upper:.6f}]")


if __name__ == "__main__":
    main()
