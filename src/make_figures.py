"""Render the M6 process and class distribution figures from repository data."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"


def configure_font() -> None:
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def flowchart() -> None:
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#f8fafc")
    ax.set(xlim=(0, 15), ylim=(0, 6))
    ax.axis("off")
    stages = [
        (0.2, "ISIC 影像与元数据", "精确标签筛选\n去重 / 按组划分"),
        (3.2, "单张图像输入", "长边 512\n七项质量量"),
        (6.2, "质量门控", "超出训练/验证阈值\n拒判并说明理由"),
        (9.2, "粗分割与特征", "光照校正 / a* Otsu\n20 项形态与颜色特征"),
        (12.2, "模型与单图报告", "标准化 + 逻辑回归\n风险分 / 贡献 / 建议"),
    ]
    for x, title, detail in stages:
        box = FancyBboxPatch((x, 2.1), 2.55, 2.0, boxstyle="round,pad=0.13",
                             facecolor="#e5f1f7", edgecolor="#28739a", linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + 1.275, 3.48, title, ha="center", va="center", fontsize=12, weight="bold", color="#17324d")
        ax.text(x + 1.275, 2.7, detail, ha="center", va="center", fontsize=10, linespacing=1.4)
    for x in (2.75, 5.75, 8.75, 11.75):
        ax.add_patch(FancyArrowPatch((x, 3.1), (x + 0.4, 3.1), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=1.6, color="#28739a"))
    ax.annotate("拒判 → 重新拍摄", xy=(7.475, 2.05), xytext=(7.475, 0.88),
                arrowprops={"arrowstyle": "<-", "color": "#b42332", "lw": 1.7},
                ha="center", fontsize=12, color="#b42332", weight="bold")
    ax.set_title("Q-Derm 端到端处理流程", fontsize=19, weight="bold", color="#17324d", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "flowchart.png", dpi=170)
    plt.close(fig)


def class_distribution() -> None:
    with (ROOT / "data" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter((row["split"], row["label"]) for row in rows)
    splits = ["train", "val", "test"]
    benign = [counts[split, "0"] for split in splits]
    melanoma = [counts[split, "1"] for split in splits]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f8fafc")
    bars_a = ax.bar(splits, benign, label="良性黑素细胞增生", color="#3283aa", width=0.55)
    bars_b = ax.bar(splits, melanoma, bottom=benign, label="黑色素瘤", color="#bd4b58", width=0.55)
    for group in (bars_a, bars_b):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    str(round(bar.get_height())), ha="center", va="center", color="white", weight="bold")
    for x, a, b in zip(splits, benign, melanoma):
        ax.text(x, a + b + 7, str(a + b), ha="center", weight="bold")
    ax.set(title="数据构成：按 split × label", ylabel="图像数量", ylim=(0, max(a + b for a, b in zip(benign, melanoma)) * 1.16))
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_font()
    flowchart()
    class_distribution()
    print(f"saved={FIGURES / 'flowchart.png'}")
    print(f"saved={FIGURES / 'class_distribution.png'}")
