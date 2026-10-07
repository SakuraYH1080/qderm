# TASKS.md — Q-Derm 里程碑看板
> 规则：每完成一个里程碑，把 `[ ]` 改为 `[x]`，并在 `outputs/session_log.md` 追加记录，然后 git commit。开工前先读 `PLAN.md`。

- [x] **M0（15min）方案冻结与骨架**
  产出：目录骨架、`requirements.txt`（仅记录已装版本）、`README.md` 初稿、`PLAN.md` 已在仓库内。
  DoD：`& "D:\anaconda\python.exe" -c "import cv2,sklearn,skimage,matplotlib,pandas;print(cv2.__version__)"` 输出 5.0.0。
- [x] **M1（50min）样本清单与下载**
  产出：`src/collect.py`、`data/manifest.csv`、`data/api_meta_snapshot.jsonl`、`data/images/`。
  DoD：两类各 400 张（或按 PLAN 降级并记录）；划分以 group_id 为单位且 60/20/20；`data/` ≤900MiB；下载失败有重试记录；控制台打印最终配额与容量。
- [x] **M2（40min）质控与粗分割**
  产出：`src/vision.py` 的质控与分割函数、`outputs/figures/seg_overlays/` ≥20 张。
  DoD：7 个质量量对全量图像可计算；`seg_fallback` 比例被打印；叠加图含成功与失败案例。
- [x] **M3（55min）特征与模型**
  产出：`src/train.py`、`data/features.csv`（20 特征无缺列无 NaN）、`artifacts/model.joblib`、`artifacts/thresholds.json`。
  DoD：val 上可复现出指标；阈值仅由 train/val 决定。
- [x] **M4（45min）退化合成与门控**
  产出：`src/evaluate.py` 的退化与门控评估、`outputs/figures/degradation_perf.png`、`coverage_perf.png`。
  DoD：12 种退化 × {val,test} 全部跑通；门控工作点满足 val 原图接受率 ≥90%。
- [x] **M5（35min）留出测试与消融**
  产出：`outputs/metrics.csv`（含 gate=on/off 全表）、`outputs/figures/roc_pr.png`、`confusion_matrix.png`、`feature_importance.png`。
  DoD：test 集指标 + 500 次 group 自助 95% 区间；report 所需的每个数字都能在 csv 中查到。
- [x] **M6（30min）单图报告与文档**
  产出：`src/predict.py`、`outputs/figures/single_image_report.png`、`outputs/report.md`、`outputs/slides.md`、`README.md` 定稿、`outputs/figures/flowchart.png`、`class_distribution.png`。
  DoD：一条命令可复现单图报告；report.md 含真实数字、局限与参考文献。
