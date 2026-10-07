# Q-Derm 冻结方案（唯一权威规格）
> 本文件是项目的冻结设计规格与唯一权威依据。任何实现步骤开工前必须先完整阅读；不得自行更改范围、指标口径与目录结构。若与任何其他文件冲突，以本文件为准。

## 0. 项目与边界
- 名称：质控驱动的皮肤镜病灶筛查；代号 **Q-Derm**。
- 一句话：在公开皮肤镜影像上构建"图像质量门控 + 可解释特征"的黑色素瘤二分类筛查原型，并量化质量门控对覆盖率与分类性能的影响。
- 任务定义（二分类，仅使用 diagnosis_2 精确等于以下两者的样本）：
  - label=1 正类：`Malignant melanocytic proliferations (Melanoma)`
  - label=0 负类：`Benign melanocytic proliferations`
- **Must**：质量检测 / 病灶粗分割 / 形态+颜色共 20 个可解释特征 / 二分类 / 单图可视化报告 / 留出测试 + 门控消融。
- **Won't**：临床诊断结论、以毫米为单位的直径特征（无尺度标定）、深度 CNN 或任何 torch/tensorflow 方案、动静脉或全身慢病推断、任何未实际运行得到的指标。
- **诚实性红线**：禁止编造、修饰、估算或手写任何数据与指标；禁止报告无真值的 Dice；未达指标目标就如实报告并分析原因。

## 1. 环境（已实测，禁止更改）
- 唯一解释器：`D:\anaconda\python.exe`。已装：opencv-python-headless 5.0.0.93、numpy 2.4.6、scikit-learn 1.9.0、scipy 1.18、scikit-image、matplotlib、pandas、Pillow。
- **禁止**创建 venv / conda 新环境；**禁止**安装 torch、tensorflow、pytest、shap 等重型依赖。除非确有必要，尽量不安装任何包。
- 调用形式：`& "D:\anaconda\python.exe" src\collect.py --limit ...`
- Windows 控制台编码：所有脚本内一律显式 `encoding="utf-8"` 写文件；命令行前先 `$env:PYTHONIOENCODING="utf-8"`。禁止把中文直接 echo 到控制台后依赖其编码。
- Shell 为 Windows PowerShell 5.1：禁止使用 `&&`、`//`、三元运算符等 PS7 语法；复杂逻辑一律写进 .py 文件再执行。
- 磁盘：E: 盘剩余约 11GB，`data/` 目录硬上限 **900MiB**。
- 随机种子固定 **42**；所有抽样、划分、退化、自助抽样必须可复现。

## 2. 数据协议
- 接口：`GET https://api.isic-archive.com/api/v2/images/?limit=100&cursor=<cursor>`。该端点**仅支持 limit 与 cursor**（已读 openapi.json 确认），`diagnosis/collection/query` 等参数会被静默忽略——**禁止用它们做筛选**。
- 客户端筛选：只保留 `metadata.clinical.diagnosis_2` 精确等于上述两个取值的记录。
- 配额：label=1 共 400 张 + label=0 共 400 张 = 800 张。若 50 分钟内未达标，按 300+300 → 250+250 依次下调，并如实记录实际数量。
- 下载：优先 `files.full.url`；失败重试 3 次（指数退避）；仍失败则标记 `download_status=failed` 并继续换样本补足。
- 容量：`data/` 累计写入达 900MiB 即停止下载，并在报告中说明。
- 去重与分组：sha256 完全相同→丢弃；pHash（64bit）汉明距离 ≤6 视为近重复，同组 `group_id`。**划分必须以 group_id 为单位**（GroupShuffleSplit 语义），train/val/test = 60/20/20，按 label 分层，种子 42。
- `data/manifest.csv` 列顺序固定：
  `isic_id,diagnosis_1,diagnosis_2,diagnosis_3,label,image_url,path,bytes,sha256,phash,group_id,split,download_status`
- 溯源：把拉到的原始元数据逐行存 `data/api_meta_snapshot.jsonl`（附加分用）。

## 3. 视觉流水线（src/vision.py）
### 3.1 质量度量（每张图先缩放到长边 512 再计算，保证尺度一致）
`blur_lapvar` 灰度 Laplacian 方差；`exposure_mean` L 通道均值；`exposure_clip_hi_ratio` L≥250 占比；`exposure_clip_lo_ratio` L≤20 占比；`contrast_std` L 标准差；`glare_ratio` L≥240 且 S≤40 占比；`occlusion_border_ratio` L≤35 且与图像边框连通像素占比。
### 3.2 光照校正与病灶粗分割
- 光照校正：LAB 的 L 通道用大核（≈短边 1/8，奇数）形态学闭运算估计背景，相除或相减后再 CLAHE(clipLimit=2.0, tile 8×8)。
- 分割：Lab 的 a* 通道 Otsu → 形态学开闭（椭圆核）→ 取最大连通域 → 填孔 → 掩膜。若掩膜面积占比 <2% 或 >95%，判定分割失败，退化为"图像中心 60% 区域为 ROI"，并置 `seg_fallback=1`。
- 输出叠加图 ≥20 张（须含失败案例）到 `outputs/figures/seg_overlays/`。
### 3.3 特征（严格 20 个，均按掩膜与包围盒归一化；禁止把绝对像素尺寸当特征）
形态 10：`asym_major`（1−IoU(掩膜, 掩膜关于主轴镜像)）、`asym_minor`、`circularity`(4πA/P²)、`solidity`(A/凸包面积)、`eccentricity`、`perimeter_norm`(P/包围盒对角线)、`bbox_aspect`(短边/长边)、`area_ratio`(A/图像面积)、`border_roughness`(P/凸包周长)、`radial_cv`(质心到边界距离的 std/mean，距离按包围盒对角线归一)。
颜色 10（病灶掩膜内 Lab）：`lab_mean_L/a/b`、`lab_std_L/a/b`、`color_count`(a*b* 上 k-means k=3，占比>5% 的簇数)、`dark_ratio`(L 低于掩膜内 20 分位的像素占比)、`center_edge_diff`(中心 50% 半径区域与外侧环带的 ΔE76)、`color_entropy`(a*/b* 各 16 bin 联合直方图的归一化香农熵)。
- `data/features.csv` 列顺序：`isic_id,split,group_id,label,` + 20 特征 + 7 质量量 + `seg_fallback`。禁止缺列、禁止 NaN。

## 4. 退化合成与门控协议
对 val 与 test 的原图分别施加 6 类 × 2 级 = 12 种退化（**仅在内存生成，禁止落盘副本**）：
gauss_blur（轻 σ=1.5 / 重 σ=3.0，按图幅归一）、motion_blur（轻核长=短边1% / 重3%）、under_exposure（×0.6 / ×0.35）、over_exposure（×1.4 / ×2.0 并截断）、glare（轻 1 个半径=短边8%高光斑 / 重 3 个半径15%）、occlusion（轻 1 个覆盖8%面积边框暗块 / 重 2 个共20%）。原图记 level=0。
- **门控阈值**：只用 train 原图估计各质量量的正常区间（1/99 分位或中位数±k·MAD）；在 **val** 上选工作点，约束"原图接受率 ≥90%"，目标"重度退化拒绝率最大化"。写入 `artifacts/thresholds.json`（含每量阈值、选择依据、val 上的覆盖率与拒绝率）。**test 绝不参与调参**。
- **分类阈值**：在 **val** 上选"灵敏度 ≥0.75 时特异度最大"的阈值，写入 `artifacts/thresholds.json`。

## 5. 建模（src/train.py）
- `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)`。
- 特征贡献：①标准化空间 LogReg 系数；②val 上 `permutation_importance(n_repeats=10, scoring="roc_auc", random_state=42)`。
- **禁止** SHAP、CNN、超参搜索。随机森林对照仅在时间富余时做。
- 划分以 group_id 为单位，禁止跨组泄漏。产物 `artifacts/model.joblib`（含 scaler、特征名顺序）。

## 6. 评测协议（src/evaluate.py）
- 原图 test 集必报：n、AUC、PR-AUC、灵敏度、特异度、混淆矩阵、拒判率、覆盖率；AUC 的 95% 自助区间（按 group 重抽 500 次）。
- `outputs/metrics.csv` 列固定：
  `set,level,deg_type,gate,n_input,n_accepted,coverage,reject_rate,auc,pr_auc,sens,spec,tn,fp,fn,tp`
  `set`∈{val,test}；`deg_type`∈{none,gauss_blur,motion_blur,under_exposure,over_exposure,glare,occlusion}；`level`∈{0,1,2}；`gate`∈{on,off}；gate=off 时 coverage=1.0、reject_rate=0.0。
- 分割稳定性：对 test 原图施加 ±5% 亮度与轻度模糊，计算掩膜 IoU 均值/标准差（命名 `seg_iou_stability`），**明确注明这不是 Dice、不是分割精度**。
- 所有数字必须来自真实运行输出。

## 7. 交付物与 DoD
每个任务结束必须：①更新 `TASKS.md` 勾选状态；②在 `outputs/session_log.md` 追加"做了什么/关键结果/问题/下一步"；③`git add -A; git commit -m "<Mx>: ..."`。
最终产物：
- `data/manifest.csv`、`data/features.csv`、`data/api_meta_snapshot.jsonl`
- `artifacts/model.joblib`、`artifacts/thresholds.json`
- `outputs/metrics.csv`
- `outputs/report.md`（中文技术报告：背景/数据/方法/结果/局限/参考文献）
- `outputs/slides.md`（演示幻灯片提纲）
- `outputs/figures/`：flowchart、class_distribution、seg_overlays、roc_pr、feature_importance、degradation_perf、coverage_perf、confusion_matrix、single_image_report
- `src/predict.py`：单图 → 一页可视化报告（原图 / 质量判定与理由 / 掩膜叠加 / 特征贡献条形 / 风险分与建议 / 免责声明）
- `README.md`：环境、数据来源与授权、一键复现命令、结果摘要、局限

## 8. 砍单顺序（超时按序砍）
① 细分退化参数扫描 → ② permutation_importance 重复次数 → ③ 非核心展示图 → ④ 随机森林对照。
**必须保留**：留出测试、门控消融、单图报告、report.md。
若第 4 小时结束时 test AUC < 0.65：按原结果交付"质控与失效分析原型"，报告中明确写出未达预期并分析原因，**禁止调参硬凑**。

## 9. 时间预算
M0 15 / M1 50 / M2 40 / M3 55 / M4 45 / M5 35 / M6 30 = 270 分钟，留 30 分钟缓冲。

## 10. 参考文献（报告须引用）
ISIC Archive API 文档 https://api.isic-archive.com/api/docs/swagger/ ；ISIC Archive https://www.isic-archive.com/ ；Tschandl P, et al. The HAM10000 dataset. Scientific Data, 2018；Codella N, et al. Skin Lesion Analysis Toward Melanoma Detection (ISIC 2018)；Nachbar F, et al. The ABCD rule of dermatoscopy. J Am Acad Dermatol, 1994；Stolz W, et al. ABCD rule of dermatoscopy (TDS)；scikit-learn permutation_importance 文档；OpenCV 官方文档。
