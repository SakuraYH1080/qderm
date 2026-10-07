# Q-Derm 执行日志

## 2026-10-07：M0 + M1

- 做了什么：建立 `src/`、`data/images/`、`artifacts/`、`outputs/figures/`；记录已安装依赖版本并编写 README 初稿；实现按 limit/cursor 分页、客户端精确标签过滤、下载重试、sha256 去重、pHash 分组和以组为单位的分层划分。运行采集并逐文件校验大小与 sha256，检查所有近重复对的分组及划分。
- 关键结果：API 扫描 108 页、10,800 条原始元数据；成功下载良性 400 张、黑色素瘤 400 张，共 800 张；完全重复丢弃 0 张，pHash 汉明距离 ≤6 的近重复对 12 对，形成 789 组。train：良性 240、黑色素瘤 240；val：良性 80、黑色素瘤 80；test：良性 80、黑色素瘤 80。`data/` 占用 115.344 MiB（上限 900 MiB）。请求重试 0 次，下载失败 0 次。目标 400+400 已达成，未触发降级。
- 遇到的问题：API 不支持服务端 diagnosis/collections/query 筛选；按冻结方案仅用 limit/cursor 并在客户端过滤。部分连续页没有目标黑色素瘤标签，继续顺序扫描后达标。本次未遇到 SSL EOF。
- 下一步：按 PLAN.md 的 M2 实现 7 项图像质量量、光照校正与病灶粗分割；本次未执行 M2。

## 2026-10-07：M2 + M3

- 做了什么：在 `src/vision.py` 实现长边 512 归一化、7 项质量量、LAB L 大核闭运算光照校正与 CLAHE、a* Otsu 粗分割和固定判定的中心 60% ROI 回退；在全量 800 张图上提取严格 20 个形态/颜色特征，断言列顺序、特征数与有限值，输出 `data/features.csv` 和 20 张分割叠加图（17 张成功、3 张 fallback）。在 `src/train.py` 仅取 train/val，用 StandardScaler + 固定参数 LogisticRegression 建模，在 val 上按灵敏度约束选择分类阈值，并运行 10 次重复的 AUC permutation importance。产出模型、阈值和 `outputs/train_metrics.json`。
- 关键结果：全量 `seg_fallback=3/800=0.375%`，原始最大连通域掩膜面积占比分位数（0/5/25/50/75/95/100%）为 `0.0272/0.0794/0.1610/0.2613/0.4148/0.6954/1.0000`；fallback 未超过 60%，无需触发分割调参。train n=480（0:240、1:240），val n=160（0:80、1:80）；val AUC=0.83484375、PR-AUC=0.81457527；分类阈值=0.48548549，灵敏度=0.7500、特异度=0.7875。标准化系数绝对值前五为 `lab_mean_a -0.7977`、`perimeter_norm -0.7912`、`border_roughness +0.6516`、`center_edge_diff -0.6334`、`radial_cv +0.5650`。permutation importance 前五（AUC 平均下降）为 `lab_mean_a 0.0990`、`border_roughness 0.0813`、`radial_cv 0.0731`、`center_edge_diff 0.0624`、`solidity 0.0338`；全部 20 项及其 10 次重复标准差见 `outputs/train_metrics.json`。
- 遇到的问题：3 张图的 a* Otsu 最大连通域几乎覆盖整幅皮肤（原始面积 >95%），按冻结规则回退；已在叠加图中保留失败实例。无人工分割真值，叠加图只用于定性检查，低 fallback 比例不能当作分割准确率。为限制全量运行成本，`color_count` 的 k-means 对掩膜像素做确定性均匀最多 4096 点取样。
- 下一步：M4 根据 train 原图质量量和 val 退化图选择门控阈值；本次未使用 test 进行建模、阈值选择或评估。

## 2026-10-07：M4 + M5

- 做了什么：在 `src/evaluate.py` 对 val/test 各 160 张原图分别于内存中生成 6 类 × 2 级退化；用 train 原图的七个质量量 1/99 分位数估计区间，在 val 上选择统一缩放系数，满足原图接受率 ≥90% 并最大化六类重度退化的合并拒绝率。保留 M3 分类阈值，仅在门控冻结后评估 test。输出固定列序的 52 行 `outputs/metrics.csv`、`outputs/evaluation_summary.json` 和 5 张评测图；原图 test AUC 按 `group_id` 重抽 500 次，分割稳定性比较每张原图与三种扰动后的掩膜 IoU。
- 关键结果：门控区间缩放系数 1.0；val 原图覆盖率 147/160=91.875%，val 重度退化合并拒绝率 81.7708%。test 原图 gate off：n=160、AUC=0.780469（group bootstrap 95% CI 0.710667–0.852997，500/500 次有效）、PR-AUC=0.763506、灵敏度=0.6500、特异度=0.7750、TN/FP/FN/TP=62/18/28/52、拒判率=0、覆盖率=1.0000。test 原图 gate on：148/160 接受、拒判率=0.0750、覆盖率=0.9250、AUC=0.775890、灵敏度=0.630137、特异度=0.773333。test 六类重度退化合并共 960 次输入（同一批 160 张图的六种视图），gate off AUC=0.661363、覆盖率=1.0000；gate on AUC=0.698247、覆盖率=0.19375（186/960）。`seg_iou_stability`=0.935731±0.122044，160 张图共 480 次掩膜比较；这是扰动下的掩膜一致性，不是 Dice，也不是分割精度。
- 遇到的问题：门控收益有明显代价且不均匀：原图 test 的 AUC 和灵敏度在 gate on 后略降；重度 glare 在 gate off 下 AUC=0.585938，gate on 仅接受 5/160 张、AUC=0.5；重度高斯模糊和欠曝门控后零接受，AUC 无定义，在 CSV 中留空。重度 occlusion gate off/on AUC=0.691875/0.675137，未获得改善。合并重度退化的 960 次输入不是独立患者样本。test 原图 AUC 未低于 0.65，无需触发低 AUC 预案；所有负结果按原值保留，未调整 test 阈值或剔除样本。
- 下一步：下一次任务再做 M6 的单图报告与文档；当前未执行 M6。

## 2026-10-07：M6

- 做了什么：实现 `src/predict.py` 单图推理与一页可视化报告，复用 M2/M4/M5 的特征、退化和冻结阈值；实现 `src/make_figures.py` 生成端到端流程与按 split×label 数据构成图；完成中文技术报告、演示提纲与 README 定稿。报告正文按去除空白字符计 6059 字符；八节标题、报告图片引用与 15 页提纲已程序核验。两例使用同一张 `ISIC_0000000.jpg`：原图质量合格、筛查分 0.399833；重度欠曝图在内存中生成，清晰度 Laplacian 方差 5.953374 低于下界 10.383628、亮度均值 53.500137 低于下界 100.248339，报告拒判并停止风险输出。两例报告、流程图和数据图均已目视检查；`predict.py` 与 `make_figures.py` 已通过语法编译。
- 交付物清单与文件大小（字节）：`src/predict.py` 9039 B；`src/make_figures.py` 4031 B；`outputs/report.md` 13613 B；`outputs/slides.md` 4455 B；`README.md` 3257 B；`outputs/figures/single_image_report.png` 810386 B；`outputs/figures/single_image_report_degraded.png` 644912 B；`outputs/figures/flowchart.png` 123099 B；`outputs/figures/class_distribution.png` 52567 B。沿用此前里程碑交付的 `data/manifest.csv`、`data/features.csv`、`data/api_meta_snapshot.jsonl`、`artifacts/model.joblib`、`artifacts/thresholds.json`、`outputs/metrics.csv`、20 张 `seg_overlays` 与五张评测图；尺寸以工作区实际文件为准。
- 遇到的问题：报告封面未填写无法核验的信息。图表中文字体使用本机可用的中文字体；拒判图上的贡献仅供失效分析。原图与重度欠曝图为同一病灶的两种视图，不构成独立样本。
- 遗留局限：合成退化不代表真实低质数据；平衡采样导致患病比例失真，PR-AUC 不能代表真实临床场景；无分割金标准故不报 Dice；未做临床有效性验证；标签来自 ISIC 元数据。真实场景推广前需逐图复核授权并开展外部与临床验证。
- 下一步：如需对外发布，应复核影像授权与署名。
