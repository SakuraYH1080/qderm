# Q-Derm · 质控驱动的皮肤镜病灶筛查

在公开皮肤镜影像上构建的筛查研究原型：**先判断"这张图能不能用"，再给出黑色素瘤风险分**，并同时输出可复核的判断依据。

> 本原型仅用于研究与教学，**不构成临床诊断**，不能替代医师判断。

## 为什么先做质量门控

失焦、欠曝、过曝、镜面眩光与遮挡在真实采集中非常常见。当一张不可用的图像进入分类器时，模型依然会给出一个高置信度分数——使用者看到的是"恶性概率 0.93"，却无从知道这个数字建立在无法辨认的输入之上。**不可信的高置信度结论，比没有结论更危险。**

Q-Derm 因此把图像质量从预处理细节提升为一个可解释的决策环节：质量量越界即拒判，并明确列出触发原因；只有通过质控的图像才进入风险评分。

## 方法概览

![端到端流程](outputs/figures/flowchart.png)

1. **采集与溯源**（`src/collect.py`）— 通过 ISIC Archive 公开 API 分页拉取元数据，按 `metadata.clinical.diagnosis_2` 精确筛选标签，下载影像并记录来源 URL、sha256、pHash、近重复组与划分归属。
2. **质量门控**（`src/vision.py`）— 七项质量量：Laplacian 方差、亮度均值、高/低亮截断占比、对比度标准差、眩光占比、边框连通暗遮挡占比。正常区间由训练集原图估计，工作点在验证集上选择，越界即拒判。
3. **光照校正与病灶粗分割** — LAB 的 L 通道大核背景校正 + CLAHE；a* 通道 Otsu + 形态学处理 + 最大连通域；面积异常时回退为中心 ROI 并标记。
4. **20 项可解释特征 + 逻辑回归**（`src/train.py`）— 10 项形态特征（不对称性、圆形度、实心度、偏心率、归一化周长、边界粗糙度、径向距离变异等）+ 10 项颜色特征（Lab 均值与标准差、颜色簇数、暗色占比、中心-边缘色差、颜色熵）。所有尺度相关量按包围盒归一化，不引入缺少标定的毫米直径。
5. **评测与单图报告**（`src/evaluate.py`、`src/predict.py`）— 门控消融、受控退化实验、自助法区间；单图输出"质量判定 → 轮廓 → 特征贡献 → 风险分 → 建议"一页报告。

设计取舍：项目的核心主张是**可解释与可信**，因此刻意选择可逐项追溯的传统图像处理与线性模型，而非不可解释的深度模型。逻辑回归系数可直接换算为单图特征贡献。

## 结果速览

数据：ISIC 公开 API，扫描 108 页 / 10,800 条元数据 → 良性黑色素细胞增生 400 张 + 黑色素瘤 400 张；sha256 去重后经 pHash 近重复分组形成 789 个病灶组，以病灶组为单位分层划分 60/20/20（种子 42），**跨划分组泄漏 = 0**。

| 留出测试集（原图，n=160） | 门控关闭 | 门控开启 |
|---|---:|---:|
| 覆盖率 / 拒判率 | 1.0000 / 0.0000 | 0.9250 / 0.0750 |
| ROC-AUC | 0.7805 | 0.7759 |
| PR-AUC | 0.7635 | 0.7551 |
| 灵敏度 / 特异度 | 0.6500 / 0.7750 | 0.6301 / 0.7733 |

AUC 按病灶组重抽 500 次的 95% 区间为 **0.7107–0.8530**。

| 六类重度退化合并（960 次输入） | 门控关闭 | 门控开启 |
|---|---:|---:|
| ROC-AUC | 0.6614 | 0.6982 |
| 覆盖率 | 1.0000 | 0.1938 |

**门控不是免费的**：它在原图上略微降低 AUC 与灵敏度，只在退化场景下才体现价值，且必须以覆盖率下降（拒绝 80.6% 的输入）为代价。重度眩光与重度遮挡未被有效拦截，这些负结果原样保留在 `outputs/metrics.csv` 中，未针对测试集调整任何阈值。

![ROC 与 PR](outputs/figures/roc_pr.png)
![退化类型与等级的影响](outputs/figures/degradation_perf.png)
![覆盖率与性能的权衡](outputs/figures/coverage_perf.png)

单图报告示例（左：质量合格；右：同图重度退化后被拒判并停止输出风险分）：

![单图报告](outputs/figures/single_image_report.png)
![拒判报告](outputs/figures/single_image_report_degraded.png)

## 目录结构

```text
Q-Derm/
├─ src/
│  ├─ collect.py         数据采集、去重、近重复分组与划分
│  ├─ vision.py          质量量、光照校正、粗分割、20 项特征
│  ├─ train.py           划分、标准化、逻辑回归、阈值冻结
│  ├─ evaluate.py        退化合成、门控消融、留出测试、自助法区间
│  ├─ predict.py         单图推理与一页可视化报告
│  └─ make_figures.py    流程图与数据构成图
├─ data/                 样本清单、特征表、元数据快照
├─ artifacts/            模型与冻结阈值
├─ outputs/              指标表、图表、技术报告、幻灯片提纲
├─ PLAN.md               冻结的设计规格
└─ TASKS.md              里程碑看板
```

## 环境与依赖

- Python 3.11+（本项目在 Windows + Python 3.14 上开发与验证）
- 依赖见 `requirements.txt`：opencv-python-headless、numpy、scikit-learn、scipy、scikit-image、matplotlib、pandas、Pillow、requests

```bash
pip install -r requirements.txt
```

无 GPU 亦可运行；全部随机过程固定种子 42。

## 数据

影像与元数据来自 [ISIC Archive](https://www.isic-archive.com/) 的[公开 API](https://api.isic-archive.com/api/docs/swagger/)。仅使用 `metadata.clinical.diagnosis_2` 精确等于 `Malignant melanocytic proliferations (Melanoma)` 或 `Benign melanocytic proliferations` 的记录。

- 原始影像**不随仓库分发**（第三方数据，出于版权与体积考虑）。请用 `src/collect.py` 重新获取；`data/manifest.csv` 记录了每张图的 ISIC ID、来源 URL、哈希与划分归属，可据此复现同一数据集。
- 逐图的 `copyright_license` 与 `attribution` 字段保留在 `data/api_meta_snapshot.jsonl`。展示、再发布或进一步使用影像前，请核对对应授权与署名要求。
- 标签来自 ISIC 元数据，未经本项目临床复核。本仓库中的示例图均为基于上述公开影像的处理结果，仅供研究展示。

## 快速开始

```bash
# 1) 采集：分页拉取元数据、客户端筛选、下载、去重分组与划分
python src/collect.py --limit 100 --target-per-class 400

# 2) 质量量与 20 项特征
python src/vision.py

# 3) 训练并冻结分类阈值（仅用 train/val）
python src/train.py

# 4) 退化合成、门控消融与留出测试
python src/evaluate.py

# 5) 补充流程图与数据构成图
python src/make_figures.py

# 6) 单图可视化报告
python src/predict.py --image data/images/ISIC_0000000.jpg
```

演示"坏图被拒判"（同一张图在内存中退化，不落盘）：

```bash
python src/predict.py --image data/images/ISIC_0000000.jpg --degrade gauss_blur --level 2
```

`src/predict.py` 接受任意本地单张皮肤镜图像；拒判时会列出越界的质量量、实测值与冻结区间，并停止输出风险分与分级建议。

## 项目边界与已知局限

- 合成退化由确定性算子生成，**不代表真实低质图像的复杂退化模式**；
- 为控制变量采用两类 1:1 平衡采样，患病比例失真，**PR-AUC 不代表真实临床场景**；
- 无专家分割金标准，故不报告 Dice；扰动下的掩膜 IoU 只是稳定性指标，不是分割精度；
- 未做任何临床有效性验证，模型分数未经真实患病率校准，不能解释为个体患病概率；
- 960 次退化输入来自同一批 160 张图，不是独立病例；
- 按 pHash 分组不能保证识别同一病灶的所有不同照片；门控会降低原图覆盖率。

## 参考文献

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Scientific Data*, 2018.
2. Codella N, et al. Skin Lesion Analysis Toward Melanoma Detection (ISIC 2018).
3. Nachbar F, et al. The ABCD rule of dermatoscopy. *J Am Acad Dermatol*, 1994.
4. ISIC Archive — <https://www.isic-archive.com/>
5. scikit-learn — permutation feature importance
6. OpenCV — <https://docs.opencv.org/>

## 许可

代码以 [MIT 许可](LICENSE) 发布。仓库内引用的第三方影像与元数据版权归其各自权利人所有，不受本许可覆盖。
