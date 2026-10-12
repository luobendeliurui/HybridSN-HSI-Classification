<div align="center">

# HybridSN 高光谱图像分类复现与实验分析

**HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification 复现**

论文复现 · 消融实验 · 对比实验 · 独立思考

刘睿　|　学号：**107552603696**　|　2026-10

</div>

---

> **仓库地址**：https://github.com/luobendeliurui/HybridSN-HSI-Classification
>
> ```bash
> git clone https://github.com/luobendeliurui/HybridSN-HSI-Classification.git
> cd HybridSN-HSI-Classification
> ```

---

## 目录

- [一、项目目标](#一项目目标)
- [二、论文与开源代码来源（声明）](#二论文与开源代码来源声明)
- [三、数据集](#三数据集)
- [四、环境配置](#四环境配置)
- [五、目录结构](#五目录结构)
- [六、运行方法](#六运行方法)
- [七、参数设置](#七参数设置)
- [八、主要结果](#八主要结果)
- [九、实验内容与任务对应](#九实验内容与任务对应)
- [十、结果图表索引](#十结果图表索引)
- [十一、已知局限](#十一已知局限)

---

## 一、项目目标

本项目以经典高光谱图像分类模型 **HybridSN** 为对象，完整走通高光谱分类实验的标准流程：

```
论文阅读 → 环境配置 → 数据认识 → 模型复现 → 结果评价 → 对比实验 → 总结分析 → GitHub 整理
```

**主要工作**：

1. 复现 HybridSN 在 **Pavia University** 数据集上的分类结果，与论文报告值及作者开源代码结果对照；
2. 完成 **训练样本比例对比实验**（5% / 10% / 20% / 30% 四组）；
3. 完成 **三组组件消融实验**（移除 Dropout / 移除 2-D 卷积 / 纯 2-D-CNN baseline）；
4. 在消融基础上自行设计 **严格控制系统变量的对照实验**（25×25 patch 中心 11×11 置零），分离出两类混淆机制。

> 本项目的重点不是追求最高精度，而是**理解完整流程、独立完成实验、解释实验结果**。

---

## 二、论文与开源代码来源（声明）

### 2.1 原论文

Roy, S. K., Krishna, G., Dubey, S. R., & Chaudhuri, B. B.
**HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification.**
*IEEE Geoscience and Remote Sensing Letters*, 17(2), 277–281, 2020.
DOI: [10.1109/LGRS.2019.2918719](https://doi.org/10.1109/LGRS.2019.2918719)

### 2.2 参考开源代码

| 来源 | 链接 |
|---|---|
| 官方代码（Keras） | https://github.com/gokriznastic/HybridSN |
| PyTorch 参考实现 | https://github.com/Pancakerr/HybridSN |

### 2.3 ⚠️ 原创性声明

> **本仓库的模型结构、数据划分方式与训练流程来源于上述公开代码，本人未声称为原创。**
>
> 本人在此基础上**独立完成**的内容包括：
> - 环境适配（论文版 PyTorch 1.9.0 + CUDA 11.1 无法适配本机 GPU，改为 PyTorch 2.10.0 + CUDA 13.0）与依赖梳理；
> - 全部实验的组织、执行与结果记录（训练样本比例对比、三组组件消融、邻域对照实验）；
> - 自行编写的分析脚本与全部图表（地物光谱曲线、混淆矩阵、趋势图、消融总览图、机制示意图等）；
> - 全部实验结论与独立发现（详见第八、九节）。

---

## 三、数据集

### 3.1 Pavia University（PU）

| 项目 | 值 |
|---|---|
| 传感器 | ROSIS |
| 影像尺寸 | **610 × 340** 像元（空间分辨率 1.3 m） |
| 波段数 | **103**（430–860 nm） |
| 类别数 | **9** |
| 有标注像元 | **42,776** |

**类别分布**

| # | 类别 | 像元数 | 占比 |
|---|---|---|---|
| 1 | Asphalt | 6,631 | 15.5% |
| 2 | Meadows | 18,649 | 43.6% |
| 3 | Gravel | 2,099 | 4.9% |
| 4 | Trees | 3,064 | 7.2% |
| 5 | Painted metal sheets | 1,345 | 3.1% |
| 6 | Bare Soil | 5,029 | 11.8% |
| 7 | Bitumen | 1,330 | 3.1% |
| 8 | Self-Blocking Bricks | 3,682 | 8.6% |
| 9 | Shadows | 947 | 2.2% |
| | **合计** | **42,776** | 100% |

> **最大类是最小类的 19.7 倍** —— 这是总体精度（OA）与平均精度（AA）出现差异的根本原因。

### 3.2 数据获取与放置（⚠️ 本仓库不包含原始数据集）

受仓库体积限制，**不上传原始数据文件**，请自行下载后按下列路径放置：

```
dataset/
└── PaviaU/
    ├── PaviaU.mat        (~33.2 MB)
    └── PaviaU_gt.mat
```

**公开下载地址**（任选其一）：

- https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- http://www.ehu.es/ccwintco/uploads/e/ee/PaviaU.mat 与 http://www.ehu.es/ccwintco/uploads/7/72/PaviaU_gt.mat

放置完成后运行环境自检，`[Dataset files]` 两项应显示 `[OK]`：

```bash
python check_env.py
```

### 3.3 数据划分

采用分层抽样（`stratify=y`, `random_state=100`），先取训练集、再从剩余部分按 1:6 划分验证/测试：

| 集合 | 像元数 | 占比 | 用途 |
|---|---|---|---|
| 训练集 | 12,832 | 30% | 更新模型权重 |
| 验证集 | 4,277 | 10% | 选取最优模型 |
| 测试集 | 25,667 | 60% | 最终评估（仅使用一次） |

> 训练比例实验中，比例越低则测试集越大（5% → 36,361；10% → 34,222；20% → 29,944；30% → 25,667），比较时以 OA 为准。

---

## 四、环境配置

### 4.1 实际环境

| 组件 | 版本 |
|---|---|
| 操作系统 | Windows 11 Pro (build 26100) |
| CPU / 内存 | Intel i5-12600KF (10C/16T) / 31.8 GB |
| GPU | NVIDIA GeForce RTX 5060 (8 GB, sm_120) |
| Python | 3.10.0 |
| **PyTorch** | **2.10.0+cu130** |
| torchvision | 0.25.0+cu130 |
| **CUDA** | **13.0** |
| cuDNN | 91200 |
| numpy / scipy / scikit-learn | 2.2.6 / 1.15.3 / 1.7.2 |
| matplotlib / spectral / torchinfo / tqdm | 3.10.9 / 0.22.4 / 1.8.0 / 4.67.3 |

### 4.2 ⚠️ 与论文的版本差异及原因

| 组件 | 论文 | 本项目 | 原因 |
|---|---|---|---|
| PyTorch | 1.9.0 | **2.10.0** | 论文版仅提供 cp39 wheel 且依赖 CUDA ≤ 11.1，**无法适配本机 GPU 架构（sm_120）** |
| CUDA | 11.1 | **13.0** | 同上 |
| spectral | 0.20 | **0.22.4** | 0.22.2 为源码包需 MSVC 编译；0.22.4 提供纯 Python wheel，**API 兼容** |

**一致性保障**：固定全部随机种子（`RANDOM_SEED = 666`，同时设置 `random` / `numpy` / `torch` / `torch.cuda`），并对齐超参数与数据划分方式。

> **可复现性验证**：消融① 的两次独立运行结果**逐位相同**（OA 均为 `0.9990649472084778`），说明在移除 Dropout 后本环境下的训练过程完全确定。

详细环境说明见 [`docs/环境说明.md`](docs/环境说明.md)，依赖清单见 [`docs/依赖与框架清单.md`](docs/依赖与框架清单.md)。

### 4.3 依赖安装

```bash
pip install -r requirements.txt
```

> `requirements.txt` 已包含 PyTorch 官方 CUDA 索引（`--extra-index-url https://download.pytorch.org/whl/cu130`）。
> 若使用其他 CUDA 版本，请先按 https://pytorch.org 的指引安装对应版本的 PyTorch，再安装其余依赖。

---

## 五、目录结构

```
HybridSN-HSI-Classification/
├── README.md                      本文件
├── requirements.txt               依赖清单（含 CUDA 索引）
├── LICENSE                        许可与来源说明
├── check_env.py                   环境自检脚本
├── HybridSN.ipynb                 主实验代码（含全部输出）
├── HybridSN.py                    脚本版（可命令行运行）
│
├── docs/                          文档
│   ├── 实验报告.pdf / .docx         完整实验报告 ★（报告类均为 Word / PDF）
│   ├── 论文阅读总结.docx            论文精读笔记与结论卡
│   ├── 任务1_论文理解.docx          任务 1 的 5 个问题答案
│   ├── 任务6_总结与思考题.docx       任务 6 的 8 道思考题答案
│   ├── 任务5_对比实验结果与分析.docx  训练比例对比实验分析
│   ├── 任务5_消融实验结果与分析.docx  三组组件消融 + 邻域对照实验分析
│   ├── 任务5_消融实验方案与结果表.docx 消融实验设计文档
│   ├── 环境说明.md                  环境搭建过程与版本选择依据
│   └── 依赖与框架清单.md             全部依赖及用途
│
├── figures/                       结果图（19 张）★
│   ├── 01_model_architecture.png       模型结构示意图
│   ├── 02_spectral_curves.png          地物光谱曲线（9 类）
│   ├── 03_training_curve_baseline.png  基线训练曲线（Loss + Accuracy）★
│   ├── 04_rgb_origin.png               假彩色影像
│   ├── 05_ground_truth.png             Ground Truth ★
│   ├── 06_prediction_baseline.png      整幅分类结果图 ★
│   ├── 07_prediction_masked_baseline.png 分类结果（掩膜后）
│   ├── 08_confusion_matrix_baseline.png  基线混淆矩阵 ★
│   ├── 09_train_ratio_trend.png        训练比例趋势（4 子图）★
│   ├── 10_ablation_training_compare.png 消融训练过程对比
│   ├── 11_ablation_summary.png         四组消融结果总览
│   ├── 12_error_distribution.png       误差分布对比
│   ├── 13_mask_scheme.png              邻域对照实验方案示意
│   ├── 14_confusion_matrix_mask11.png  邻域对照实验混淆矩阵
│   ├── 15_confusion_mechanisms.png     两类混淆机制示意
│   ├── 16_env_check.png                环境自检输出
│   ├── 17_confusion_matrix_opt6_nir1.png 无光谱卷积消融混淆矩阵
│   ├── 18_confusion_matrix_patch11.png   patch=11 混淆矩阵
│   └── 19_training_curve_nodropout.png  无 Dropout 训练曲线
│
├── results/                       各实验的分类报告与训练记录 ★
│   ├── PU_tr30/                  基线（30% 训练）
│   │   ├── classfication_report.txt   OA/AA/Kappa + 逐类精度
│   │   ├── loss.csv / train_acc.csv / val_acc.csv
│   ├── PU_tr05/  PU_tr10/  PU_tr20/   训练比例对比实验
│   ├── ablation_nodropout/        消融① 移除 Dropout
│   ├── ablation_no2dconv/         消融② 移除 2-D 卷积（纯 3-D）
│   ├── ablation_2dcnn_only/       消融③ 纯 2-D-CNN baseline
│   ├── opt6_nir1/                 独立探索：移除光谱方向卷积
│   ├── opt2_patch11/              独立探索：patch 缩至 11
│   └── opt2b_mask11/              独立探索：25×25 + 中心 11×11 置零 ★
│
└── dataset/
    └── README.md                  数据放置说明（原始数据不随仓库上传）
```

---

## 六、运行方法

### 6.1 环境自检（建议先运行）

```bash
python check_env.py
```

期望输出末尾为：

```
RESULT: ALL CHECKS PASSED
```

同时会校验 PyTorch/CUDA、GPU 可用性（含矩阵乘冒烟测试）与数据集可读性。

### 6.2 运行主实验（推荐 Jupyter Notebook）

```bash
jupyter lab HybridSN.ipynb
```

按顺序执行全部单元格即可。**关键节点**：

| 单元格 | 内容 | 核对点 |
|---|---|---|
| 第 2 格 | 超参数配置 | `DATASET='PU'`、`TRAIN_RATE=0.3`、`N_PCA=15`、`PATCH_SIZE=25` |
| 第 16 格 | 模型定义与结构摘要 | `Total params: 4,844,793` |
| 第 17 格 | 训练（100 epoch，约 6 分钟） | Loss / Accuracy 曲线 |
| 第 19 格 | 整幅影像推理 | 输出 `best model: epoch_79_acc_1.0000.pth` |
| 第 20 格 | 分类报告 + 混淆矩阵 | 生成 `classfication_report.txt` 与 `confusion_matrix.png` |

### 6.3 命令行运行

```bash
python HybridSN.py
```

> `HybridSN.py` 中的 `DATASET` 变量控制数据集（`'PU'` / `'IP'` / `'SA'`）。

### 6.4 复现本项目中的其他实验

各实验通过修改 Notebook 第 2 格的 `SAVE_PATH` 与相应模型/训练代码实现，改动清单见：

- **训练比例对比**：改 `TRAIN_RATE`（0.05 / 0.10 / 0.20 / 0.30），详见 [`docs/任务5_对比实验结果与分析.docx`](docs/任务5_对比实验结果与分析.docx)
- **组件消融**：改 Dropout 概率 / `conv4` 结构 / 模型类，详见 [`docs/任务5_消融实验结果与分析.docx`](docs/任务5_消融实验结果与分析.docx)
- **邻域对照实验**：在 `PatchSet.__getitem__` 中对 patch 施加中心 11×11 掩膜，详见 [`docs/任务5_消融实验结果与分析.docx`](docs/任务5_消融实验结果与分析.docx)

> ⚠️ **每次切换实验务必修改 `SAVE_PATH`**，否则会覆盖已有结果目录。

---

## 七、参数设置

### 7.1 训练参数（全部实验保持一致，除被研究因素外）

| 参数 | 值 | 说明 |
|---|---|---|
| 数据集 | PU | Pavia University |
| PCA 主成分数 `N_PCA` | **15** | 保留 15 个主成分（`whiten=True`） |
| Patch 尺寸 `PATCH_SIZE` | **25** | 25 × 25 邻域 |
| 训练/验证/测试比例 | **30% / 10% / 60%** | 分层抽样 |
| Batch Size | **256** | |
| 优化器 | **Adam** | `lr = 0.001`, `weight_decay = 1e-6` |
| 损失函数 | CrossEntropyLoss | |
| Epoch | **100** | 每 5 轮验证一次 |
| 随机种子 | **666** | `random` / `numpy` / `torch` / `torch.cuda` 全部固定 |
| 划分种子 | `random_state = 100` | |

### 7.2 模型结构

```
输入 patch (15, 25, 25)                    ← 15 个 PCA 主成分
   ↓ unsqueeze(1)
(N, 1, 15, 25, 25)
   ↓ Conv3d(1→8,  (7,3,3)) + ReLU          ← 光谱核 7
   ↓ Conv3d(8→16, (5,3,3)) + ReLU          ← 光谱核 5
   ↓ Conv3d(16→32,(3,3,3)) + ReLU          ← 光谱核 3
(N, 32, 3, 19, 19) → view → (N, 96, 19, 19)
   ↓ Conv2d(96→64, (3,3)) + ReLU
(N, 64, 17, 17) → flatten → (N, 18496)
   ↓ Linear(18496→256) + ReLU + Dropout(0.4)
   ↓ Linear(256→128)   + ReLU + Dropout(0.4)
   ↓ Linear(128→9)
```

**参数量 4,844,793**，计算量 **50.95 M mult-adds**。

> 卷积层占 **90.6% 的 FLOPs**，但第一个全连接层 `dense1` 独占 **97.7% 的参数量** ——
> 这一结构特征在消融实验中被证明具有重要意义（详见第八节）。

---

## 八、主要结果

### 8.1 基线复现结果

| 指标 | **本次复现** | 论文报告（UP） | 作者代码 |
|---|---|---|---|
| **OA** | **0.999494** | 0.9998 | 0.99973 |
| **AA** | **0.999225** | 0.9997 | 0.99956 |
| **Kappa** | **0.999329** | 0.9998 | 0.99964 |
| 测试样本数 | 25,667 | 42,776 | 25,667 |

> 与论文差 0.03 pp，**源于测试集划分不同**（论文在全部 42,776 个标注像元上评估，本项目使用 60% 测试集）。

**逐类精度**

| 类别 | Precision | Recall | F1 | 支持数 |
|---|---|---|---|---|
| Asphalt | 0.9995 | 1.0000 | 0.9997 | 3,979 |
| Meadows | 0.9991 | 1.0000 | 0.9996 | 11,190 |
| Gravel | 1.0000 | 1.0000 | 1.0000 | 1,259 |
| **Trees** | 0.9989 | **0.9935** | 0.9962 | 1,839 |
| Painted metal sheets | 1.0000 | 1.0000 | 1.0000 | 808 |
| Bare Soil | 1.0000 | 1.0000 | 1.0000 | 3,017 |
| Bitumen | 1.0000 | 1.0000 | 1.0000 | 798 |
| Self-Blocking Bricks | 1.0000 | 0.9995 | 0.9998 | 2,209 |
| Shadows | 1.0000 | 1.0000 | 1.0000 | 568 |

**混淆矩阵中仅 3 个非对角单元非零**（共 13 例错分）：

| 混淆路径 | 数量 | 占比 |
|---|---|---|
| **Trees → Meadows** | **10** | **76.9%** |
| Trees → Asphalt | 2 | 15.4% |
| Self-Blocking Bricks → Trees | 1 | 7.7% |

见 [`results/PU_tr30/classfication_report.txt`](results/PU_tr30/classfication_report.txt) 与 [`figures/08_confusion_matrix_baseline.png`](figures/08_confusion_matrix_baseline.png)。

### 8.2 对比实验：训练样本比例

| 指标 | 5% | 10% | 20% | **30%** |
|---|---|---|---|---|
| **OA** | 0.994307 | 0.997925 | **0.999566** | 0.999494 |
| **AA** | 0.988631 | 0.995430 | 0.999026 | **0.999225** |
| **Kappa** | 0.992458 | 0.997250 | 0.999425 | 0.999329 |
| **错分像元** | **207** | **71** | **13** | **13** |
| 测试样本数 | 36,361 | 34,222 | 29,944 | 25,667 |
| **OA − AA** | 0.5676 pp | 0.2495 pp | 0.0540 pp | **0.0269 pp** |

**核心结论**

1. 精度随训练样本增加而提升，但**边际收益递减**（20% → 30% 已饱和，OA −0.0071 pp）；
2. ⭐ **OA 与 AA 的差距随样本减少单调扩大，最大达 21.1 倍** ——
   样本不足时模型优先保证多数类，**OA 会掩盖少数类的失效，AA 才能暴露它**；
3. 少数类与易混类最先退化（Shadows 召回 95.53% → 100%）；
4. ⚠️ **但"样本少"并非退化唯一原因**：Bitumen（67 个训练样本）与 Painted metal sheets（67 个）
   在 5% 配置下仍保持 **100% 召回率** —— **判别信息的独特性比样本数量更关键**。

见 [`figures/09_train_ratio_trend.png`](figures/09_train_ratio_trend.png)。

### 8.3 消融实验：三组组件消融

| 指标 | **基线** | 消融① 无 Dropout | 消融② 纯 3-D | 消融③ 纯 2-D |
|---|---|---|---|---|
| **Total params** | 4,844,793 | 4,844,793 | 8,926,393 | **4,794,681** |
| ΔParams | — | 0% | **+84%** | **−1.0%** |
| **mult-adds** | 50.95 M | 50.95 M | 39.09 M | **12.84 M** |
| **OA** | **0.999494** | 0.999065 | 0.999377 | 0.999377 |
| **AA** | 0.999225 | 0.998496 | 0.998783 | **0.999710** |
| **Kappa** | **0.999329** | 0.998761 | 0.999174 | 0.999174 |
| **错分像元** | **13** | 24 | 16 | 16 |
| **OA − AA** | +0.0269 pp | +0.0569 pp | +0.0594 pp | **−0.0333 pp** |

**逐类错分分解**

| 类别 | 基线 | 消融① | 消融② | 消融③ |
|---|---|---|---|---|
| **Trees** | 12 | 15 | **12** | **1** |
| **Meadows** | **0** | 0 | **0** | **13** |
| Self-Blocking Bricks | 1 | 8 | 2 | 2 |
| Shadows | 0 | 1 | 2 | 0 |
| **合计** | **13** | **24** | **16** | **16** |

**四条结论**

1. **Dropout 是有效的正则化手段**：移除后三项指标全面下降，错分增加 **85%**；
   且 **AA 降幅为 OA 的 1.70 倍**，说明失去正则化后少数类退化更重；
2. **2-D 卷积对总体精度贡献有限**（仅 3 个像元），但对类别均衡影响明显（AA 降幅为 OA 的 3.78 倍）；
3. ⭐ **参数量增加 84%、计算量减少 23%，精度反而略降** ——
   说明 2-D 卷积阶段的空间特征提取能力**无法由全连接层的参数容量替代**；
4. ⭐⭐ **纯 2-D-CNN 在参数量对等（−1.0%）的条件下取得相当的总体精度与更高的平均精度，
   且计算量仅为 HybridSN 的 25%** ——
   **论文"3-D + 2-D 混合结构优于单一 2-D 结构"的主张，在本实验协议下未获支持。**

**⭐ 关键发现：总体精度完全相同，平均精度存在差异**

```
消融②（纯 3-D）OA = 0.9993766314723186
消融③（纯 2-D）OA = 0.9993766314723186
                    ↑ 16 位有效数字完全一致
```

两者**恰好都是 16 个错分**（总体精度只取决于错分数量），
但**误差的类别分布截然不同**：纯 3-D 为 Trees 12 / Meadows 0，纯 2-D 为 Trees 1 / Meadows 13，
导致**平均精度相差 0.0927 pp**。

> **该结果说明：仅依据总体精度评价模型性能存在局限。**

见 [`figures/12_error_distribution.png`](figures/12_error_distribution.png)。

### 8.4 独立探索：邻域对照实验（本项目的核心独立发现）

**问题的由来**：基线 13 例错分中，**10 例（76.9%）集中于 `Trees → Meadows`** 这一条路径。

**假说**：PU 为 1.3 m 分辨率城区影像，25×25 patch 覆盖 **32.5 m**，窗口必然跨越多个类别，
边界处"树"像元的邻域被最大类 Meadows（占比 43.6%）主导 → **邻域污染**。

**验证的难点**：直接缩小 patch 会同时使参数量下降 94.7%，性能变化无法归因。

**方案设计**

| | **方案 A：缩小 patch** | **方案 B：25×25 + 中心 11×11 置零** |
|---|---|---|
| 做法 | patch 尺寸改为 11 | 输入张量仍为 25×25，外围 7 圈置 0 |
| 有效邻域 | 11×11 | 11×11 |
| **参数量** | 257,273（−94.7%） | **4,844,793（完全一致）** |
| **计算量** | 3.49 M（−93.1%） | **50.95 M（完全一致）** |
| **变化的因素数** | **2 个** | **1 个** |

**结果**

| 指标 | 原实验（25×25 全信息） | **方案 B（11×11 置零）** | 变化 |
|---|---|---|---|
| **OA** | 0.9994935 | **0.9997662** | **+0.0273 pp** |
| **AA** | 0.9992247 | **0.9995023** | **+0.0278 pp** |
| **Kappa** | 0.9993287 | **0.9996902** | **+0.0362 pp** |
| **错分像元** | **13** | **6** | **−7（−54%）** |
| **参数量 / 计算量** | 4,844,793 / 50.95 M | **完全一致** | — |

**因素分离（三方逐条对照）**

| 错误路径 | 原实验 | 方案 A | **方案 B** |
|---|---|---|---|
| Trees → Meadows | 10 | 4 | **3** |
| **Asphalt → Bitumen** | 0 | **4** | **0** |
| **Asphalt → Bare Soil** | 0 | **2** | **0** |
| **Self-Blocking Bricks → Gravel** | 0 | **2** | **0** |
| **总错分** | **13** | **15** | **6** |

**⭐ 核心结论：两类混淆机制**

| | **类型 A：邻域污染型** | **类型 B：光谱相似型** |
|---|---|---|
| 代表 | Trees ↔ Meadows | Asphalt ↔ Bitumen（光谱角 1.81°）、Gravel ↔ Self-Blocking Bricks（0.96°） |
| 成因 | 边界像元的邻域被**多数类主导** | **判别容量不足**，难以区分光谱高度相似的类别 |
| 优化方向 | **缩减有效邻域** | **增加或保持模型容量** |
| 方向关系 | ⬅ **两者相反** ➡ | |

> **该结论说明：单一配置难以在所有类别上同时取得最优性能。**

**实用建议**：在 PU 类城区高光谱数据上，将 patch 的有效邻域缩减至 **11×11 ~ 15×15**，
**在不增加任何参数与计算成本的前提下**，可将 OA 由 99.949% 提升至 99.977%、错分像元由 13 降至 6。

**附带发现：参数存在显著冗余**

参数量减少 94.7%、计算量减少 93.1% 时，OA 仅下降 **0.0078 pp**、AA 仅下降 **0.0008 pp**；
冗余集中于第一个全连接层（占全模型参数量 **97.7%**）。

### 8.5 结果汇总

| 配置 | params | mult-adds | OA | AA | Kappa | 错分 |
|---|---|---|---|---|---|---|
| **邻域对照（方案 B）** | 4,844,793 | 50.95 M | **0.9997662** | 0.9995023 | **0.9996902** | **6** |
| 独立探索（无光谱卷积） | 5,051,721 | 87.76 M | 0.9996494 | 0.9991858 | 0.9995353 | 9 |
| **基线** | 4,844,793 | 50.95 M | 0.9994935 | 0.9992247 | 0.9993287 | 13 |
| 独立探索（patch=11） | **257,273** | **3.49 M** | 0.9994156 | 0.9992163 | 0.9992256 | 15 |
| 消融③（纯 2-D） | 4,794,681 | 12.84 M | 0.9993766 | **0.9997099** | 0.9991742 | 16 |
| 消融②（纯 3-D） | 8,926,393 | 39.09 M | 0.9993766 | 0.9987831 | 0.9991738 | 16 |
| 消融①（无 Dropout） | 4,844,793 | 50.95 M | 0.9990649 | 0.9984957 | 0.9987607 | 24 |

> ⭐ **总体精度、类别均衡与计算效率三个维度的最优值分属三个不同配置** ——
> 说明高光谱分类**不存在单一最优结构**，需依据任务侧重选择。

---

## 九、实验内容与任务对应

| 考核任务 | 本项目对应内容 | 产出位置 |
|---|---|---|
| **任务 1** 阅读论文并理解高光谱分类 | H×W×B 数据形式、GT 含义、HybridSN 核心思想、OA/AA/Kappa 指标（5 个问题） | [`docs/任务1_论文理解.docx`](docs/任务1_论文理解.docx)、[`docs/论文阅读总结.docx`](docs/论文阅读总结.docx) |
| **任务 2** 配置环境并跑通开源代码 | 环境适配与自检、数据集读取、训练测试流程 | [`docs/环境说明.md`](docs/环境说明.md)、[`check_env.py`](check_env.py)、[`requirements.txt`](requirements.txt) |
| **任务 3** 完成 PU 分类实验 | 数据尺寸/波段/类别统计、GT 可视化、9 类光谱曲线、PCA/Patch/划分说明、OA/AA/Kappa/逐类精度/混淆矩阵 | [`figures/02_spectral_curves.png`](figures/02_spectral_curves.png)、[`results/PU_tr30/`](results/PU_tr30/)、[`figures/08_confusion_matrix_baseline.png`](figures/08_confusion_matrix_baseline.png) |
| **任务 4** 生成整幅分类图并分析错误 | 整幅预测、伪彩色/GT/分类结果三联对比、易混类别分析、误差空间分布分析（>300 字） | [`figures/04_rgb_origin.png`](figures/04_rgb_origin.png)、[`figures/05_ground_truth.png`](figures/05_ground_truth.png)、[`figures/06_prediction_baseline.png`](figures/06_prediction_baseline.png)、[`docs/实验报告.pdf`](docs/实验报告.pdf) |
| **任务 5** 对比/消融实验 | 训练比例 4 组 + 组件消融 3 组 + 独立探索 3 组，**共 10 组**（要求 ≥3 组） | [`docs/任务5_对比实验结果与分析.docx`](docs/任务5_对比实验结果与分析.docx)、[`docs/任务5_消融实验结果与分析.docx`](docs/任务5_消融实验结果与分析.docx) |
| **任务 6** 总结、反思并发布 GitHub 项目 | 本仓库：README、代码、依赖、结果、报告、来源声明；8 道总结思考题 | 本仓库、[`docs/任务6_总结与思考题.docx`](docs/任务6_总结与思考题.docx) |

---

## 十、结果图表索引

| 图 | 内容 | 对应要求 |
|---|---|---|
| [`01_model_architecture.png`](figures/01_model_architecture.png) | HybridSN 模型结构示意图 | 论文阅读总结与模型结构图 |
| [`02_spectral_curves.png`](figures/02_spectral_curves.png) | 9 类地物平均光谱曲线 + 植被类 ±1σ | ≥3 类地物光谱曲线 |
| [`03_training_curve_baseline.png`](figures/03_training_curve_baseline.png) | **基线 Loss 曲线 + Accuracy 曲线** | Loss/Accuracy 曲线 |
| [`04_rgb_origin.png`](figures/04_rgb_origin.png) | 假彩色影像 | 伪彩色图像 |
| [`05_ground_truth.png`](figures/05_ground_truth.png) | **Ground Truth** | Ground Truth |
| [`06_prediction_baseline.png`](figures/06_prediction_baseline.png) | **整幅分类结果图** | 分类结果图 |
| [`08_confusion_matrix_baseline.png`](figures/08_confusion_matrix_baseline.png) | **基线混淆矩阵** | Confusion Matrix |
| [`09_train_ratio_trend.png`](figures/09_train_ratio_trend.png) | 训练比例趋势（4 组） | 对比实验结果 |
| [`10_ablation_training_compare.png`](figures/10_ablation_training_compare.png) | 消融训练过程对比 | 消融实验结果 |
| [`11_ablation_summary.png`](figures/11_ablation_summary.png) | 四组消融结果总览 | 消融实验结果 |
| [`12_error_distribution.png`](figures/12_error_distribution.png) | 误差分布对比 | 消融实验结果 |
| [`13_mask_scheme.png`](figures/13_mask_scheme.png) | 邻域对照实验方案示意 | 独立探索 |
| [`14_confusion_matrix_mask11.png`](figures/14_confusion_matrix_mask11.png) | 邻域对照实验混淆矩阵 | 独立探索 |
| [`15_confusion_mechanisms.png`](figures/15_confusion_mechanisms.png) | 两类混淆机制示意 | 独立思考 |
| [`16_env_check.png`](figures/16_env_check.png) | 环境自检输出 | 环境可复现证据 |

---

## 十一、已知局限

1. **消融实验参数量不对等**：消融②（纯 3-D）的参数量比基线多 84%，该组精度差异无法纯归因于结构；
2. **单次运行**：各组均为单次实验，3–11 个像元的差异处于波动边缘，未做重复实验与显著性检验；
3. **仅单一数据集**：本仓库的消融与独立探索**仅在 PU 数据集上完成**，结论能否迁移至 IP / SA 尚未验证；
4. **邻域对照实验的合成输入**：方案 B 通过置零构造"有效邻域"，与"真实的 11×11 patch"在输入形式上并不完全等价
   （模型在外围看到零帧，而真实小 patch 无零帧），这是该方法固有的局限；
5. **光谱角为逐波段的粗粒度度量**：无法完整反映多维联合可分性；
6. **少数类错分可能含标注噪声**：追求 100% 精度可能是在拟合标注误差。

---

## 参考

- Roy, S. K., et al. (2020). HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification. *IEEE GRSL*, 17(2), 277–281.
- gokriznastic. HybridSN (Keras). https://github.com/gokriznastic/HybridSN
- Pancakerr. HybridSN (PyTorch). https://github.com/Pancakerr/HybridSN
- Pavia University 数据集. https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
