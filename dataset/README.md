# 数据集说明

> ⚠️ **本仓库不包含原始数据集文件。** 请按本文说明自行下载并放置。

---

## 一、Pavia University（PU）

| 项目 | 值 |
|---|---|
| 传感器 | ROSIS |
| 影像尺寸 | 610 × 340 像元（空间分辨率 1.3 m） |
| 波段数 | 103（430–860 nm） |
| 类别数 | 9 |
| 有标注像元 | 42,776 |

### 1.1 需要下载的文件

```
PaviaU.mat        影像数据立方体（约 33.2 MB）
PaviaU_gt.mat     人工标注真值图
```

### 1.2 公开下载地址

任选其一：

- **官方数据集页面（推荐）**
  https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
  进入页面后找到 *Pavia University scene* 一节。

- **直接下载链接**
  - http://www.ehu.es/ccwintco/uploads/e/ee/PaviaU.mat
  - http://www.ehu.es/ccwintco/uploads/7/72/PaviaU_gt.mat

### 1.3 放置路径

下载后按**下列目录结构**放置（与本仓库根目录同级）：

```
HybridSN-HSI-Classification/
├── HybridSN.ipynb
├── HybridSN.py
├── check_env.py
└── dataset/
    └── PaviaU/
        ├── PaviaU.mat          ← 放在这里
        └── PaviaU_gt.mat       ← 放在这里
```

> 若下载得到的是压缩包，请先解压，确保两个 `.mat` 文件的文件名与上述完全一致。

### 1.4 验证是否放置正确

在仓库根目录运行环境自检脚本：

```bash
python check_env.py
```

`[Dataset files]` 部分应显示两项 `[OK]`，并打印数据立方体信息：

```
[Dataset files]
  [OK  ] dataset\PaviaU\PaviaU.mat -> 33.2 MB
  [OK  ] dataset\PaviaU\PaviaU_gt.mat -> 0.0 MB
  cube shape : (610, 340, 103) uint16
  gt shape   : (610, 340) uint8
  classes    : 9 (excluding background 0)
  labeled px : 42776
  [OK  ] PaviaU readable
```

若显示 `[FAIL]` 或形状不符，请检查文件是否放置在 `dataset/PaviaU/` 下、文件名是否完全一致。

---

## 二、可选数据集（本项目未使用）

`HybridSN.py` 与 `HybridSN.ipynb` 中的 `DATASET` 变量支持另外两个常用高光谱数据集。
本项目的全部实验结果**仅在 PU 上完成**；若需运行其他数据集，请按下表放置：

| `DATASET` 取值 | 数据集 | 需要文件 | 放置路径 |
|---|---|---|---|
| `'PU'` | Pavia University | `PaviaU.mat`、`PaviaU_gt.mat` | `dataset/PaviaU/` |
| `'IP'` | Indian Pines | `Indian_pines_corrected.mat`、`Indian_pines_gt.mat` | `dataset/IndianPines/` |
| `'SA'` | Salinas | `Salinas_corrected.mat`、`Salinas_gt.mat` | `dataset/Salinas/` |

> 上述三个数据集均可从同一页面获取：
> https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

---

## 三、为什么不上传数据集

1. **体积**：PU 的影像文件约 33 MB，三个数据集合计约 100 MB，不宜随代码仓库分发；
2. **许可**：数据集由原发布方（Grupo de Inteligencia Computacional, Universidad del País Vasco）提供，
   请遵循其使用条款，从官方渠道获取；
3. **可复现性**：仓库已提供完整的下载地址与放置路径，任何人按本文说明即可还原实验数据环境。

---

## 四、数据引用

如使用本仓库代码发表工作，请引用数据集的原始来源：

> Grupo de Inteligencia Computacional (GIC).
> *Hyperspectral Remote Sensing Scenes.*
> Universidad del País Vasco (UPV/EHU).
> https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
