# Semiconductor Yield & Process Analytics Platform — Project Plan

**Version:** 1.0  
**Date:** 2026-05-04  
**Author:** Wang Handa  
**Status:** Planning

---

## 1. 项目目标 (Project Objectives)

### 业务目标

本项目模拟半导体制造数据工程师的两类核心工作：

1. **Wafer Map Defect Classification（晶圆图缺陷分类）**  
   在量产 fab 中，每片晶圆测试完成后会产生 wafer map，工程师需要快速判断缺陷图案属于哪种失效模式（如 scratch、ring、edge-loc 等），以便追溯工艺根因。本模块模拟这一流程，使用公开数据集 WM-811K 训练 CNN 分类器，并提供 Web Demo 展示预测结果。

2. **SPC + Process Anomaly Detection（统计过程控制 + 异常检测）**  
   fab 中的工艺参数（如刻蚀速率、薄膜厚度、温度）需要持续监控。SPC（Statistical Process Control）是 fab 标准工具；在参数复杂、相关性高的场景下，机器学习异常检测可以作为补充。本模块实现 Western Electric Rules + Isolation Forest/Autoencoder 双路检测，并构建 dashboard 展示 SPC chart 和异常事件。

### 工程目标

- 代码工程化：模块化 Python 包结构，非单一 notebook
- 可复现：使用 `requirements.txt` / `pyproject.toml`，固定随机种子，提供完整训练脚本
- 可测试：核心逻辑覆盖 pytest 单元测试
- 可部署：Streamlit Demo 可本地一键启动，提供 Dockerfile
- 专业文档：README 面向两类受众：半导体行业工程师（业务背景）& 软件工程师（工程能力）

### 目标岗位匹配

| 岗位 | 本项目覆盖能力 |
|------|--------------|
| Yield Engineer | Wafer map pattern 分类、缺陷失效模式理解 |
| Process Data Engineer | SPC pipeline、特征工程、数据质量处理 |
| Semiconductor Data Analyst | SPC chart、异常可视化 dashboard |
| Semiconductor AI Engineer | CNN 训练、类不平衡处理、模型评估 |
| Manufacturing Data Scientist | 异常检测、Isolation Forest、Autoencoder |
| 製造DXエンジニア | 工艺数据流、监控 dashboard、模型部署 |

---

## 2. 推荐目录结构

```
semiconductor-yield-analytics/
│
├── README.md
├── pyproject.toml               # 项目元数据 + 依赖
├── requirements.txt             # 锁定依赖版本
├── requirements-dev.txt         # 开发/测试依赖
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── docs/
│   ├── project_plan.md          # 本文件
│   ├── module_a_design.md       # Module A 详细设计
│   └── module_b_design.md       # Module B 详细设计
│
├── data/
│   ├── raw/                     # 原始数据（不入 git，.gitignore 排除）
│   │   ├── wm811k/              # WM-811K .pkl 文件
│   │   └── secom/               # UCI SECOM（可选）
│   ├── processed/               # 预处理后数据
│   │   ├── wafer_maps/          # 归一化后的 wafer map numpy arrays
│   │   └── spc/                 # 清洗后的工艺参数 CSV
│   └── synthetic/               # 模拟生成的 SPC 数据
│
├── src/
│   ├── __init__.py
│   │
│   ├── module_a/                # Wafer Map Defect Classification
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── dataset.py       # WM-811K 数据加载、解析
│   │   │   ├── preprocessing.py # 尺寸统一、归一化
│   │   │   └── augmentation.py  # 数据增强策略
│   │   ├── models/
│   │   │   ├── cnn_baseline.py  # ResNet-based CNN
│   │   │   └── model_factory.py # 模型注册/工厂函数
│   │   ├── training/
│   │   │   ├── trainer.py       # 训练循环、早停、checkpoint
│   │   │   ├── loss.py          # Focal Loss / class-weighted CE
│   │   │   └── metrics.py       # macro F1、confusion matrix、per-class recall
│   │   └── inference/
│   │       ├── predictor.py     # 单图/批量推理
│   │       └── explainability.py # Grad-CAM 可视化
│   │
│   ├── module_b/                # SPC + Process Anomaly Detection
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── generator.py     # 模拟工艺数据生成器
│   │   │   ├── loader.py        # SECOM / CSV 加载
│   │   │   └── preprocessing.py # 缺失值、标准化、特征选择
│   │   ├── spc/
│   │   │   ├── control_chart.py # X-bar、R、EWMA chart 计算
│   │   │   └── western_electric.py # 8 条 Western Electric Rules
│   │   ├── anomaly/
│   │   │   ├── isolation_forest.py
│   │   │   ├── autoencoder.py   # PyTorch Autoencoder
│   │   │   └── ensemble.py      # 多模型投票/融合
│   │   └── reporting/
│   │       └── alert.py         # 异常事件汇总、报告生成
│   │
│   └── shared/
│       ├── config.py            # 全局配置、路径常量
│       ├── logger.py            # 统一日志
│       └── utils.py             # 通用工具函数
│
├── scripts/
│   ├── download_data.py         # 数据下载脚本（WM-811K、SECOM）
│   ├── train_module_a.py        # Module A 训练入口
│   ├── evaluate_module_a.py     # Module A 评估入口
│   ├── run_module_b_pipeline.py # Module B 完整 pipeline 入口
│   └── generate_synthetic_data.py
│
├── app/
│   ├── main.py                  # Streamlit 入口
│   ├── pages/
│   │   ├── wafer_map_demo.py    # Module A Demo 页面
│   │   └── spc_dashboard.py     # Module B Dashboard 页面
│   └── components/
│       ├── wafer_map_viz.py     # wafer map 渲染组件
│       └── spc_chart.py         # SPC chart 渲染组件
│
├── notebooks/
│   ├── 01_eda_wafer_map.ipynb   # 数据探索（非生产代码，仅分析用）
│   ├── 02_eda_spc_data.ipynb
│   └── 03_model_comparison.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_western_electric.py
│   │   ├── test_preprocessing.py
│   │   ├── test_dataset.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_spc_pipeline.py
│   └── fixtures/
│       ├── sample_wafer_maps.npy
│       └── sample_spc_data.csv
│
├── configs/
│   ├── module_a.yaml            # 模型超参、训练配置
│   └── module_b.yaml            # SPC 参数、异常检测配置
│
└── outputs/
    ├── models/                  # 训练好的模型权重（.pt）
    ├── reports/                 # 评估报告、图表
    └── logs/                    # 训练日志
```

---

## 3. 每个模块的功能范围

### Module A: Wafer Map Defect Classification

#### 3.A.1 数据处理

- **数据集：** WM-811K（公开数据集，约 811,457 张 wafer map，来自 MiraCle 论文）
- **9 类缺陷：** Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Random, Scratch, None（无缺陷）
- **类不平衡问题：** None 类约占 79%，其余 8 类合计约 21%，需专项处理
- **预处理流程：**
  - 从 `.pkl` 文件解析 wafer map 矩阵和标签
  - 将所有 wafer map resize 到统一尺寸（如 64×64）
  - 编码缺陷类别为数值标签
  - 划分 train / val / test（按 patient ID 避免数据泄露）

#### 3.A.2 类不平衡处理策略

- Class-weighted Cross Entropy Loss（优先）
- Focal Loss（备选，对难样本加权）
- 过采样：对少数类使用 `WeightedRandomSampler`
- 数据增强：随机旋转（wafer map 具有旋转对称性）、随机翻转、轻微噪声注入

#### 3.A.3 模型

- **CNN Baseline：** 基于 ResNet-18（ImageNet pretrained，fine-tune 最后几层）
- **输入：** 单通道灰度图（wafer map 矩阵值 0/1/2 → 归一化为 float）
- **输出：** 9 类 softmax，输出类别概率
- **可选扩展：** EfficientNet-B0、Vision Transformer（仅作对比，不作主线）

#### 3.A.4 评估指标

- Macro F1（主指标，适合不平衡场景）
- Per-class Precision / Recall / F1
- Confusion Matrix（可视化）
- AUC-ROC（One-vs-Rest）
- **不使用 Accuracy 作为主指标**（因为类不平衡会使 Accuracy 虚高）

#### 3.A.5 可解释性

- Grad-CAM：对 CNN 最后一个卷积层生成热力图，叠加到 wafer map 上，直观展示模型关注区域

#### 3.A.6 Streamlit Web Demo

- 上传 wafer map 图片 / 随机抽样已有样本
- 显示 wafer map 原图 + Grad-CAM 热力图
- 显示预测类别 + 各类置信度条形图
- 显示该缺陷模式的工艺根因说明（静态文本）

---

### Module B: SPC + Process Anomaly Detection

#### 3.B.1 数据处理

- **主要数据：** 模拟生成（`generator.py`），模拟 fab 工艺参数时序数据
  - 参数示例：Etch Rate、Deposition Thickness、Chamber Temperature、Pressure、RF Power
  - 包含正常过程、均值漂移（mean shift）、方差增大（variance increase）、渐变趋势（trend）、周期性异常（tool effect）等注入的异常
- **可选补充数据：** UCI SECOM 数据集（1567 样本，590 特征，真实半导体工艺，有 pass/fail 标签）
- **预处理：**
  - 缺失值处理（SECOM 中缺失率高达 5-15%，用中值填充 + 标记缺失 flag）
  - Z-score 标准化
  - 特征相关性分析（移除相关系数 > 0.95 的冗余特征）

#### 3.B.2 SPC 实现

- **控制图类型：**
  - X-bar Chart（均值控制图）
  - R Chart（极差控制图）
  - EWMA Chart（指数加权移动平均，对小漂移更敏感）
- **Western Electric Rules（8 条规则）：**
  1. 单点超出 3σ 控制限
  2. 连续 9 点在中心线同侧
  3. 连续 6 点单调递增或递减
  4. 连续 14 点交替升降
  5. 连续 2/3 点超出 2σ 区域（同侧）
  6. 连续 4/5 点超出 1σ 区域（同侧）
  7. 连续 15 点在 ±1σ 内（过于稳定，可能仪器问题）
  8. 连续 8 点在 ±1σ 外（无点在中心区）
- 输出每个违规点的规则编号和描述

#### 3.B.3 机器学习异常检测

- **Isolation Forest：**
  - 使用 scikit-learn `IsolationForest`
  - 多变量输入（多参数联合检测）
  - 输出异常分数（anomaly score）
- **Autoencoder（PyTorch）：**
  - 多层 MLP encoder-decoder 结构
  - 基于重构误差（reconstruction error）检测异常
  - 阈值设定：以训练集 95th percentile 重构误差为基准
- **Ensemble：**
  - SPC violations + IF 异常分数 + AE 重构误差 三路输出
  - 可配置"多数投票"或"任一触发"策略
  - 输出最终 anomaly label + 各模型置信度

#### 3.B.4 Dashboard

- SPC 控制图：折线图 + 控制限线 + 违规点标注
- 异常时间轴：标注 ML 检测到的异常区间
- 参数相关性热力图
- 异常原因候选列表（基于规则，非 LLM 生成）
- 参数趋势对比图（支持多参数叠加）

---

## 4. 技术栈

### 核心依赖

| 类别 | 库 | 版本要求 | 用途 |
|------|----|---------|------|
| Python | Python | 3.11+ | 运行环境 |
| 数据处理 | pandas | 2.x | 表格数据、时序数据 |
| 数据处理 | numpy | 1.x / 2.x | 数组计算、矩阵操作 |
| 机器学习 | scikit-learn | 1.4+ | Isolation Forest、预处理、评估 |
| 深度学习 | PyTorch | 2.2+ | CNN、Autoencoder 训练 |
| 深度学习 | torchvision | 0.17+ | ResNet pretrained weights |
| 可视化 | plotly | 5.x | 交互式图表（SPC chart、dashboard）|
| 可视化 | matplotlib | 3.x | 静态图表、Grad-CAM 叠加图 |
| Web Demo | streamlit | 1.3x+ | 交互式 Web 界面 |
| 配置管理 | pyyaml | 6.x | 读取 YAML 配置文件 |
| 日志 | loguru | 0.7+ | 结构化日志 |
| 测试 | pytest | 7.x+ | 单元测试、集成测试 |
| 测试 | pytest-cov | 4.x | 代码覆盖率 |
| 容器化 | Docker | - | 环境打包、可复现部署 |

### 开发工具

| 工具 | 用途 |
|------|------|
| black | 代码格式化 |
| ruff | Linting |
| pre-commit | Git hooks 自动检查 |
| jupyter | EDA notebook |
| tqdm | 训练进度条 |

---

## 5. 数据目录设计

### 5.1 WM-811K 数据集

- **来源：** MiraCle 研究项目公开数据，[Kaggle: WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **原始文件：** `LSWMD.pkl`（约 350MB）
- **存放路径：** `data/raw/wm811k/LSWMD.pkl`
- **处理后：**
  - `data/processed/wafer_maps/train.npz`（wafer map 数组 + 标签）
  - `data/processed/wafer_maps/val.npz`
  - `data/processed/wafer_maps/test.npz`
  - `data/processed/wafer_maps/label_map.json`（类别名称映射）

### 5.2 模拟 SPC 数据

- **生成脚本：** `scripts/generate_synthetic_data.py`
- **参数设计：**
  - `n_samples`：默认 5000 个时间点
  - `n_features`：默认 8 个工艺参数
  - `anomaly_rate`：默认 5%
  - `anomaly_types`：`["mean_shift", "variance_increase", "trend", "spike"]`
  - `random_seed`：固定为 42（可复现）
- **存放路径：**
  - `data/synthetic/spc_normal.csv`
  - `data/synthetic/spc_with_anomalies.csv`（含注入异常 + ground truth 标签）

### 5.3 UCI SECOM 数据集（可选）

- **来源：** UCI Machine Learning Repository
- **原始文件：** `secom.data`（传感器数据）、`secom_labels.data`（pass/fail 标签）
- **存放路径：** `data/raw/secom/`
- **注意：** SECOM 数据集类不平衡（fail 约占 6.5%），需在文档中说明

### 5.4 .gitignore 策略

```
data/raw/          # 原始数据不入 git（体积大）
data/processed/    # 处理后数据不入 git（可由脚本重新生成）
data/synthetic/    # 模拟数据可选入 git（体积小）
outputs/models/    # 模型权重不入 git（体积大，用 DVC 或 HuggingFace 托管）
outputs/logs/      # 日志不入 git
```

---

## 6. 训练流程设计

### Module A 训练流程

```
1. 数据准备
   scripts/download_data.py --dataset wm811k
   → 下载 LSWMD.pkl 到 data/raw/wm811k/

2. 预处理
   scripts/train_module_a.py --stage preprocess
   → 解析 pkl，resize 到 64×64，划分 train/val/test
   → 输出 data/processed/wafer_maps/

3. 训练
   scripts/train_module_a.py --config configs/module_a.yaml
   → 加载配置（lr, batch_size, epochs, loss_type, model_arch）
   → 初始化模型（ResNet-18, pretrained=True, 修改第一层接受单通道）
   → 训练循环：
       for epoch in epochs:
           train one epoch with WeightedRandomSampler
           validate on val set
           log metrics: loss, macro_f1, per_class_f1
           save checkpoint if val_macro_f1 improves
   → 早停（patience=10）

4. 评估
   scripts/evaluate_module_a.py --checkpoint outputs/models/module_a_best.pt
   → 在 test set 上评估
   → 输出 outputs/reports/module_a_eval_report.json
   → 生成 confusion matrix 图：outputs/reports/confusion_matrix.png
   → 生成 per-class metrics 表格

5. 可解释性
   → 对每类缺陷抽取样本，生成 Grad-CAM 热力图
   → 输出 outputs/reports/gradcam/
```

### Module B 训练流程

```
1. 数据生成
   scripts/generate_synthetic_data.py --seed 42 --samples 5000
   → 输出 data/synthetic/spc_with_anomalies.csv

2. 特征工程
   scripts/run_module_b_pipeline.py --stage preprocess
   → 缺失值处理、标准化、相关性过滤
   → 输出 data/processed/spc/features_scaled.csv

3. SPC 分析
   scripts/run_module_b_pipeline.py --stage spc
   → 计算控制限（基于 training period 前 30%）
   → 应用 8 条 Western Electric Rules
   → 输出违规事件表：outputs/reports/spc_violations.csv

4. 异常检测模型训练
   scripts/run_module_b_pipeline.py --stage train_anomaly
   → 用正常数据（训练期，无注入异常）训练 IsolationForest
   → 用正常数据训练 Autoencoder（50 epochs）
   → 保存模型：outputs/models/isolation_forest.pkl, autoencoder.pt

5. 完整 pipeline 运行
   scripts/run_module_b_pipeline.py --stage full
   → 在全量数据上运行 SPC + ML 检测
   → 合并 ensemble 结果
   → 输出 outputs/reports/anomaly_events.csv

6. 评估（基于模拟数据的 ground truth）
   → Precision、Recall、F1（以注入异常 label 为 ground truth）
   → False Alarm Rate（误报率，在 fab 场景中重要）
```

---

## 7. Dashboard 页面设计

### 页面 1: Wafer Map Defect Classifier Demo

```
布局：两栏
左栏（输入控制）：
  - [按钮] 随机抽取测试样本
  - [文件上传] 上传自定义 wafer map（.png / .npy）
  - [下拉] 选择显示的真实标签类别
  - [复选框] 是否显示 Grad-CAM

右栏（输出展示）：
  - 原始 wafer map 图像（heatmap 渲染，0/1/2 三值）
  - Grad-CAM 叠加图（显示模型关注区域）
  - 预测标签 + 置信度（大字显示）
  - 9 类置信度条形图（plotly horizontal bar）
  - 缺陷类型说明卡片（描述该缺陷的工艺根因，静态文本）

底部（信息栏）：
  - 模型信息：架构、训练集大小、测试集 Macro F1
  - 数据来源说明：WM-811K public dataset
```

### 页面 2: SPC & Process Anomaly Dashboard

```
顶部（参数选择）：
  - [下拉多选] 选择要监控的工艺参数
  - [日期范围滑块] 选择时间窗口
  - [单选] 检测方法：SPC Only / ML Only / Ensemble

主图区（SPC Control Chart）：
  - 折线图：参数时序值
  - 上下控制限（UCL/LCL，3σ）：红色虚线
  - 警戒线（2σ）：橙色虚线
  - 违规点：红色标注 + hover 显示违规规则
  - EWMA 平滑曲线：蓝色叠加

右侧信息面板：
  - 当前统计：均值、标准差、CPK（过程能力指数）
  - Western Electric Rules 触发统计表
  - ML 异常分数趋势图（小图）

底部（异常事件列表）：
  - 表格：时间戳、参数名、异常类型、触发方式（SPC/IF/AE）、严重程度
  - [按钮] 导出 CSV

侧边栏（分析工具）：
  - 参数相关性热力图（选定时间窗口内）
  - 多参数趋势对比图
```

---

## 8. 测试计划

### 8.1 单元测试

| 测试文件 | 测试内容 |
|---------|---------|
| `test_western_electric.py` | 8 条规则各自的边界情况、正确触发、不触发 |
| `test_preprocessing.py` | 缺失值填充、标准化、resize 函数的输入输出形状 |
| `test_dataset.py` | WM-811K 数据加载、标签编码、划分比例 |
| `test_metrics.py` | macro_f1 计算、confusion matrix 形状、per-class precision/recall |

### 8.2 集成测试

| 测试文件 | 测试内容 |
|---------|---------|
| `test_training_pipeline.py` | 用 fixtures 中的小样本数据（10 张 wafer map），完整运行一次训练，验证输出文件存在、metrics 为合法值 |
| `test_spc_pipeline.py` | 用 fixtures 中的 sample_spc_data.csv，完整运行 SPC + ML pipeline，验证 violations 表格格式正确 |

### 8.3 测试覆盖率目标

- 核心 SPC / 规则判断逻辑：覆盖率 ≥ 90%
- 数据预处理函数：覆盖率 ≥ 80%
- 整体代码：覆盖率 ≥ 70%
- 运行命令：`pytest tests/ --cov=src --cov-report=html`

### 8.4 Fixtures 设计

- `sample_wafer_maps.npy`：10 张 64×64 wafer map（手动构造，覆盖 3-4 类）
- `sample_spc_data.csv`：200 行时序数据（含已知异常位置，用于验证检测结果）

---

## 9. README 结构

```markdown
# Semiconductor Yield & Process Analytics Platform

## Overview（业务背景 + 技术定位，3-4 段）
  - 项目要解决什么工程问题
  - 两个模块分别对应 fab 中的什么场景
  - 使用的数据来源声明（public dataset / simulated data）

## Architecture（架构图或目录结构说明）

## Module A: Wafer Map Defect Classification
  ### Problem Statement（业务问题描述）
  ### Dataset（WM-811K 说明、类别分布、类不平衡数据）
  ### Approach（方法：CNN + class weighting + data augmentation）
  ### Results（测试集 Macro F1、per-class metrics 表格、confusion matrix 图）
  ### Key Engineering Decisions（工程决策：为什么选 Macro F1 而非 Accuracy 等）

## Module B: SPC + Process Anomaly Detection
  ### Problem Statement
  ### Data（模拟数据生成策略说明）
  ### SPC Implementation（Western Electric Rules 说明）
  ### ML Anomaly Detection（Isolation Forest + Autoencoder 方法说明）
  ### Results（在模拟数据上的 Precision/Recall/False Alarm Rate）
  ### Real-World Considerations（重要章节，见下）

## Real-World Considerations（面试加分项章节）
  - Recipe Drift：如何处理工艺配方漂移导致的基准偏移
  - Tool-to-Tool Variation：如何处理不同机台间的系统性偏差
  - Domain Shift：模型在新产品线上的泛化问题
  - False Alarm Cost：fab 中误报成本远高于漏报，如何调整阈值
  - Model Monitoring：生产环境中如何监控模型退化

## Getting Started
  ### Prerequisites
  ### Installation
  ### Download Data
  ### Run Training
  ### Launch Dashboard

## Docker
  ### Build & Run

## Testing
  ### Run Tests
  ### Coverage Report

## Limitations & Future Work
  - 明确说明：本项目使用 public dataset 和 simulated data，结果不代表真实 fab 性能
  - 未来可扩展方向

## License
```

---

## 10. 预计实现顺序

### Phase 1: 基础框架搭建（第 1-2 天）

```
Step 1.1  创建完整目录结构（空目录 + __init__.py）
Step 1.2  写 pyproject.toml / requirements.txt
Step 1.3  写 src/shared/config.py（路径常量、全局配置）
Step 1.4  写 src/shared/logger.py
Step 1.5  写 .gitignore、Dockerfile 框架
```

### Phase 2: Module B 数据生成 & SPC（第 3-4 天）

```
Step 2.1  写 src/module_b/data/generator.py（模拟数据生成，不依赖外部数据）
Step 2.2  写 scripts/generate_synthetic_data.py
Step 2.3  写 src/module_b/spc/control_chart.py（X-bar、R、EWMA）
Step 2.4  写 src/module_b/spc/western_electric.py（8 条规则）
Step 2.5  写 tests/unit/test_western_electric.py（先写测试，确保规则正确）
Step 2.6  验证：运行测试，确保通过
```

### Phase 3: Module B 异常检测（第 5-6 天）

```
Step 3.1  写 src/module_b/data/preprocessing.py
Step 3.2  写 src/module_b/anomaly/isolation_forest.py
Step 3.3  写 src/module_b/anomaly/autoencoder.py（PyTorch）
Step 3.4  写 src/module_b/anomaly/ensemble.py
Step 3.5  写 scripts/run_module_b_pipeline.py
Step 3.6  写 tests/integration/test_spc_pipeline.py
Step 3.7  端到端跑通 Module B，生成 anomaly_events.csv
```

### Phase 4: Module A 数据处理（第 7-8 天）

```
Step 4.1  下载 WM-811K 数据集（手动，见第 11 节）
Step 4.2  写 src/module_a/data/dataset.py（解析 pkl）
Step 4.3  写 src/module_a/data/preprocessing.py（resize、归一化）
Step 4.4  写 src/module_a/data/augmentation.py
Step 4.5  写 tests/unit/test_dataset.py、test_preprocessing.py
Step 4.6  做 EDA：notebooks/01_eda_wafer_map.ipynb（类别分布、样本可视化）
```

### Phase 5: Module A 模型训练（第 9-11 天）

```
Step 5.1  写 src/module_a/models/cnn_baseline.py（ResNet-18 改单通道）
Step 5.2  写 src/module_a/training/loss.py（Focal Loss + class-weighted CE）
Step 5.3  写 src/module_a/training/metrics.py
Step 5.4  写 src/module_a/training/trainer.py（训练循环、checkpoint）
Step 5.5  写 scripts/train_module_a.py
Step 5.6  运行训练（预计 2-4 小时，视 GPU 情况）
Step 5.7  写 scripts/evaluate_module_a.py
Step 5.8  生成评估报告、confusion matrix
Step 5.9  写 src/module_a/inference/explainability.py（Grad-CAM）
```

### Phase 6: Streamlit Dashboard（第 12-13 天）

```
Step 6.1  写 app/components/wafer_map_viz.py
Step 6.2  写 app/components/spc_chart.py（plotly SPC chart 组件）
Step 6.3  写 app/pages/wafer_map_demo.py（Module A Demo 页面）
Step 6.4  写 app/pages/spc_dashboard.py（Module B Dashboard 页面）
Step 6.5  写 app/main.py（Streamlit 多页面入口）
Step 6.6  本地测试 Dashboard，检查所有交互功能
```

### Phase 7: 测试完善 & 文档（第 14-15 天）

```
Step 7.1  补全所有单元测试，运行 pytest --cov，确保覆盖率达标
Step 7.2  写 tests/integration/ 集成测试
Step 7.3  写 Dockerfile + docker-compose.yml，测试 Docker 构建
Step 7.4  写完整 README.md
Step 7.5  补全 docs/module_a_design.md、docs/module_b_design.md
Step 7.6  最终 review：检查所有 hardcoded path、TODO、调试代码
Step 7.7  git 整理：squash 杂乱 commits，写清晰 commit message
```

---

## 11. 哪些地方需要手动准备数据

### 必须手动下载

| 数据集 | 下载地址 | 操作 | 存放路径 |
|--------|---------|------|---------|
| WM-811K | https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map | 需要 Kaggle 账号，下载 `LSWMD.pkl`（约 350MB） | `data/raw/wm811k/LSWMD.pkl` |

**操作步骤：**
1. 登录 Kaggle → 下载 `LSWMD.pkl`
2. 将文件放到 `data/raw/wm811k/LSWMD.pkl`
3. 运行 `scripts/train_module_a.py --stage preprocess` 生成处理后数据

### 可自动生成（无需手动准备）

| 数据 | 生成方式 |
|------|---------|
| 模拟 SPC 数据 | `scripts/generate_synthetic_data.py` 自动生成 |
| 测试 fixtures | 由 `scripts/generate_synthetic_data.py --mode fixtures` 生成 |

### 可选下载（非必须）

| 数据集 | 下载地址 | 说明 |
|--------|---------|------|
| UCI SECOM | https://archive.ics.uci.edu/dataset/179/secom | 真实半导体工艺数据，作为 Module B 的备选数据源 |

---

## 12. 哪些内容不能夸大写进简历

### 绝对不能写的内容

| 错误写法 | 原因 |
|---------|------|
| "提高生产良率 X%" | 本项目无真实 fab 数据，没有生产良率可言 |
| "部署于 XXX 晶圆厂" | 这是个人项目，未经过任何生产验证 |
| "检测出 YYY 故障，避免损失 ZZZ 万元" | 完全无真实数据支撑，属于捏造 |
| "减少 XX% 误报" | 只有模拟数据验证，不代表真实场景表现 |
| "处理了 XX 万片晶圆的数据" | WM-811K 是公开数据集，非生产数据 |

### 可以合理写的内容

| 正确写法 | 说明 |
|---------|------|
| "在 WM-811K 公开数据集（811K 样本）上训练 CNN，测试集 Macro F1 达到 X.XX" | 真实可验证结果 |
| "实现了 Western Electric 8 条控制规则，并集成 Isolation Forest + Autoencoder 异常检测" | 工程实现，事实陈述 |
| "构建了模拟半导体工艺数据的 SPC 监控 Pipeline，包含 X 种异常注入场景" | 明确说明模拟数据 |
| "使用 PyTorch 实现多层 Autoencoder，在模拟数据上 Anomaly F1 达到 X.XX" | 明确限定在模拟数据上 |
| "构建了 Streamlit 交互 Demo，支持 Wafer Map 上传、CNN 预测和 Grad-CAM 可视化" | 功能描述，可 demo 验证 |
| "处理了 9 类缺陷模式的严重类不平衡问题（最大类占比 79%）" | 技术挑战描述 |

### 简历措辞建议

- 始终在项目简介第一句写明："使用公开数据集（WM-811K）和模拟工艺数据构建的全栈数据工程项目"
- 用"in simulation"或"on public dataset"限定所有量化结果
- 强调工程能力：模块化设计、测试覆盖、Docker 部署、可复现流程
- 强调领域知识：Western Electric Rules、Wafer Map Pattern、CPK、SPC 的业务背景理解

---

## 附录：关键设计决策说明

### 为什么用 ResNet-18 而非更复杂模型

- ResNet-18 参数量适中，在有限计算资源上可完整训练
- Fine-tune pretrained weights 利用 ImageNet 的低级特征
- 本项目目的是展示工程流程，而非刷 SOTA

### 为什么 Western Electric Rules 比纯 ML 更重要

- 在 fab 中，SPC 是 SEMI/ISO 标准要求的工具，工程师必须理解规则
- ML 方法作为辅助，不能替代可解释的统计方法
- 体现对行业标准的理解是面试加分项

### 为什么用模拟数据而非 SECOM 作为主数据

- SECOM 有 590 个特征，但只有 1567 个样本，对展示工程 Pipeline 不够友好
- 模拟数据可以精确控制异常类型和位置，方便定量评估检测器性能
- 模拟数据生成器本身也是展示数据工程能力的方式

### 关于 SECOM 的使用方式

- SECOM 作为 Module B 的可选补充数据，展示"可以处理真实数据"的能力
- 若使用 SECOM，在 README 和报告中明确说明其来源和局限性（样本量小、高缺失率）
