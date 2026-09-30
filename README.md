# Kaggle_House_Prices

_本项目为完整工程化落地练习项目，除方法论与代码方法查询外尽量不使用AI辅助。目前项目进度约90%。_

_当前项目尚未完成部分：最终演示demo_

本项目基于 Kaggle **House Prices: Advanced Regression Techniques** 数据集，目标是完整实践一次从原始表格数据到最终预测提交的监督学习回归流程。项目重点并不只是追求排行榜分数，而是系统练习数据清洗、特征预处理、特征工程、特征选择、交叉验证、Pipeline 构建以及代码模块化等数据科学基本工作流。

在数据处理阶段，项目针对缺失值的不同语义分别进行处理，并区分数值变量、名义类别变量和具有顺序关系的类别变量。其中，名义变量使用 One-Hot Encoding，数值变量使用 RobustScaler；同时将 SalePrice 进行自然对数变换，以减弱目标变量明显的右偏分布。

建模阶段以 Linear Regression 作为基础模型，并设计了三组实验进行比较：

- **Set A — Baseline:** 经过清洗和预处理后的全部原始特征直接进入线性回归；
- **Set B — BIC Feature Selection:** 在预处理后使用基于 BIC 的 backward selection 进行特征筛选；
- **Set C — Feature Engineering + BIC Selection:** 在原始特征基础上构造房龄、总使用面积、等价浴室数量、质量交互项等人工特征，再进行 BIC 后向选择。

所有模型均使用相同的 5-fold cross-validation 进行评估，并比较 RMSE、MAE 和 R² 等指标。实验结果表明，复杂度更高的方案并不一定具有更好的预测表现。Baseline 模型取得了最好的验证结果，平均 validation RMSE 为 **0.1540**，而加入 BIC 后向选择后的模型虽然显著减少了特征数量，但 RMSE 和 R² 均有所下降。人工特征工程能够略微改善经过 BIC 筛选后的模型，但仍未超过 Baseline。

最终使用全部训练数据重新训练 Baseline Pipeline，并对 Kaggle 测试集进行预测，获得 **0.14583** 的 Public Leaderboard Score，与本地交叉验证结果保持较好的一致性。

通过该项目，我不仅完成了一次完整的 Kaggle 回归任务，也重点实践了如何避免 preprocessing leakage、如何将自定义 Transformer 集成到 sklearn Pipeline，以及如何在模型复杂度、特征选择和实际泛化性能之间进行比较。

项目结构
```
Housing_Prices/
├── data
│   ├── processed/
│   └── raw/
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── notebooks/
│   ├── 01_Exploration.ipynb
│   ├── 02_Cleaning_and_preprocesing.ipynb
│   ├── 03_Feature Engineering.ipynb
│   ├── 04_Modelling_baseline.ipynb
│   ├── 05_Modelling_feature_engineered.ipynb
│   ├── 06_predicting.ipynb
│   └── 07_Final_Demo.ipynb
├── outputs/
│   └── submission.csv
├── scripts/
└── src/
    ├── cleaning.py
    ├── feature_engineering.py
    ├── feature_selection.py
    ├── preprocessing.py
    └── evaluation.py
```

其中，01-06 Jupyter Notebook都是探索过程，内容比较混乱，清晰的实验和完整流程在`07_Final_Demo.ipynb`（即将完成），`07_Final_Demo.ipynb`没有细节实现，而是直接调用由探索过程中固定到`.py`中的代码。

.py文件包括：

- `cleaning.py` — 数据清洗与缺失值处理
- `preprocessing.py` — 数值缩放与类别编码
- `feature_engineering.py` — sklearn-compatible 特征工程 Transformer
- `feature_selection.py` — BIC 计算与 backward feature selection
- `evaluation.py` — 交叉验证与模型结果汇总


有关项目更详细的介绍和说明可以在个人主页查看
```
https://lihao-academic.github.io/portfolio/
```





