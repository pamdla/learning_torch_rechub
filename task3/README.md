# Day3 Ranking Model 学习总结

## 概述

这是使用 **torch-rechub** 学习 Ranking Model（排序模型）的笔记，主要实现了经典的 **Wide & Deep (W&D)** 模型。

### 模型说明

| 特性 | Wide 部分 | Deep 部分 |
| --- | --- | --- |
|主要模型 | 线性回归 (Linear Layer) | 深度神经网络 (MLP) |
|输入特征 | 稀疏特征、交叉组合特征 | 连续特征、Embedding 向量 |
|核心优势 | 记忆力：强行记住有效的特征组合 |  泛化力：挖掘特征间的深层潜在关系 |
|优化目标 | 捕捉强相关的特殊偏好  | 提升对冷门或新特征的覆盖度 |

> Wide部分负责“死记硬背”那些极其重要的特征组合，而Deep部分负责“举一反三”去推测潜在的兴趣。
> Wide部分的缺点：无法处理在训练集中从未出现过的特征组合。

## 环境

- torch 2.6.0 (CPU版本)
- torch-rechub 0.3.0

## 数据集

使用 **Criteo** 广告点击数据集样本：
- **样本数**: 115条
- **特征数**: 40列
  - **Dense特征 (13个)**: I1-I13，数值型特征
  - **Sparse特征 (26个)**: C1-C26，类别型特征（哈希后的字符串）
- **标签**: 点击与否（0/1）

## 数据预处理流程

1. **缺失值填充**
   - Sparse特征: 填充为 `'0'`
   - Dense特征: 填充为 `0`

2. **离散化处理**
   - 将dense特征转换为分类特征
   - 转换公式: `int(log(v)^2)` (当 v > 2) 或 `v - 2` (当 v ≤ 2)
   - 新特征命名为 `I*_cat` 格式

3. **归一化**
   - 使用 MinMaxScaler 对dense特征进行[0,1]归一化

4. **编码**
   - 使用 LabelEncoder 对sparse特征进行数值编码

## 特征定义

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature

dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]
```

## 模型: Wide & Deep (W&D)

Wide & Deep 是Google提出的经典排序模型，结合了：
- **Wide部分**: 线性模型，处理dense特征，负责"记忆"能力
- **Deep部分**: DNN模型，处理sparse特征，负责"泛化"能力

### 模型配置

```python
from torch_rechub.models.ranking import WideDeep

model = WideDeep(
    wide_features=dense_feas,
    deep_features=sparse_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

## 训练配置

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cpu"
)
```

## 训练结果

- 训练轮数: 10 epochs
- Batch size: 256
- 验证AUC: ~0.54 (因数据量较小，结果不稳定)

## 实验记录

### 问题发现

初始训练后AUC只有约0.54，基本接近随机猜测，分析原因：
1. **数据太小**: 一共只有115条记录，训练集只有70%×115 = 80条
2. **模型参数选择不合适**
3. **W&D可能不适合这个小数据集场景**

### 调参试验

定义了 `make_trials` 函数进行多次参数试验：

```python
def make_trials(dims=[36, 128, 64], lr=1e-3, n_epoch=10, activation='relu'):
    model = WideDeep(
        wide_features=dense_feas,
        deep_features=sparse_feas,
        mlp_params={"dims": dims, "dropout": 0.2, "activation": activation}
    )
    trainer = CTRTrainer(
        model=model,
        optimizer_params={"lr": lr, "weight_decay": 1e-5},
        n_epoch=n_epoch,
        device="cpu"
    )
    trainer.fit(train_dl, val_dl)
    auc = trainer.evaluate(trainer.model, test_dl)
    return model, trainer, auc
```

> 注意：这里wide部分和deep部分可能要交换一下！

### 试验结果

| 试验次数 | dims | lr | n_epoch | activation | 测试AUC |
|---------|------|-----|---------|------------|--------|
| 1 | [36,128,64] | 1e-3 | 10 | relu | ~0.75 |
| 2 | [36,128,64] | 1e-3 | 10 | relu | ~0.81 |
| 3 | [36,128,64] | 1e-3 | 10 | relu | ~0.54 |
| 4 | [36,128,64] | 1e-3 | 10 | relu | 0.725 |

### 结论

- **随机性大**: 同样的参数多次运行结果差异明显（AUC在0.54~0.81之间波动）
- **数据量是关键**: 115条样本对于深度学习模型来说严重不足
- **需要固化随机种子**: 注释"调参不易, 随机种子需要再进一步固化"说明需要固定随机种子以获得可复现结果

---

## 小数据集 + Wide&Deep 模型注意事项和体会

**小数据集场景下使用Wide&Deep模型的建议**：

1. **数据量是瓶颈**: 深度学习模型需要大量数据才能发挥优势，小数据集（<1000条）容易导致过拟合或欠拟合
2. **固化随机种子**: 必须设置random seed确保实验可复现，否则结果波动大
3. **考虑简化模型**: 可以尝试减少MLP层数或embedding维度，降低模型复杂度
4. **正则化很重要**: 适当增大dropout权重_decay防止过拟合
5. **可能更适合传统ML**: 对于小数据集，传统机器学习（如LR、XGBoost）可能效果更好，深度学习优势不明显

## 关键代码位置

- Notebook: `Day3-ranking-model.ipynb`
- 模型定义: 第2389-2393行
- 训练器: 第2420-2425行
- make_trials函数: 第2855行 / 第5189行

## 要点

1. **特征类型区分**: Ranking模型需要区分dense（数值）和sparse（类别）特征
2. **特征工程**: 离散化和归一化对模型效果很重要
3. **Embedding**: Sparse特征需要通过embedding转换为低维向量
4. **W&D优势**: 兼顾记忆能力（Wide）和泛化能力（Deep）
